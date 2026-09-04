"""Tests for async_api_ops/concurrency.py.

Two things here are worth pinning down. One is the arithmetic that turns "how much memory is
free" into "how many requests may be in flight", which is easy to change in a way that looks
reasonable and quietly produces a limit of 4 or of 4000. The other is the gate itself, whose
awkward case is a limit that shrinks while tasks are queued behind it - the point at which a
lost wakeup would hang a run for good.

Memory probing is stubbed for all of that: a test that asked the machine how much RAM it had
would assert different things on different machines. The exception is
MemoryProbeContractTests, which runs against the real probe on purpose - the platform
underneath is exactly where those go wrong, so a fake would only prove the fake works.

Not covered here: base_ops wiring the gate's ceiling to the connection pool size. That call
needs aqt. The gate's half of it - reporting the ceiling when it moves - is covered by
CeilingReportingTests.
"""

import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from addon_modules import load_addon_module  # type: ignore

conc = load_addon_module("concurrency")

MB = conc.MB
GB = 1024 * MB


class StubMemory:
    """Stands in for the platform memory probes.

    `total`/`available` answer system_memory; `rss` answers process_memory and can be moved
    during a test to simulate the process growing or shrinking.
    """

    def __init__(self, total: int = 32 * GB, available: int = 16 * GB, rss: int = 500 * MB):
        self.total = total
        self.available = available
        self.rss = rss
        self.probe_failed = False

    def system_memory(self):
        return None if self.probe_failed else (self.total, self.available)

    def process_memory(self):
        return None if self.probe_failed else self.rss


# The real functions, captured once at import. A test that wants one of them back must take
# it from here rather than reading it off the module, which by then holds a stub.
REAL_SYSTEM_MEMORY = conc.system_memory
REAL_PROCESS_MEMORY = conc.process_memory
REAL_LOAD_PER_TASK_ESTIMATES = conc.load_per_task_estimates


def install_memory_stubs(memory: StubMemory) -> None:
    conc.system_memory = memory.system_memory
    conc.process_memory = memory.process_memory
    # Never read the real user_files estimates: they differ per machine and per run
    conc.load_per_task_estimates = lambda: {}


def restore_memory_probes() -> None:
    conc.system_memory = REAL_SYSTEM_MEMORY
    conc.process_memory = REAL_PROCESS_MEMORY
    conc.load_per_task_estimates = REAL_LOAD_PER_TASK_ESTIMATES


class MemoryStubTestCase(unittest.TestCase):
    """Replaces the memory probes and the estimates file for the duration of a test."""

    def setUp(self):
        self.memory = StubMemory()
        install_memory_stubs(self.memory)

    def tearDown(self):
        restore_memory_probes()


# --- Reading the machine -------------------------------------------------------------------


class FormatBytesTests(unittest.TestCase):
    def test_units(self):
        self.assertEqual(conc.format_bytes(512 * MB), "512 MB")
        self.assertEqual(conc.format_bytes(2 * GB), "2.0 GB")
        self.assertEqual(conc.format_bytes(None), "?")


class MemoryProbeTests(unittest.TestCase):
    def test_the_reserve_is_a_share_of_total_memory_with_a_floor(self):
        # A tenth of a big machine
        self.assertEqual(conc.memory_reserve(32 * GB), int(32 * GB * conc.RESERVE_TOTAL_FRACTION))
        # ...but never less than the floor, or a small machine would be left with nothing
        self.assertEqual(conc.memory_reserve(1 * GB), conc.MIN_RESERVE_BYTES)

    def test_the_probes_report_this_machine(self):
        # Whatever platform the suite runs on, the probes must work: everything below is
        # stubbed, so this is the one check that the real ones are wired up
        memory = conc.system_memory()
        self.assertIsNotNone(memory, "system memory probe failed on this platform")
        total, available = memory
        self.assertGreater(total, 0)
        self.assertGreater(available, 0)
        self.assertLessEqual(available, total)
        rss = conc.process_memory()
        self.assertIsNotNone(rss, "process memory probe failed on this platform")
        self.assertGreater(rss, 0)

    def test_the_probes_report_bytes(self):
        # Every one of these reads a source with its own units - kilobytes from /proc, pages
        # from vm_stat - so a missing multiplication is the easy mistake, and magnitude alone
        # catches it: a machine with under a gigabyte of RAM will not be running Anki, and a
        # Python process holding under 5MB does not exist
        total, _ = conc.system_memory()
        self.assertGreater(total, 1 * GB, "system memory looks like kilobytes, not bytes")
        rss = conc.process_memory()
        self.assertGreater(rss, 5 * MB, "process memory looks like kilobytes, not bytes")
        self.assertLess(rss, total, "process memory is larger than the machine")


class MemoryBudgetTests(MemoryStubTestCase):
    def test_the_reserve_comes_off_before_anything_is_budgeted(self):
        # Budgeting from raw available memory would plan a limit that immediately triggers the
        # pressure response it is supposed to stay clear of
        self.memory.total = 32 * GB
        self.memory.available = 16 * GB
        reserve = conc.memory_reserve(32 * GB)
        spendable = (16 * GB - reserve) * conc.MEMORY_TARGET_FRACTION
        self.assertEqual(conc.memory_budget(), min(spendable, 32 * GB * conc.MEMORY_TOTAL_FRACTION))

    def test_the_budget_is_capped_as_a_share_of_total_memory(self):
        # Free memory right after a reboot is not an invitation to use all of it
        self.memory.total = 8 * GB
        self.memory.available = 8 * GB
        self.assertEqual(conc.memory_budget(), 8 * GB * conc.MEMORY_TOTAL_FRACTION)

    def test_no_budget_when_less_is_free_than_the_reserve(self):
        self.memory.total = 32 * GB
        self.memory.available = 1 * GB
        self.assertEqual(conc.memory_budget(), 0)

    def test_no_budget_when_the_probe_is_unavailable(self):
        self.memory.probe_failed = True
        self.assertIsNone(conc.memory_budget())


class MemoryLimitConfigTests(MemoryStubTestCase):
    def test_a_numeric_memory_limit_is_interpreted_as_megabytes(self):
        self.assertEqual(
            conc.configured_memory_limit_bytes({"memory_limit": 900}, 32 * GB), 900 * MB
        )

    def test_a_percent_memory_limit_is_interpreted_as_a_share_of_total_ram(self):
        self.assertEqual(
            conc.configured_memory_limit_bytes({"memory_limit": "12.5%"}, 32 * GB), 4 * GB
        )

    def test_percent_limits_need_total_memory_to_be_known(self):
        self.assertEqual(conc.configured_memory_limit_bytes({"memory_limit": "25%"}, 0), 0)

    def test_invalid_limits_are_treated_as_disabled(self):
        self.assertEqual(conc.configured_memory_limit_bytes({"memory_limit": "oops"}, 32 * GB), 0)
        self.assertEqual(conc.configured_memory_limit_bytes({"memory_limit": "0%"}, 32 * GB), 0)
        self.assertEqual(conc.configured_memory_limit_bytes({"memory_limit": -1}, 32 * GB), 0)

    def test_invalid_limits_are_logged(self):
        with mock.patch.object(conc.logger, "warning") as warning:
            self.assertEqual(conc.configured_memory_limit_bytes({"memory_limit": "oops"}, 32 * GB), 0)
        self.assertTrue(warning.called)


class MaxConcurrencyTests(MemoryStubTestCase):
    def test_the_ceiling_is_the_budget_divided_by_what_a_slot_costs(self):
        self.memory.total = 8 * GB
        self.memory.available = 8 * GB  # budget = 2GB
        self.assertEqual(conc.max_concurrency_for(16 * MB), 2 * GB // (16 * MB * 4))

    def test_a_slot_is_budgeted_for_the_queue_behind_it_too(self):
        # A limit of N keeps N * TASK_QUEUE_DEPTH tasks alive, all holding their note and
        # prompt. Budgeting a slot at one task's cost plans a window the budget can't hold.
        self.memory.total = 8 * GB
        self.memory.available = 8 * GB
        self.assertEqual(conc.memory_per_slot(16 * MB), 16 * MB * conc.TASK_QUEUE_DEPTH)
        self.assertEqual(
            conc.max_concurrency_for(16 * MB) * conc.TASK_QUEUE_DEPTH * 16 * MB,
            conc.memory_budget(),
        )

    def test_a_cheap_task_does_not_lift_the_ceiling_without_limit(self):
        self.assertEqual(conc.max_concurrency_for(1), conc.MAX_AUTO_CONCURRENCY)

    def test_a_configured_maximum_takes_the_place_of_the_backstop(self):
        # The 256 is only there so an unconfigured run stays sane; someone who names a number
        # has decided for themselves, above it as well as below
        self.memory.total = 32 * GB
        self.memory.available = 32 * GB
        self.assertEqual(conc.max_concurrency_for(1, 1000), 1000)
        self.assertEqual(conc.max_concurrency_for(1, 12), 12)

    def test_memory_still_caps_a_configured_maximum(self):
        # It raises what a run may grow to, not what it gets
        self.memory.total = 8 * GB
        self.memory.available = 8 * GB  # budget = 2GB
        self.assertEqual(conc.max_concurrency_for(16 * MB, 1000), 2 * GB // (16 * MB * 4))

    def test_an_expensive_task_still_leaves_a_workable_floor(self):
        self.assertEqual(conc.max_concurrency_for(100 * GB), conc.MIN_AUTO_CONCURRENCY)

    def test_a_conservative_ceiling_when_memory_cannot_be_probed(self):
        self.memory.probe_failed = True
        self.assertEqual(conc.max_concurrency_for(8 * MB), conc.NO_PROBE_CONCURRENCY)

    def test_the_thread_pool_is_sized_for_the_highest_reachable_limit(self):
        # The pool is created before the gate, so it has to cover a ceiling that may be raised
        # later once the op has been measured
        self.assertEqual(conc.max_possible_concurrency({}), conc.MAX_AUTO_CONCURRENCY)
        self.assertEqual(conc.max_possible_concurrency({"max_concurrent_requests": 400}), 400)
        self.assertEqual(conc.max_possible_concurrency(None), conc.MAX_AUTO_CONCURRENCY)


class ConcurrencyLimitsTests(MemoryStubTestCase):
    def test_a_run_starts_below_its_ceiling_and_adapts_upward(self):
        start, ceiling, adaptive = conc.concurrency_limits({}, 8 * MB)
        self.assertTrue(adaptive)
        self.assertEqual(start, conc.ADAPTIVE_START_CONCURRENCY)
        self.assertGreater(ceiling, start)

    def test_a_low_ceiling_also_lowers_the_starting_limit(self):
        self.memory.total = 2 * GB
        self.memory.available = 2 * GB
        start, ceiling, adaptive = conc.concurrency_limits({}, 64 * MB)
        self.assertTrue(adaptive)
        self.assertEqual(start, ceiling)

    def test_a_configured_maximum_lowers_the_ceiling_but_keeps_adapting(self):
        # The memory-pressure response underneath is what protects the machine, so a
        # configured limit must not switch it off
        start, ceiling, adaptive = conc.concurrency_limits({"max_concurrent_requests": 6}, 8 * MB)
        self.assertTrue(adaptive)
        self.assertEqual(ceiling, 6)
        self.assertEqual(start, 6)

    def test_a_configured_maximum_above_the_backstop_is_honoured(self):
        self.memory.total = 32 * GB
        self.memory.available = 32 * GB
        _, ceiling, adaptive = conc.concurrency_limits({"max_concurrent_requests": 1000}, 1)
        self.assertTrue(adaptive)
        self.assertEqual(ceiling, 1000)

    def test_without_a_probe_the_limit_is_static(self):
        self.memory.probe_failed = True
        self.assertEqual(
            conc.concurrency_limits({}, 8 * MB),
            (conc.NO_PROBE_CONCURRENCY, conc.NO_PROBE_CONCURRENCY, False),
        )
        self.assertEqual(
            conc.concurrency_limits({"max_concurrent_requests": 3}, 8 * MB), (3, 3, False)
        )


# --- Learning what an op costs ---------------------------------------------------------------


class MemoryEstimatorTests(MemoryStubTestCase):
    def measure(self, baseline: int, peak: int, in_flight: int, tasks: int = 0):
        """Run one window: baseline at the start, `peak` reached with `in_flight` tasks.

        `tasks` is the whole window, the queued ones included, as the bulk ops report it.
        """
        estimator = conc.MemoryEstimator("op")
        self.memory.rss = baseline
        estimator.begin_window()
        if tasks:
            estimator.note_tasks(tasks)
        self.memory.rss = peak
        estimator.sample(self.memory.rss, in_flight)
        return estimator, estimator.end_window()

    def test_growth_is_attributed_across_the_tasks_that_caused_it(self):
        _, per_task = self.measure(baseline=500 * MB, peak=540 * MB, in_flight=10)
        self.assertEqual(per_task, 4 * MB)

    def test_the_queued_tasks_are_charged_for_their_own_memory(self):
        # A window is TASK_QUEUE_DEPTH times the limit in tasks and every one of them holds a
        # note and a prompt from the moment it is created. Dividing the window's growth by the
        # ten holding a slot said each task cost four times what it does.
        _, per_task = self.measure(baseline=500 * MB, peak=540 * MB, in_flight=10, tasks=40)
        self.assertEqual(per_task, 1 * MB)

    def test_a_window_smaller_than_the_queue_is_measured_as_it_ran(self):
        # The last window of a run holds whatever notes were left, not a full queue's worth
        _, per_task = self.measure(baseline=500 * MB, peak=540 * MB, in_flight=10, tasks=16)
        self.assertEqual(per_task, 2.5 * MB)

    def test_a_window_never_counts_fewer_tasks_than_ran_at_once(self):
        # Whatever the caller reports, a task holding a slot is a live task
        _, per_task = self.measure(baseline=500 * MB, peak=540 * MB, in_flight=10, tasks=4)
        self.assertEqual(per_task, 4 * MB)

    def test_a_window_that_never_ran_enough_at_once_teaches_nothing(self):
        # One task's growth says nothing about what running fifty would cost
        _, per_task = self.measure(baseline=500 * MB, peak=540 * MB, in_flight=1)
        self.assertIsNone(per_task)

    def test_reused_memory_teaches_nothing(self):
        # The allocator already held the pages, so RSS did not move
        _, per_task = self.measure(baseline=500 * MB, peak=500 * MB, in_flight=10)
        self.assertIsNone(per_task)

    def test_an_implausibly_large_measurement_is_clamped(self):
        _, per_task = self.measure(baseline=0, peak=8 * GB, in_flight=2)
        self.assertEqual(per_task, conc.MAX_PER_TASK_MEMORY)

    def test_an_implausibly_small_measurement_is_clamped(self):
        # Otherwise a near-zero cost would divide the budget into a limit of millions
        _, per_task = self.measure(baseline=500 * MB, peak=500 * MB + 100, in_flight=50)
        self.assertEqual(per_task, conc.MIN_PER_TASK_MEMORY)

    def test_within_a_run_the_largest_window_wins(self):
        # What has to fit in RAM is the peak, not the average
        estimator = conc.MemoryEstimator("op")
        self.memory.rss = 500 * MB
        estimator.begin_window()
        self.memory.rss = 540 * MB
        estimator.sample(self.memory.rss, 10)
        self.assertEqual(estimator.end_window(), 4 * MB)

        self.memory.rss = 500 * MB
        estimator.begin_window()
        self.memory.rss = 510 * MB
        estimator.sample(self.memory.rss, 10)
        self.assertIsNone(estimator.end_window(), "a cheaper window must not lower the estimate")
        self.assertEqual(estimator.measured, 4 * MB)

    def test_a_stored_estimate_is_used_until_something_is_measured(self):
        estimator = conc.MemoryEstimator("op", stored=3 * MB)
        self.assertEqual(estimator.estimate, 3 * MB)
        self.assertTrue(estimator.from_measurement)

    def test_the_default_guess_is_used_when_nothing_was_stored(self):
        estimator = conc.MemoryEstimator("op")
        self.assertEqual(estimator.estimate, conc.DEFAULT_PER_TASK_MEMORY)
        self.assertFalse(estimator.from_measurement)

    def test_end_window_without_a_baseline_teaches_nothing(self):
        estimator = conc.MemoryEstimator("op")
        self.assertIsNone(estimator.end_window())

    def test_a_window_started_without_a_working_probe_teaches_nothing(self):
        # Rather than measuring growth against a baseline it never had
        estimator = conc.MemoryEstimator("op")
        self.memory.probe_failed = True
        estimator.begin_window()
        self.memory.probe_failed = False
        self.memory.rss = 900 * MB
        estimator.sample(self.memory.rss, 10)
        self.assertIsNone(estimator.end_window())

    def test_a_probe_that_fails_for_one_tick_leaves_the_peak_alone(self):
        # A missing reading is not a reading of zero
        estimator = conc.MemoryEstimator("op")
        self.memory.rss = 500 * MB
        estimator.begin_window()
        self.memory.rss = 540 * MB
        estimator.sample(self.memory.rss, 10)
        estimator.sample(None, 10)
        self.assertEqual(estimator.end_window(), 4 * MB)


class EstimatesFileTests(MemoryStubTestCase):
    def setUp(self):
        super().setUp()
        self._tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self._tempdir.name) / "memory_estimates.json"
        self._real_path = conc._estimates_path
        conc._estimates_path = lambda: self.path
        # Undo MemoryStubTestCase's stub - these tests exercise the real reader, pointed at a
        # temporary file rather than the add-on's own user_files
        conc.load_per_task_estimates = REAL_LOAD_PER_TASK_ESTIMATES

    def tearDown(self):
        conc._estimates_path = self._real_path
        self._tempdir.cleanup()
        super().tearDown()

    def test_a_missing_file_is_not_an_error(self):
        self.assertEqual(conc.load_per_task_estimates(), {})

    def test_a_corrupt_file_is_not_an_error(self):
        # Rather than breaking every run until someone deletes it by hand
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(conc.load_per_task_estimates(), {})

    def test_a_file_holding_something_other_than_a_mapping_is_ignored(self):
        self.path.write_text("[1, 2, 3]", encoding="utf-8")
        self.assertEqual(conc.load_per_task_estimates(), {})

    def test_the_first_measurement_is_stored_as_is(self):
        conc.save_per_task_estimate("Making meanings", 2 * MB)
        self.assertEqual(
            json.loads(self.path.read_text(encoding="utf-8")),
            {"version": conc.ESTIMATES_VERSION, "estimates": {"Making meanings": 2 * MB}},
        )

    def test_a_version_1_file_is_converted_rather_than_thrown_away(self):
        # Its numbers were a slot's worth of tasks, measured before the queue behind the gate
        # was counted. Read as they stand they would be TASK_QUEUE_DEPTH times too high.
        self.path.write_text(json.dumps({"Making meanings": 8 * MB}), encoding="utf-8")
        self.assertEqual(
            conc.load_per_task_estimates()["Making meanings"], 8 * MB / conc.TASK_QUEUE_DEPTH
        )

    def test_a_file_from_a_later_version_is_ignored(self):
        # Measuring again beats guessing at what its numbers mean
        self.path.write_text(
            json.dumps({"version": conc.ESTIMATES_VERSION + 1, "estimates": {"op": 1 * MB}}),
            encoding="utf-8",
        )
        self.assertEqual(conc.load_per_task_estimates(), {})

    def test_a_file_from_a_later_version_is_left_alone_rather_than_overwritten(self):
        # Ignoring what it holds is right; destroying it is not. Running an older version for
        # one session would otherwise throw away everything the newer one had measured.
        contents = {"version": conc.ESTIMATES_VERSION + 1, "estimates": {"op": 1 * MB}}
        self.path.write_text(json.dumps(contents), encoding="utf-8")
        conc.save_per_task_estimate("Making meanings", 2 * MB)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), contents)

    def test_a_corrupt_file_is_left_alone_rather_than_overwritten(self):
        # It is unreadable here, but it may still hold what every other op measured; replacing
        # it with only this op's value would throw those away
        self.path.write_text("{not json", encoding="utf-8")
        conc.save_per_task_estimate("Making meanings", 2 * MB)
        self.assertEqual(self.path.read_text(encoding="utf-8"), "{not json")

    def test_a_file_holding_something_other_than_a_mapping_is_left_alone(self):
        self.path.write_text("[1, 2, 3]", encoding="utf-8")
        conc.save_per_task_estimate("Making meanings", 2 * MB)
        self.assertEqual(self.path.read_text(encoding="utf-8"), "[1, 2, 3]")

    def test_a_failed_write_leaves_the_previous_file_intact(self):
        # The file is written beside itself and moved into place, so a crash partway through
        # cannot leave half a file behind - which is how it came to be unreadable above
        conc.save_per_task_estimate("Making meanings", 1 * MB)
        before = self.path.read_text(encoding="utf-8")
        with mock.patch.object(conc.json, "dump", side_effect=OSError("disk full")):
            conc.save_per_task_estimate("Translating sentences", 4 * MB)
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)
        self.assertFalse(self.path.with_name(f"{self.path.name}.tmp").exists())

    def test_later_measurements_are_blended_so_one_odd_run_cannot_skew_it(self):
        conc.save_per_task_estimate("Making meanings", 1 * MB)
        conc.save_per_task_estimate("Making meanings", 2 * MB)
        expected = int(1 * MB * (1 - conc.ESTIMATE_BLEND) + 2 * MB * conc.ESTIMATE_BLEND)
        self.assertEqual(conc.load_per_task_estimates()["Making meanings"], expected)

    def test_estimates_are_kept_per_op(self):
        conc.save_per_task_estimate("Making meanings", 1 * MB)
        conc.save_per_task_estimate("Translating sentences", 4 * MB)
        stored = conc.load_per_task_estimates()
        self.assertEqual(stored["Making meanings"], 1 * MB)
        self.assertEqual(stored["Translating sentences"], 4 * MB)

    def test_nothing_is_persisted_when_nothing_was_measured(self):
        conc.MemoryEstimator("Making meanings").persist()
        self.assertFalse(self.path.exists())

    def test_an_unwritable_location_is_not_an_error(self):
        # Losing a measurement is not worth failing a run over
        conc._estimates_path = lambda: Path(self._tempdir.name) / "nope" / "\0bad" / "x.json"
        conc.save_per_task_estimate("Making meanings", 1 * MB)


# --- The gate ---------------------------------------------------------------------------------


class GateTestCase(unittest.IsolatedAsyncioTestCase):
    """Async tests against a gate with stubbed memory."""

    def setUp(self):
        self.memory = StubMemory()
        install_memory_stubs(self.memory)

    def tearDown(self):
        restore_memory_probes()

    def make_gate(self, limit=None, max_limit=None, config=None):
        # Every gate records the ceilings it reports, so any test can assert on them
        self.reported_ceilings: list[int] = []
        gate = conc.ConcurrencyGate(
            config or {},
            op_key="test op",
            on_ceiling_changed=self.reported_ceilings.append,
        )
        if limit is not None:
            gate.limit = limit
        if max_limit is not None:
            gate.max_limit = max_limit
        return gate

    def measure_window(self, gate, baseline, peak, in_flight, tasks=0):
        """Run one window through the gate, ending with `peak` reached by `in_flight` tasks.

        `tasks` is the whole window as the bulk ops report it, queued ones included; left out,
        the window is measured as though every task alive was holding a slot.
        """
        self.memory.rss = baseline
        gate.begin_window()
        if tasks:
            gate.note_window_tasks(tasks)
        self.memory.rss = peak
        gate.estimator.sample(self.memory.rss, in_flight)
        gate.end_window()

    async def settle(self):
        """Let queued tasks reach their next await."""
        for _ in range(3):
            await asyncio.sleep(0)


class GateAcquireReleaseTests(GateTestCase):
    async def test_slots_are_handed_out_up_to_the_limit(self):
        gate = self.make_gate(limit=2)
        await gate.acquire()
        await gate.acquire()
        self.assertEqual(gate.in_flight, 2)

    async def test_a_full_gate_makes_the_next_task_wait(self):
        gate = self.make_gate(limit=1)
        await gate.acquire()
        waiter = asyncio.ensure_future(gate.acquire())
        await self.settle()
        self.assertFalse(waiter.done())

        gate.release()
        await asyncio.wait_for(waiter, timeout=1)
        self.assertEqual(gate.in_flight, 1)

    async def test_releasing_wakes_exactly_one_waiter(self):
        gate = self.make_gate(limit=1)
        await gate.acquire()
        first = asyncio.ensure_future(gate.acquire())
        second = asyncio.ensure_future(gate.acquire())
        await self.settle()

        gate.release()
        await asyncio.wait_for(first, timeout=1)
        await self.settle()
        self.assertFalse(second.done())
        self.assertEqual(gate.in_flight, 1)

        gate.release()
        await asyncio.wait_for(second, timeout=1)

    async def test_a_limit_that_shrinks_holds_waiters_until_enough_have_drained(self):
        """The awkward case: the adapt loop halves the limit while tasks are queued.

        Tasks already running keep their slot, so the gate is over its limit until enough of
        them finish. Every release still wakes a waiter, and that waiter has to put itself back
        in the queue rather than slipping through - without losing the wakeup for the others.
        """
        gate = self.make_gate(limit=4)
        for _ in range(4):
            await gate.acquire()

        gate.limit = 2  # memory pressure
        waiter = asyncio.ensure_future(gate.acquire())
        await self.settle()

        gate.release()  # 3 in flight, still over the limit
        await self.settle()
        self.assertFalse(waiter.done())

        gate.release()  # 2 in flight, at the limit
        await self.settle()
        self.assertFalse(waiter.done())

        gate.release()  # 1 in flight, room at last
        await asyncio.wait_for(waiter, timeout=1)
        self.assertEqual(gate.in_flight, 2)

    async def test_a_raised_limit_lets_waiting_tasks_through(self):
        gate = self.make_gate(limit=1, max_limit=8)
        await gate.acquire()
        waiter = asyncio.ensure_future(gate.acquire())
        await self.settle()

        gate.limit = 2
        gate._wake_waiters(1)
        await asyncio.wait_for(waiter, timeout=1)
        self.assertEqual(gate.in_flight, 2)

    async def test_a_cancelled_waiter_passes_its_wakeup_on(self):
        """A task cancelled at the moment it is handed a slot must not swallow it.

        Losing that wakeup would leave the remaining tasks queued behind a gate with a free
        slot and nothing left to trigger it.
        """
        gate = self.make_gate(limit=1)
        await gate.acquire()
        first = asyncio.ensure_future(gate.acquire())
        second = asyncio.ensure_future(gate.acquire())
        await self.settle()

        # Hand a slot to `first` and cancel it in the same tick, before it can run
        gate.release()
        first.cancel()
        await self.settle()

        self.assertTrue(first.cancelled())
        await asyncio.wait_for(second, timeout=1)

    async def test_abort_releases_everything_queued_at_once(self):
        # Cancelling a run with hundreds of queued tasks must unwind in one step rather than
        # relying on each one being cancelled individually
        gate = self.make_gate(limit=1)
        await gate.acquire()
        waiters = [asyncio.ensure_future(gate.acquire()) for _ in range(5)]
        await self.settle()

        gate.abort()
        await self.settle()
        for waiter in waiters:
            with self.assertRaises(asyncio.CancelledError):
                await waiter

    async def test_nothing_gets_through_a_gate_that_has_been_aborted(self):
        gate = self.make_gate(limit=8)
        gate.abort()
        with self.assertRaises(asyncio.CancelledError):
            await gate.acquire()

    async def test_status_text_reports_the_gate_and_what_it_knows(self):
        gate = self.make_gate(limit=4)
        await gate.acquire()
        self.assertIn("1/4", gate.status_text())


class GateAdaptationTests(GateTestCase):
    async def test_the_probes_run_off_the_event_loop(self):
        # On macOS each probe spawns a subprocess with a five-second timeout, twice every two
        # seconds. On the loop that is what polls for cancellation and redraws the progress
        # dialog, so the probes must not be read there.
        loop_thread = threading.current_thread()
        stub = self.memory.system_memory
        probe_threads: list = []

        def recording_probe():
            probe_threads.append(threading.current_thread())
            return stub()

        gate = self.make_gate(limit=8)
        # Only the adapt tick is at issue; the gate's one-off reads while being built are not
        conc.system_memory = recording_probe
        gate.start_adapting()
        try:
            await gate._adapt_once()
        finally:
            gate.stop_adapting()

        self.assertTrue(probe_threads)
        self.assertNotIn(loop_thread, probe_threads)

    async def test_a_saturated_gate_grows_while_memory_is_comfortable(self):
        gate = self.make_gate(limit=16, max_limit=256)
        gate.in_flight = gate.limit
        await gate._adapt_once()
        self.assertGreater(gate.limit, 16)

    async def test_an_idle_gate_does_not_grow(self):
        # If tasks are not queueing up, a bigger limit would not be used anyway
        gate = self.make_gate(limit=16, max_limit=256)
        gate.in_flight = 2
        await gate._adapt_once()
        self.assertEqual(gate.limit, 16)

    async def test_the_gate_never_grows_past_its_ceiling(self):
        gate = self.make_gate(limit=8, max_limit=9)
        gate.in_flight = gate.limit
        await gate._adapt_once()
        self.assertEqual(gate.limit, 9)

    async def test_the_limit_halves_when_free_memory_drops_below_the_reserve(self):
        gate = self.make_gate(limit=16, max_limit=256)
        gate.in_flight = gate.limit
        self.memory.available = 100 * MB
        await gate._adapt_once()
        self.assertEqual(gate.limit, 8)

    async def test_the_limit_halves_when_the_process_passes_a_configured_memory_limit(self):
        gate = self.make_gate(config={"memory_limit": 900})
        gate.limit, gate.max_limit = 16, 256
        gate.in_flight = gate.limit
        self.memory.rss = 1000 * MB
        await gate._adapt_once()
        self.assertEqual(gate.limit, 8)

    async def test_the_limit_halves_when_the_process_passes_a_percent_memory_limit(self):
        self.memory.total = 32 * GB
        gate = self.make_gate(config={"memory_limit": "10%"})
        gate.limit, gate.max_limit = 16, 256
        gate.in_flight = gate.limit
        self.memory.rss = int(3.3 * GB)
        await gate._adapt_once()
        self.assertEqual(gate.limit, 8)

    async def test_the_limit_never_falls_below_one(self):
        gate = self.make_gate(limit=1, max_limit=256)
        self.memory.available = 100 * MB
        for _ in range(4):
            await gate._adapt_once()
        self.assertEqual(gate.limit, conc.MIN_CONCURRENCY)

    async def test_the_gate_recovers_once_the_pressure_passes(self):
        """Backing off has to be temporary, which is why the probe must report current usage.

        The whole pressure response rests on process_memory() falling again when the run's
        memory is freed. Against a probe that reports a high-water mark the halving here would
        repeat every two seconds down to a limit of 1 and stay there for the rest of the
        session - which is what the macOS probe used to do. Whether a platform's probe can
        actually fall is checked by
        test_process_memory_reports_current_usage_not_a_high_water_mark.
        """
        gate = self.make_gate(config={"memory_limit": 900})
        gate.limit, gate.max_limit = 16, 256
        gate.in_flight = gate.limit

        self.memory.rss = 1000 * MB
        await gate._adapt_once()
        self.assertEqual(gate.limit, 8)

        self.memory.rss = 400 * MB
        gate.in_flight = gate.limit
        await gate._adapt_once()
        self.assertGreater(gate.limit, 8)

    async def test_a_measured_window_can_raise_the_ceiling(self):
        # The op turned out cheaper than the default guess, so more of it fits
        self.memory.total = 8 * GB
        self.memory.available = 8 * GB  # budget = 2GB, so 1MB/task gives a ceiling of 256
        gate = self.make_gate()
        gate.max_limit = 32

        self.measure_window(gate, baseline=500 * MB, peak=504 * MB, in_flight=4)  # 1MB/task

        self.assertEqual(gate.max_limit, conc.MAX_AUTO_CONCURRENCY)

    async def test_a_measured_window_can_lower_the_ceiling_and_the_limit_with_it(self):
        self.memory.total = 8 * GB
        self.memory.available = 8 * GB  # budget = 2GB
        gate = self.make_gate(limit=200, max_limit=256)

        # 128MB over 2 live tasks, so 64MB each and 256MB for a slot and its queue
        self.measure_window(gate, baseline=500 * MB, peak=628 * MB, in_flight=2)

        self.assertEqual(gate.max_limit, 2 * GB // (64 * MB * conc.TASK_QUEUE_DEPTH))
        self.assertLessEqual(gate.limit, gate.max_limit)

    async def test_a_windows_queued_tasks_count_towards_what_one_costs(self):
        # The same window, now reported in full: the 128MB was 40 tasks' doing, not 2, so a
        # task costs a fortieth of it and the ceiling lands far higher
        self.memory.total = 8 * GB
        self.memory.available = 8 * GB
        gate = self.make_gate(limit=200, max_limit=256)

        self.measure_window(gate, baseline=500 * MB, peak=628 * MB, in_flight=10, tasks=40)

        self.assertEqual(gate.estimator.measured, 3.2 * MB)
        self.assertEqual(gate.max_limit, int(2 * GB // (3.2 * MB * conc.TASK_QUEUE_DEPTH)))

    async def test_a_configured_maximum_survives_re_measuring(self):
        gate = self.make_gate(config={"max_concurrent_requests": 10})
        self.measure_window(gate, baseline=500 * MB, peak=501 * MB, in_flight=50)  # very cheap
        self.assertEqual(gate.max_limit, 10)

    async def test_a_window_that_measured_nothing_leaves_the_ceiling_alone(self):
        gate = self.make_gate(limit=16, max_limit=64)
        gate.begin_window()
        gate.end_window()
        self.assertEqual(gate.max_limit, 64)

    async def test_probes_that_start_failing_mid_run_do_not_break_the_adapt_loop(self):
        """The probes can stop answering part way through - a WMI hiccup, a missing /proc.

        The run has to carry on. Without a reading there is no pressure signal, so the gate
        goes on growing towards the ceiling it was given; that ceiling was budgeted from a real
        reading at the start, which is what bounds the damage.
        """
        gate = self.make_gate(limit=16, max_limit=64)
        gate.in_flight = gate.limit
        self.memory.probe_failed = True

        for _ in range(10):
            await gate._adapt_once()
            gate.in_flight = gate.limit

        self.assertLessEqual(gate.limit, gate.max_limit)
        self.assertEqual(gate.available_memory, 0)

    async def test_a_window_whose_memory_is_freed_still_measures_its_peak(self):
        """What has to fit in RAM is the high-water mark of the window, not where it ended.

        Tasks release their notes and response buffers as they finish, so by the time the last
        one is done the process has usually given the memory back. Measuring at that point
        would put the per-task cost at nothing and the ceiling at its maximum.
        """
        self.memory.total = 8 * GB
        self.memory.available = 8 * GB
        gate = self.make_gate()

        self.memory.rss = 500 * MB
        gate.begin_window()
        self.memory.rss = 628 * MB  # 2 tasks at 64MB each, both in flight
        gate.estimator.sample(self.memory.rss, 2)
        self.memory.rss = 500 * MB  # both finished and their memory was freed
        gate.end_window()

        self.assertEqual(gate.estimator.measured, 64 * MB)


class CeilingReportingTests(GateTestCase):
    """The connection pool is sized from the ceiling, so it has to hear when the ceiling moves.

    set_connection_pool_size runs once at the start of a run, from a ceiling worked out from
    the starting guess at what a task costs. end_window can move that ceiling after measuring
    the real cost, and without this the pool keeps the size it was given: every request past it
    is a fresh TCP and TLS handshake plus a "connection pool is full" warning.
    """

    async def test_a_raised_ceiling_is_reported(self):
        self.memory.total = 8 * GB
        self.memory.available = 8 * GB
        gate = self.make_gate()
        gate.max_limit = 32

        self.measure_window(gate, baseline=500 * MB, peak=504 * MB, in_flight=4)

        self.assertEqual(self.reported_ceilings, [conc.MAX_AUTO_CONCURRENCY])

    async def test_a_lowered_ceiling_is_reported(self):
        self.memory.total = 8 * GB
        self.memory.available = 8 * GB  # budget = 2GB
        gate = self.make_gate(limit=200, max_limit=256)

        # 64MB per live task, so 256MB per slot and its queue
        self.measure_window(gate, baseline=500 * MB, peak=628 * MB, in_flight=2)

        self.assertEqual(self.reported_ceilings, [2 * GB // (64 * MB * conc.TASK_QUEUE_DEPTH)])

    async def test_a_window_that_measured_nothing_reports_nothing(self):
        gate = self.make_gate(limit=16, max_limit=64)
        gate.begin_window()
        gate.end_window()
        self.assertEqual(self.reported_ceilings, [])

    async def test_a_measurement_that_leaves_the_ceiling_where_it_was_reports_nothing(self):
        # Resizing the pool drops every session, so doing it once per window for a ceiling
        # that has not moved would rebuild connections all run long
        self.memory.total = 8 * GB
        self.memory.available = 8 * GB
        gate = self.make_gate()
        self.assertEqual(gate.max_limit, conc.MAX_AUTO_CONCURRENCY)

        self.measure_window(gate, baseline=500 * MB, peak=504 * MB, in_flight=4)

        self.assertEqual(gate.max_limit, conc.MAX_AUTO_CONCURRENCY)
        self.assertEqual(self.reported_ceilings, [])

    async def test_a_non_adaptive_gate_never_reports(self):
        # Nothing to adapt against, so the ceiling it was given is the ceiling it keeps
        self.memory.probe_failed = True
        gate = self.make_gate()
        self.assertFalse(gate.adaptive)

        self.measure_window(gate, baseline=500 * MB, peak=504 * MB, in_flight=4)

        self.assertEqual(self.reported_ceilings, [])

    async def test_a_gate_with_nobody_listening_still_works(self):
        self.memory.total = 8 * GB
        self.memory.available = 8 * GB
        gate = conc.ConcurrencyGate({}, op_key="test op")
        gate.max_limit = 32
        self.measure_window(gate, baseline=500 * MB, peak=504 * MB, in_flight=4)
        self.assertEqual(gate.max_limit, conc.MAX_AUTO_CONCURRENCY)


class MemoryProbeContractTests(unittest.TestCase):
    """The one thing every platform's probe has to agree on, checked against the real probe.

    Deliberately not faked: a fake would only assert that the fake behaves, and the whole point
    is that the platform underneath is where these go wrong. This runs on whatever machine the
    suite runs on and fails on the one whose probe is broken.
    """

    def test_process_memory_reports_current_usage_not_a_high_water_mark(self):
        """Everything reading process_memory() assumes it tracks usage now, not the peak ever.

        The gate backs off while it is above the configured memory limit, and the estimator
        takes per-task cost from the growth over a window's baseline. A probe that cannot fall
        latches the first on forever - halving concurrency every two seconds down to 1 for the
        rest of the session - and makes the second measure zero growth.

        macOS is the platform this catches: _macos_process_memory used to return
        resource.ru_maxrss, a high-water mark, and now shells out to `ps -o rss=` instead. That
        replacement has not been run on a Mac yet, so this test is what will say whether it
        works - along with test_the_probes_report_bytes, which catches the KB-to-bytes
        conversion `ps` needs and `ru_maxrss` did not.
        """
        size = 256 * MB
        before = conc.process_memory()
        if before is None:
            self.skipTest("no process memory probe on this platform")

        blob = bytearray(size)
        blob[::4096] = bytes(len(blob[::4096]))  # touch every page so it is resident
        peak = conc.process_memory()
        del blob
        after = conc.process_memory()

        self.assertGreater(peak - before, size // 2, "allocating barely moved the probe")
        self.assertLess(
            after - before,
            size // 4,
            "the probe did not fall after the memory was freed, so it reports a peak",
        )


if __name__ == "__main__":
    unittest.main()
