"""Logging helpers for working out what a cancelled run is still busy doing.

Cancelling has been hard to diagnose because the interesting state is spread across threads
that nothing is waiting for any more: the run has moved on, the progress window is showing
"Cancelling operations...", and something invisible is still burning CPU. The counters we had
said how many threads were alive but never what they were executing.

So the two things here are a stage marker, which the ops call as they move through their work,
and a sampler that reports what every thread's call stack looks like. Both stay quiet during a
normal run and only speak up once a run has been cancelled, which is the case worth watching.
"""

import logging
import sys
import threading
import time
from collections import Counter
from typing import Optional

from .api_client import run_cancelled

logger = logging.getLogger(__name__)

# The prefix selected_notes_op gives its shared pool, so a worker running op code can be told
# apart from Anki's own threads
WORKER_THREAD_PREFIX = "simple_anki_ai_prompts"
WATCHDOG_THREAD_NAME = "sap_cancel_watchdog"

# How deep to walk each stack. Three frames is enough to tell "in a dictionary lookup" from
# "in a JSON parse" without turning the log into a wall of tracebacks.
_STACK_DEPTH = 3


def diagnostic_level() -> int:
    """Info once the run is cancelled, debug otherwise.

    A cancelled run is the one we need to be able to read after the fact, and it only produces
    a bounded burst of these, so they are worth promoting. During a normal run the same calls
    would be per-note noise.
    """
    return logging.INFO if run_cancelled() else logging.DEBUG


def log_stage(stage_logger: logging.Logger, label: str, **extra) -> None:
    """Mark that a thread reached a point in an op.

    Used to see how far an abandoned task gets after its run was cancelled, and how long it
    spends there. Threads that keep working after a cancel are the whole question.
    """
    level = diagnostic_level()
    if not stage_logger.isEnabledFor(level):
        return
    details = "".join(f" {key}={value}" for key, value in extra.items())
    stage_logger.log(level, "[stage] %s%s", label, details)


def log_timed_stage(
    stage_logger: logging.Logger, label: str, started: float, **extra
) -> float:
    """log_stage, with how long the step took. Returns a fresh marker for the next step."""
    now = time.monotonic()
    level = diagnostic_level()
    if stage_logger.isEnabledFor(level):
        details = "".join(f" {key}={value}" for key, value in extra.items())
        stage_logger.log(level, "[stage] %s took %.3fs%s", label, now - started, details)
    return now


def _describe_frame(frame) -> str:
    """The innermost few frames of a stack, shortened to something greppable."""
    parts: list[str] = []
    while frame is not None and len(parts) < _STACK_DEPTH:
        code = frame.f_code
        filename = code.co_filename.replace("\\", "/").rsplit("/", 1)[-1]
        parts.append(f"{filename}:{frame.f_lineno} {code.co_name}")
        frame = frame.f_back
    return " <- ".join(parts)


def thread_stack_counts() -> "Counter":
    """What every thread is executing right now, grouped by where it is.

    Grouping matters: sixty threads in the same dictionary lookup should read as one line
    saying sixty, not sixty lines.
    """
    frames = sys._current_frames()
    names = {thread.ident: thread.name for thread in threading.enumerate()}
    counts: "Counter" = Counter()
    for ident, frame in frames.items():
        name = names.get(ident, f"thread-{ident}")
        if name == WATCHDOG_THREAD_NAME:
            continue
        group = "worker" if name.startswith(WORKER_THREAD_PREFIX) else name
        counts[(group, _describe_frame(frame))] += 1
    return counts


def worker_threads() -> list:
    return [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith(WORKER_THREAD_PREFIX)
    ]


def dump_thread_stacks(reason: str) -> int:
    """Log where every thread currently is. Returns the number of addon worker threads."""
    counts = thread_stack_counts()
    workers = sum(count for (group, _), count in counts.items() if group == "worker")
    logger.info(
        "[stacks] %s: %d threads total, %d addon workers",
        reason,
        sum(counts.values()),
        workers,
    )
    for (group, where), count in counts.most_common():
        logger.info("[stacks]   %3d x [%s] %s", count, group, where)
    return workers


def start_cancel_watchdog(max_seconds: float = 600.0) -> None:
    """Sample every thread's stack until the addon's workers are gone.

    Runs on its own thread so it keeps reporting even while the event loop, the main thread and
    the collection operation are all stuck behind whatever the workers are doing. The interval
    backs off so a long hang does not fill the log.
    """

    def run() -> None:
        started = time.monotonic()
        interval = 1.0
        while True:
            elapsed = time.monotonic() - started
            workers = dump_thread_stacks(f"{elapsed:.1f}s after cancel")
            if workers == 0:
                logger.info(
                    "[stacks] all addon worker threads finished %.1fs after the cancel", elapsed
                )
                return
            if elapsed >= max_seconds:
                logger.info(
                    "[stacks] giving up watching after %.0fs, %d workers still alive",
                    elapsed,
                    workers,
                )
                return
            time.sleep(interval)
            interval = min(interval * 1.6, 10.0)

    watchdog = threading.Thread(target=run, name=WATCHDOG_THREAD_NAME, daemon=True)
    watchdog.start()


class StageTimer:
    """Records how long each step of one op call took, and reports it in a single line.

    One line per task rather than one per step: with dozens of tasks unwinding at once, the
    per-step lines interleave into something unreadable, and what we actually want to compare
    is where each task spent its time.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.started = time.monotonic()
        self._marker = self.started
        self._steps: list[tuple[str, float]] = []

    def step(self, name: str) -> None:
        now = time.monotonic()
        self._steps.append((name, now - self._marker))
        self._marker = now

    def report(self, stage_logger: logging.Logger, outcome: str, **extra) -> None:
        level = diagnostic_level()
        if not stage_logger.isEnabledFor(level):
            return
        total = time.monotonic() - self.started
        steps = " ".join(f"{name}={seconds:.2f}s" for name, seconds in self._steps)
        details = "".join(f" {key}={value}" for key, value in extra.items())
        stage_logger.log(
            level,
            "[stage] %s %s in %.2fs (%s)%s",
            self.label,
            outcome,
            total,
            steps or "no steps",
            details,
        )


_cancel_marker_lock = threading.Lock()
_cancelled_at: Optional[float] = None


def note_cancel_time() -> None:
    """Remember when the cancel happened, so later work can be reported relative to it."""
    global _cancelled_at
    with _cancel_marker_lock:
        _cancelled_at = time.monotonic()


def clear_cancel_time() -> None:
    """Forget the previous cancel, at the start of a run.

    The marker is process-wide, and left standing it made every later run report every task as
    having "returned Ns after the cancel" - a cancel that happened in some earlier run and has
    nothing to do with the work being timed.
    """
    global _cancelled_at
    with _cancel_marker_lock:
        _cancelled_at = None


def seconds_since_cancel() -> Optional[float]:
    with _cancel_marker_lock:
        if _cancelled_at is None:
            return None
        return time.monotonic() - _cancelled_at
