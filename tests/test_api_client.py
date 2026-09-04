"""Tests for async_api_ops/api_client.py.

The retry loop is the part of this add-on most likely to be got wrong by a later change, and
least likely to be noticed when it is: it only misbehaves under a burst of concurrent tasks
hitting a provider limit, which is exactly the situation nobody reproduces by hand. So the
scenarios here are written as the concurrency they stand for - "another task's 429 landed
while this request was in flight" - rather than as calls in isolation.

Nothing here touches the network or sleeps: a fake session supplies the responses and a fake
clock makes backoffs pass instantly.
"""

import json
import socket
import threading
import unittest
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from requests.structures import CaseInsensitiveDict

from addon_modules import FakeClock, load_addon_module  # type: ignore

api = load_addon_module("api_client")

URL = "https://example.invalid/v1/messages"


class FakeResponse:
    """Enough of requests.Response for classify_response and the retry loop.

    Headers are case-insensitive like the real thing, so a test can't pass by spelling a
    header differently from the provider.
    """

    def __init__(
        self,
        status_code: int = 200,
        headers: Optional[dict] = None,
        body=None,
        text: Optional[str] = None,
    ):
        self.status_code = status_code
        self.headers = CaseInsensitiveDict(headers or {})
        if text is not None:
            self.text = text
        elif body is not None:
            self.text = json.dumps(body)
        else:
            self.text = ""


class FakeSession:
    """Serves scripted outcomes to post().

    An outcome is a FakeResponse, an exception to raise, or a callable taking no arguments
    that returns or raises one. The last outcome repeats, so a test only has to script the
    part it cares about.
    """

    def __init__(self, *outcomes):
        if not outcomes:
            raise ValueError("FakeSession needs at least one outcome")
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def post(self, url, headers=None, json=None, timeout=None):  # noqa: A002 - requests' name
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if callable(outcome):
            outcome = outcome()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    @property
    def call_count(self) -> int:
        return len(self.calls)


class FakeCancelState:
    def __init__(self, cancelled: bool = False):
        self.cancelled = cancelled

    def is_cancelled(self) -> bool:
        return self.cancelled


class SpyTracker(api.RateLimitTracker):
    """The real tracker, with a record of what the retry loop asked of it.

    Whether a cooldown is *installed* is the thing worth asserting on: by the time a call
    returns, the fake clock has usually run the backoff out and the cooldown has expired on
    its own, which would hide the difference.
    """

    def __init__(self):
        super().__init__()
        self.rate_limited_calls: list[tuple[str, float]] = []
        self.success_calls: list[tuple[str, Optional[float]]] = []

    def note_rate_limited(self, key: str, delay: float) -> None:
        self.rate_limited_calls.append((key, delay))
        super().note_rate_limited(key, delay)

    def note_success(self, key: str, sent_at: Optional[float] = None) -> None:
        self.success_calls.append((key, sent_at))
        super().note_success(key, sent_at)


def rfc3339_in(seconds: float) -> str:
    """An RFC 3339 timestamp `seconds` from now, in the form Anthropic's reset headers use."""
    moment = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    return moment.isoformat().replace("+00:00", "Z")


# --- Pure helpers --------------------------------------------------------------------------


class DurationParsingTests(unittest.TestCase):
    def test_parse_seconds_header(self):
        self.assertEqual(api.parse_seconds_header("12"), 12.0)
        self.assertEqual(api.parse_seconds_header(" 12.5 "), 12.5)
        # A provider that has already reset can hand back a negative; never wait a negative time
        self.assertEqual(api.parse_seconds_header("-5"), 0.0)
        self.assertIsNone(api.parse_seconds_header("soon"))
        self.assertIsNone(api.parse_seconds_header(""))
        self.assertIsNone(api.parse_seconds_header(None))

    def test_parse_go_duration(self):
        self.assertEqual(api.parse_go_duration("1s"), 1.0)
        self.assertEqual(api.parse_go_duration("6m0s"), 360.0)
        self.assertEqual(api.parse_go_duration("1.5h"), 5400.0)
        self.assertAlmostEqual(api.parse_go_duration("20ms"), 0.02)
        self.assertAlmostEqual(api.parse_go_duration("500us"), 0.0005)

    def test_parse_go_duration_rejects_anything_it_did_not_fully_understand(self):
        # Partial matches are the dangerous case: "6m0s of quota" must not become 360 seconds
        # by ignoring the tail
        self.assertIsNone(api.parse_go_duration("6m0s extra"))
        self.assertIsNone(api.parse_go_duration("soon"))
        self.assertIsNone(api.parse_go_duration("  "))
        self.assertIsNone(api.parse_go_duration(None))

    def test_parse_google_duration(self):
        self.assertEqual(api.parse_google_duration("34s"), 34.0)
        self.assertEqual(api.parse_google_duration("1.5s"), 1.5)
        self.assertIsNone(api.parse_google_duration("34"))
        self.assertIsNone(api.parse_google_duration("34ms"))
        self.assertIsNone(api.parse_google_duration(None))

    def test_parse_rfc3339_reset(self):
        delay = api.parse_rfc3339_reset(rfc3339_in(30))
        self.assertIsNotNone(delay)
        self.assertGreater(delay, 25.0)
        self.assertLessEqual(delay, 30.0)

    def test_parse_rfc3339_reset_never_returns_a_negative_wait(self):
        self.assertEqual(api.parse_rfc3339_reset(rfc3339_in(-60)), 0.0)

    def test_parse_rfc3339_reset_treats_a_naive_timestamp_as_utc(self):
        naive = (datetime.now(timezone.utc) + timedelta(seconds=30)).replace(tzinfo=None)
        delay = api.parse_rfc3339_reset(naive.isoformat())
        self.assertIsNotNone(delay)
        self.assertGreater(delay, 25.0)

    def test_parse_rfc3339_reset_rejects_junk(self):
        self.assertIsNone(api.parse_rfc3339_reset("tomorrow"))
        self.assertIsNone(api.parse_rfc3339_reset(None))


class ErrorBodyTests(unittest.TestCase):
    def test_non_json_body_is_not_an_error(self):
        # Gateways return HTML error pages; classify_response has to survive them
        self.assertEqual(api._error_body(FakeResponse(502, text="<html>Bad Gateway</html>")), {})

    def test_json_that_is_not_an_object_is_ignored(self):
        self.assertEqual(api._error_body(FakeResponse(400, text="[1, 2]")), {})

    def test_json_object_is_returned(self):
        response = FakeResponse(400, body={"error": {"message": "nope"}})
        self.assertEqual(api._error_body(response), {"error": {"message": "nope"}})


class IsRateLimitedTests(unittest.TestCase):
    def test_rate_limit_statuses(self):
        self.assertTrue(api.is_rate_limited(FakeResponse(429)))
        # Anthropic's "overloaded", which is also about the service rather than the request
        self.assertTrue(api.is_rate_limited(FakeResponse(529)))

    def test_everything_else_is_this_request_s_problem(self):
        for status in (200, 400, 401, 500, 502, 503, 504):
            self.assertFalse(api.is_rate_limited(FakeResponse(status)), status)
        # No response at all: a timeout or a dropped connection
        self.assertFalse(api.is_rate_limited(None))


# --- Response classification ---------------------------------------------------------------


class ClassifyResponseTests(unittest.TestCase):
    def test_200_is_ok_for_every_provider(self):
        for provider in (api.ANTHROPIC, api.GEMINI, api.OPENAI, api.TOGETHER):
            action, delay = api.classify_response(provider, FakeResponse(200))
            self.assertEqual(action, api.ResponseAction.OK, provider)
            self.assertIsNone(delay, provider)

    def test_anthropic_429_uses_retry_after(self):
        response = FakeResponse(429, headers={"retry-after": "7"})
        self.assertEqual(
            api.classify_response(api.ANTHROPIC, response), (api.ResponseAction.RETRY, 7.0)
        )

    def test_anthropic_overloaded_is_retryable(self):
        action, _ = api.classify_response(api.ANTHROPIC, FakeResponse(529))
        self.assertEqual(action, api.ResponseAction.RETRY)

    def test_anthropic_falls_back_to_the_bucket_that_resets_soonest(self):
        # Waiting for the slowest bucket would idle the run long after it could resume
        response = FakeResponse(
            429,
            headers={
                "anthropic-ratelimit-requests-reset": rfc3339_in(120),
                "anthropic-ratelimit-input-tokens-reset": rfc3339_in(20),
            },
        )
        action, delay = api.classify_response(api.ANTHROPIC, response)
        self.assertEqual(action, api.ResponseAction.RETRY)
        self.assertLessEqual(delay, 20.0)
        self.assertGreater(delay, 15.0)

    def test_anthropic_client_error_is_terminal(self):
        self.assertEqual(
            api.classify_response(api.ANTHROPIC, FakeResponse(400)), (api.ResponseAction.FAIL, None)
        )

    def test_gemini_daily_quota_is_terminal(self):
        # A per-day quota will not clear during this run, so retrying just burns attempts
        body = {
            "error": {
                "message": "quota",
                "details": [{
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}],
                }],
            }
        }
        self.assertEqual(
            api.classify_response(api.GEMINI, FakeResponse(429, body=body)),
            (api.ResponseAction.FAIL, None),
        )

    def test_gemini_per_minute_quota_is_retryable_with_its_hint(self):
        body = {
            "error": {
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel"}],
                    },
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "34s"},
                ]
            }
        }
        self.assertEqual(
            api.classify_response(api.GEMINI, FakeResponse(429, body=body)),
            (api.ResponseAction.RETRY, 34.0),
        )

    def test_gemini_server_error_is_retryable(self):
        action, delay = api.classify_response(api.GEMINI, FakeResponse(503, body={}))
        self.assertEqual(action, api.ResponseAction.RETRY)
        self.assertIsNone(delay)

    def test_gemini_survives_malformed_details(self):
        body = {"error": {"details": "not a list"}}
        action, _ = api.classify_response(api.GEMINI, FakeResponse(429, body=body))
        self.assertEqual(action, api.ResponseAction.RETRY)

    def test_openai_insufficient_quota_is_terminal(self):
        # Out of credit, not going too fast: retrying cannot help
        body = {"error": {"code": "insufficient_quota", "message": "billing"}}
        self.assertEqual(
            api.classify_response(api.OPENAI, FakeResponse(429, body=body)),
            (api.ResponseAction.FAIL, None),
        )

    def test_openai_429_prefers_retry_after(self):
        response = FakeResponse(
            429,
            headers={
                "Retry-After": "9",
                "x-ratelimit-reset-requests": "6m0s",
            },
            body={},
        )
        self.assertEqual(
            api.classify_response(api.OPENAI, response), (api.ResponseAction.RETRY, 9.0)
        )

    def test_openai_429_falls_back_to_the_bucket_that_takes_longest(self):
        # Both buckets have to have room before the next request can succeed
        response = FakeResponse(
            429,
            headers={
                "x-ratelimit-reset-requests": "20ms",
                "x-ratelimit-reset-tokens": "6m0s",
            },
            body={},
        )
        self.assertEqual(
            api.classify_response(api.OPENAI, response), (api.ResponseAction.RETRY, 360.0)
        )

    def test_openai_429_without_hints_leaves_the_delay_to_the_caller(self):
        self.assertEqual(
            api.classify_response(api.TOGETHER, FakeResponse(429, body={})),
            (api.ResponseAction.RETRY, None),
        )

    def test_an_error_member_that_is_not_an_object_is_still_classified(self):
        # A proxy in front of the provider may answer with a bare string there. Reaching into
        # it for a code raised AttributeError, which turned a retryable 429 into a hard failure
        # for that note.
        for provider in (api.OPENAI, api.TOGETHER, api.GEMINI):
            self.assertEqual(
                api.classify_response(provider, FakeResponse(429, body={"error": "slow down"})),
                (api.ResponseAction.RETRY, None),
                provider,
            )

    def test_openai_server_errors_are_retryable_and_client_errors_are_not(self):
        for status in (500, 502, 503, 504):
            action, _ = api.classify_response(api.OPENAI, FakeResponse(status, body={}))
            self.assertEqual(action, api.ResponseAction.RETRY, status)
        for status in (400, 401, 403, 404):
            action, _ = api.classify_response(api.OPENAI, FakeResponse(status, body={}))
            self.assertEqual(action, api.ResponseAction.FAIL, status)


class PreemptiveCooldownTests(unittest.TestCase):
    """Turning a would-be rejection into a wait, using the headers of a successful response."""

    def test_anthropic_out_of_requests(self):
        response = FakeResponse(
            200,
            headers={
                "anthropic-ratelimit-requests-remaining": "0",
                "anthropic-ratelimit-requests-reset": rfc3339_in(25),
            },
        )
        hold = api.preemptive_cooldown(api.ANTHROPIC, response)
        self.assertIsNotNone(hold)
        self.assertGreater(hold, 20.0)

    def test_anthropic_with_headroom_left(self):
        response = FakeResponse(
            200,
            headers={
                "anthropic-ratelimit-requests-remaining": "5",
                "anthropic-ratelimit-requests-reset": rfc3339_in(25),
            },
        )
        self.assertIsNone(api.preemptive_cooldown(api.ANTHROPIC, response))

    def test_openai_out_of_requests(self):
        response = FakeResponse(
            200,
            headers={
                "x-ratelimit-remaining-requests": "0",
                "x-ratelimit-reset-requests": "30s",
            },
        )
        self.assertEqual(api.preemptive_cooldown(api.OPENAI, response), 30.0)

    def test_gemini_returns_no_quota_headers(self):
        self.assertIsNone(api.preemptive_cooldown(api.GEMINI, FakeResponse(200)))

    def test_unparseable_headers_are_ignored(self):
        response = FakeResponse(200, headers={"x-ratelimit-remaining-requests": "lots"})
        self.assertIsNone(api.preemptive_cooldown(api.TOGETHER, response))

    def test_missing_headers_are_ignored(self):
        self.assertIsNone(api.preemptive_cooldown(api.OPENAI, FakeResponse(200)))


# --- The cooldown tracker ------------------------------------------------------------------


class RateLimitTrackerTests(unittest.TestCase):
    """One cooldown per provider+model, shared by every task running against that model."""

    def setUp(self):
        self.clock = FakeClock()
        self._real_time = api.time
        api.time = self.clock
        self.tracker = api.RateLimitTracker()
        self.key = "openai:gpt-4o"

    def tearDown(self):
        api.time = self._real_time

    def test_unknown_model_is_clear(self):
        self.assertEqual(self.tracker.wait_time("nobody:nothing"), 0.0)

    def test_cooldown_counts_down_and_clears_itself(self):
        self.tracker.note_rate_limited(self.key, 30.0)
        self.assertEqual(self.tracker.wait_time(self.key), 30.0)
        self.clock.advance(10)
        self.assertEqual(self.tracker.wait_time(self.key), 20.0)
        self.clock.advance(20)
        self.assertEqual(self.tracker.wait_time(self.key), 0.0)

    def test_a_second_rejection_never_shortens_an_active_cooldown(self):
        # Two tasks rejected at once, the second told to wait less: the longer hold has to win
        self.tracker.note_rate_limited(self.key, 60.0)
        self.tracker.note_rate_limited(self.key, 5.0)
        self.assertEqual(self.tracker.wait_time(self.key), 60.0)

    def test_a_later_rejection_can_extend_a_cooldown(self):
        self.tracker.note_rate_limited(self.key, 30.0)
        self.clock.advance(20)
        self.tracker.note_rate_limited(self.key, 30.0)
        self.assertEqual(self.tracker.wait_time(self.key), 30.0)

    def test_cooldowns_are_per_model(self):
        self.tracker.note_rate_limited(self.key, 30.0)
        self.assertEqual(self.tracker.wait_time("openai:gpt-4o-mini"), 0.0)

    def test_reset_clears_every_model_so_a_new_run_starts_clean(self):
        self.tracker.note_rate_limited(self.key, 30.0)
        self.tracker.note_rate_limited("gemini:gemini-3.5-flash-lite", 30.0)
        self.tracker.reset()
        self.assertEqual(self.tracker.wait_time(self.key), 0.0)
        self.assertEqual(self.tracker.wait_time("gemini:gemini-3.5-flash-lite"), 0.0)

    # -- Finding A ---------------------------------------------------------------------------

    def test_success_from_a_request_sent_before_the_cooldown_does_not_clear_it(self):
        """Regression: a 200 from a request already in flight used to wipe the cooldown.

        Fifty tasks send at once, the provider accepts the first few and rejects the rest.
        The rejections put the model on a cooldown; the accepted ones then return 200. Those
        200s say nothing about whether the limit has cleared - they were granted before it was
        reached - and clearing on them released every waiting task straight back into the
        limit, where the burst spent its whole retry budget on the same rejection.
        """
        sent_at = self.clock.monotonic()
        self.clock.advance(0.4)
        self.tracker.note_rate_limited(self.key, 30.0)

        self.tracker.note_success(self.key, sent_at=sent_at)

        self.assertAlmostEqual(self.tracker.wait_time(self.key), 30.0, places=6)

    def test_a_tie_counts_as_stale(self):
        # time.monotonic moves in ~16ms steps on Windows, so two events inside one step say
        # nothing about their order; the safe reading is that the cooldown still stands
        now = self.clock.monotonic()
        self.tracker.note_rate_limited(self.key, 30.0)
        self.tracker.note_success(self.key, sent_at=now)
        self.assertEqual(self.tracker.wait_time(self.key), 30.0)

    def test_success_from_a_request_sent_after_the_cooldown_clears_it(self):
        # The other direction: once a request that started under the cooldown succeeds, the
        # limit really has cleared and the rest of the run should not keep waiting
        self.tracker.note_rate_limited(self.key, 30.0)
        self.clock.advance(5)
        self.tracker.note_success(self.key, sent_at=self.clock.monotonic())
        self.assertEqual(self.tracker.wait_time(self.key), 0.0)

    def test_success_with_no_cooldown_is_harmless(self):
        self.tracker.note_success(self.key, sent_at=self.clock.monotonic())
        self.assertEqual(self.tracker.wait_time(self.key), 0.0)

    def test_success_without_a_timestamp_clears_unconditionally(self):
        self.tracker.note_rate_limited(self.key, 30.0)
        self.tracker.note_success(self.key)
        self.assertEqual(self.tracker.wait_time(self.key), 0.0)

    def test_a_stale_success_after_expiry_leaves_nothing_behind(self):
        # The bookkeeping for "when did the cooldown go up" must not outlive the cooldown
        self.tracker.note_rate_limited(self.key, 10.0)
        self.clock.advance(11)
        self.assertEqual(self.tracker.wait_time(self.key), 0.0)
        self.tracker.note_success(self.key, sent_at=0.0)
        self.assertEqual(self.tracker.wait_time(self.key), 0.0)


# --- The retry loop ------------------------------------------------------------------------


class PostWithRetryTestCase(unittest.TestCase):
    """Base for retry-loop tests: fake clock, fake session, spy tracker, no run cancellation."""

    provider = api.OPENAI
    model = "gpt-4o"

    def setUp(self):
        self.clock = FakeClock()
        self._real_time = api.time
        api.time = self.clock

        self.tracker = SpyTracker()
        self._real_tracker = api.rate_limit_tracker
        api.rate_limit_tracker = self.tracker

        self._real_sessions = dict(api._sessions)
        api._sessions.clear()

        api.begin_run()
        self.key = f"{self.provider}:{self.model}"

    def tearDown(self):
        api.time = self._real_time
        api.rate_limit_tracker = self._real_tracker
        api._sessions.clear()
        api._sessions.update(self._real_sessions)
        api.end_run()

    def serve(self, *outcomes) -> FakeSession:
        session = FakeSession(*outcomes)
        api._sessions[self.provider] = session
        return session

    def post(self, **kwargs):
        params = {
            "provider": self.provider,
            "model": self.model,
            "url": URL,
            "headers": {},
            "json_body": {},
            "timeout": 30.0,
        }
        params.update(kwargs)
        return api.post_with_retry(**params)


class RetrySuccessTests(PostWithRetryTestCase):
    def test_a_successful_response_is_returned_without_retrying(self):
        session = self.serve(FakeResponse(200))
        response = self.post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(session.call_count, 1)
        self.assertEqual(self.clock.total_slept, 0.0)

    def test_the_request_is_sent_as_given(self):
        session = self.serve(FakeResponse(200))
        self.post(headers={"x-key": "secret"}, json_body={"prompt": "hi"}, timeout=12.0)
        call = session.calls[0]
        self.assertEqual(call["url"], URL)
        self.assertEqual(call["headers"], {"x-key": "secret"})
        self.assertEqual(call["json"], {"prompt": "hi"})
        self.assertEqual(call["timeout"], 12.0)

    def test_a_rate_limit_is_waited_out_and_the_retry_returned(self):
        session = self.serve(
            FakeResponse(429, headers={"Retry-After": "20"}, body={}),
            FakeResponse(200),
        )
        response = self.post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(session.call_count, 2)
        self.assertEqual(self.clock.total_slept, 20.0)

    def test_an_existing_cooldown_is_waited_out_before_sending(self):
        # What one task's 429 does for all the others: they wait rather than spending a
        # request on the same rejection
        self.tracker.note_rate_limited(self.key, 15.0)
        session = self.serve(FakeResponse(200))
        response = self.post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(session.call_count, 1)
        self.assertGreaterEqual(self.clock.total_slept, 15.0)

    def test_a_success_that_used_up_the_last_of_a_bucket_holds_the_model_off(self):
        session = self.serve(
            FakeResponse(
                200,
                headers={
                    "x-ratelimit-remaining-requests": "0",
                    "x-ratelimit-reset-requests": "45s",
                },
            )
        )
        response = self.post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(session.call_count, 1)
        self.assertEqual(self.tracker.wait_time(self.key), 45.0)


class RetryFailureTests(PostWithRetryTestCase):
    def test_a_terminal_response_is_returned_without_retrying(self):
        session = self.serve(FakeResponse(400, body={"error": {"message": "bad request"}}))
        response = self.post()
        self.assertEqual(response.status_code, 400)
        self.assertEqual(session.call_count, 1)

    def test_retries_are_bounded_and_the_last_response_is_returned(self):
        session = self.serve(FakeResponse(429, headers={"Retry-After": "1"}, body={}))
        response = self.post(max_retries=2)
        self.assertEqual(response.status_code, 429)
        # max_retries is retries after the first attempt
        self.assertEqual(session.call_count, 3)

    def test_a_wait_longer_than_the_configured_maximum_gives_up_at_once(self):
        # An hour-long reset is not worth blocking the run for; the note is left unprocessed
        session = self.serve(FakeResponse(429, headers={"Retry-After": "3600"}, body={}))
        response = self.post(max_retry_wait=120.0)
        self.assertEqual(response.status_code, 429)
        self.assertEqual(session.call_count, 1)
        self.assertEqual(self.clock.total_slept, 0.0)
        # And nothing is held back for a wait we decided not to take
        self.assertEqual(self.tracker.rate_limited_calls, [])

    def test_transport_failures_are_retried_and_return_none_when_they_never_recover(self):
        session = self.serve(requests.exceptions.ConnectionError("no route"))
        response = self.post(max_retries=1)
        self.assertIsNone(response)
        self.assertEqual(session.call_count, 2)

    def test_an_expired_hint_backs_off_rather_than_retrying_at_once(self):
        # A reset header that has already passed parses as 0.0. Taken at face value it would
        # spend every attempt in a burst and leave the model's cooldown at nothing.
        self.serve(FakeResponse(429, headers={"Retry-After": "0"}, body={}))
        self.post(max_retries=3)
        self.assertGreater(self.clock.total_slept, 7.0)
        self.assertLess(self.clock.total_slept, 11.0)
        self.assertTrue(all(delay > 0 for _, delay in self.tracker.rate_limited_calls))

    def test_backoff_grows_between_attempts_when_the_provider_gives_no_hint(self):
        self.serve(FakeResponse(500, body={}))
        self.post(max_retries=3)
        # Three waits of 2^n seconds plus jitter under a second: 1 + 2 + 4 at least
        self.assertGreater(self.clock.total_slept, 7.0)
        self.assertLess(self.clock.total_slept, 11.0)


class ModelWideCooldownTests(PostWithRetryTestCase):
    """Which failures hold back every other task for the model, and which do not.

    The cooldown is shared, so installing one for a failure that was only this request's
    problem stalls the whole run.
    """

    def test_a_rate_limit_rejection_holds_the_model_back(self):
        self.serve(
            FakeResponse(429, headers={"Retry-After": "20"}, body={}),
            FakeResponse(200),
        )
        self.post()
        self.assertEqual(self.tracker.rate_limited_calls, [(self.key, 20.0)])

    def test_an_overloaded_response_holds_the_model_back(self):
        self.provider = api.ANTHROPIC
        self.key = f"{self.provider}:{self.model}"
        self.serve(FakeResponse(529, headers={"retry-after": "10"}), FakeResponse(200))
        self.post()
        self.assertEqual(self.tracker.rate_limited_calls, [(self.key, 10.0)])

    # -- Finding B ---------------------------------------------------------------------------

    def test_a_timeout_does_not_hold_the_model_back(self):
        """Regression: one slow request used to put every task for the model on a cooldown.

        note_rate_limited sat on the shared retry path, so a timeout - which says nothing
        about rate limits - installed a model-wide hold. With `max(new, existing)` never
        shortening it, a handful of timeouts across a burst compounded into a long stall of
        every other task.
        """
        self.serve(requests.exceptions.Timeout("timed out"), FakeResponse(200))
        response = self.post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.tracker.rate_limited_calls, [])

    def test_a_dropped_connection_does_not_hold_the_model_back(self):
        self.serve(requests.exceptions.ConnectionError("reset by peer"), FakeResponse(200))
        self.post()
        self.assertEqual(self.tracker.rate_limited_calls, [])

    def test_a_server_error_does_not_hold_the_model_back(self):
        self.serve(FakeResponse(500, body={}), FakeResponse(200))
        self.post()
        self.assertEqual(self.tracker.rate_limited_calls, [])

    def test_repeated_timeouts_never_compound_into_a_stall(self):
        self.serve(requests.exceptions.Timeout("timed out"))
        self.post(max_retries=4)
        self.assertEqual(self.tracker.rate_limited_calls, [])
        self.assertEqual(self.tracker.wait_time(self.key), 0.0)

    # -- Finding A, through the retry loop ----------------------------------------------------

    def test_a_success_does_not_clear_a_cooldown_set_while_it_was_in_flight(self):
        """Regression, end to end: the burst must stay held after one task's 200 lands.

        This request is sent, another task is rejected while it is in flight, and then this
        one comes back 200. The cooldown that the other task installed has to survive.
        """

        def respond():
            # Another task's 429 lands, and time passes, while this request is in flight
            self.clock.advance(0.4)
            self.tracker.note_rate_limited(self.key, 30.0)
            return FakeResponse(200)

        self.serve(respond)
        response = self.post()

        self.assertEqual(response.status_code, 200)
        self.assertAlmostEqual(self.tracker.wait_time(self.key), 30.0, places=6)

    def test_a_success_reports_when_its_request_was_sent(self):
        self.serve(FakeResponse(200))
        sent_before = self.clock.monotonic()
        self.post()
        self.assertEqual(len(self.tracker.success_calls), 1)
        key, sent_at = self.tracker.success_calls[0]
        self.assertEqual(key, self.key)
        self.assertIsNotNone(sent_at)
        self.assertGreaterEqual(sent_at, sent_before)


class CancellationTests(PostWithRetryTestCase):
    def test_a_cancelled_run_sends_nothing(self):
        session = self.serve(FakeResponse(200))
        api.cancel_run()
        self.assertIsNone(self.post())
        self.assertEqual(session.call_count, 0)

    def test_begin_run_clears_a_previous_cancellation(self):
        api.cancel_run()
        self.assertTrue(api.run_cancelled())
        api.begin_run()
        self.assertFalse(api.run_cancelled())
        self.serve(FakeResponse(200))
        self.assertIsNotNone(self.post())

    def test_a_cancelled_caller_sends_nothing(self):
        session = self.serve(FakeResponse(200))
        self.assertIsNone(self.post(cancel_state=FakeCancelState(cancelled=True)))
        self.assertEqual(session.call_count, 0)

    def test_a_response_arriving_after_a_cancel_is_discarded(self):
        # A blocking request cannot be aborted from inside, so it lands after everything else
        # has moved on. Acting on it would edit notes behind the cleanup phase's back.
        state = FakeCancelState()

        def respond():
            state.cancelled = True
            return FakeResponse(200)

        self.serve(respond)
        self.assertIsNone(self.post(cancel_state=state))
        self.assertEqual(self.tracker.success_calls, [])

    def test_a_cancel_during_a_backoff_stops_the_retries_without_waiting_it_out(self):
        # Backoffs are slept in short slices for exactly this: a run cancelled while a task is
        # waiting out a half-minute rate limit must not take half a minute to notice
        state = FakeCancelState()
        sleep_slice = self.clock.sleep

        def cancel_once_this_task_starts_waiting(seconds):
            state.cancelled = True
            sleep_slice(seconds)

        self.clock.sleep = cancel_once_this_task_starts_waiting

        session = self.serve(
            FakeResponse(429, headers={"Retry-After": "30"}, body={}),
            FakeResponse(200),
        )
        self.assertIsNone(self.post(cancel_state=state))
        self.assertEqual(session.call_count, 1)
        self.assertLessEqual(self.clock.total_slept, api.CANCEL_POLL_INTERVAL)

    def test_is_cancelled_reads_both_the_run_and_the_caller(self):
        self.assertFalse(api.is_cancelled(None))
        self.assertFalse(api.is_cancelled(FakeCancelState(cancelled=False)))
        self.assertTrue(api.is_cancelled(FakeCancelState(cancelled=True)))
        api.cancel_run()
        self.assertTrue(api.is_cancelled(None))

    def test_a_thread_outside_the_run_keeps_working_after_it_is_cancelled(self):
        # The editor hooks run their single-note ops on the main thread, which takes no part in
        # any bulk run. With one process-wide flag, cancelling a bulk run left every one of them
        # - a story or a translation on field unfocus - silently doing nothing from then on.
        session = self.serve(FakeResponse(200))
        api.cancel_run()

        outcome = {}

        def outside_the_run():
            outcome["cancelled"] = api.run_cancelled()
            outcome["response"] = self.post()

        thread = threading.Thread(target=outside_the_run)
        thread.start()
        thread.join(5)

        self.assertFalse(outcome["cancelled"])
        self.assertIsNotNone(outcome["response"])
        self.assertEqual(session.call_count, 1)

    def test_a_worker_enrolled_in_the_run_is_cancelled_with_it(self):
        # The point of the run being shared rather than per-thread: the ops overwhelmingly do
        # not pass their cancel_state down, so the worker pool has to be stopped by the run.
        session = self.serve(FakeResponse(200))
        run = api.begin_run()
        joined = threading.Event()
        cancelled = threading.Event()
        outcome = {}

        def worker():
            api.join_run(run)
            joined.set()
            cancelled.wait(5)
            outcome["cancelled"] = api.run_cancelled()
            outcome["response"] = self.post()

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(joined.wait(5))
        api.cancel_run()
        cancelled.set()
        thread.join(5)

        self.assertTrue(outcome["cancelled"])
        self.assertIsNone(outcome["response"])
        self.assertEqual(session.call_count, 0)

    def test_leaving_a_cancelled_run_does_not_uncancel_it(self):
        # end_run hands the op's thread back uncancelled, because Anki reuses it for unrelated
        # work. The threads the run abandoned keep seeing the cancellation, which is what stops
        # them issuing requests for as long as they are alive.
        run = api.begin_run()
        api.cancel_run()
        self.assertTrue(api.run_cancelled())

        api.end_run()

        self.assertFalse(api.run_cancelled())
        self.assertTrue(run.cancelled.is_set())


class SessionTests(unittest.TestCase):
    """Connection pools are sized to the concurrency ceiling at the start of each run."""

    def setUp(self):
        self._real_sessions = dict(api._sessions)
        self._real_pool_size = api._pool_size
        api._sessions.clear()

    def tearDown(self):
        api._sessions.clear()
        api._sessions.update(self._real_sessions)
        api._pool_size = self._real_pool_size

    def test_one_session_per_provider_is_reused(self):
        first = api.get_session(api.OPENAI)
        self.assertIs(api.get_session(api.OPENAI), first)
        self.assertIsNot(api.get_session(api.ANTHROPIC), first)

    def test_resizing_the_pool_drops_existing_sessions_so_the_size_takes_effect(self):
        api.set_connection_pool_size(8)
        first = api.get_session(api.OPENAI)
        api.set_connection_pool_size(64)
        self.assertEqual(api._pool_size, 64)
        self.assertIsNot(api.get_session(api.OPENAI), first)

    def test_resizing_to_the_same_size_keeps_the_sessions(self):
        api.set_connection_pool_size(16)
        first = api.get_session(api.OPENAI)
        api.set_connection_pool_size(16)
        self.assertIs(api.get_session(api.OPENAI), first)

    def test_the_pool_is_never_sized_to_nothing(self):
        api.set_connection_pool_size(0)
        self.assertGreaterEqual(api._pool_size, 1)

    def test_sessions_mount_the_adapter_that_can_abort_requests(self):
        # Cancellation depends on it: an ordinary adapter's sockets cannot be shut down
        session = api.get_session(api.GEMINI)
        self.assertIsInstance(session.get_adapter("https://example.invalid"), api._TrackedAdapter)

    def test_direct_connections_are_registered_for_aborting(self):
        adapter = api.get_session(api.GEMINI).get_adapter("https://example.invalid")
        for pool_class in adapter.poolmanager.pool_classes_by_scheme.values():
            self.assertTrue(issubclass(pool_class.ConnectionCls, api._TrackedConnection))

    def test_proxied_connections_are_registered_too(self):
        # A proxied request never touches the adapter's own poolmanager, so without this the
        # sockets to abort on cancellation are not registered anywhere and a cancel waits out
        # the whole request timeout
        adapter = api.get_session(api.GEMINI).get_adapter("https://example.invalid")
        manager = adapter.proxy_manager_for("http://proxy.invalid:8080")
        for pool_class in manager.pool_classes_by_scheme.values():
            self.assertTrue(issubclass(pool_class.ConnectionCls, api._TrackedConnection))

    def test_tracking_keeps_whatever_pool_the_manager_was_going_to_use(self):
        # Subclassed rather than replaced: a proxy's pools know how to reach the proxy, and a
        # SOCKS proxy's connection class is not an HTTP one at all
        from urllib3.connectionpool import HTTPSConnectionPool

        tracked = api._tracked_pool_class(HTTPSConnectionPool)
        self.assertTrue(issubclass(tracked, HTTPSConnectionPool))
        self.assertTrue(issubclass(tracked.ConnectionCls, HTTPSConnectionPool.ConnectionCls))
        self.assertIs(api._tracked_pool_class(HTTPSConnectionPool), tracked)

    def test_closing_sessions_is_safe_when_there_are_none(self):
        api.close_all_sessions()
        api.close_all_sessions()

    def test_aborting_with_no_live_connections_reports_nothing_aborted(self):
        self.assertEqual(api.abort_in_flight_requests(), 0)


class FakeSocket:
    """Records what aborting did to it."""

    family = socket.AF_INET
    type = socket.SOCK_STREAM
    proto = 0

    def __init__(self):
        self.shutdown_calls = 0
        self.detached = False

    def shutdown(self, how):
        self.shutdown_calls += 1

    def detach(self):
        self.detached = True
        return -1


class FakeConnection:
    def __init__(self):
        self.sock = FakeSocket()


class AbortingRequestsTests(unittest.TestCase):
    def setUp(self):
        self.connection = FakeConnection()
        api._live_connections.add(self.connection)

    def tearDown(self):
        api._live_connections.clear()

    def abort_as(self, platform: str) -> int:
        real = api.sys.platform
        api.sys.platform = platform
        try:
            return api.abort_in_flight_requests()
        finally:
            api.sys.platform = real

    def test_windows_takes_the_descriptor_away_because_a_shutdown_is_not_enough(self):
        self.abort_as("win32")
        self.assertEqual(self.connection.sock.shutdown_calls, 1)
        self.assertTrue(self.connection.sock.detached)

    def test_elsewhere_the_shutdown_is_the_whole_story(self):
        # Freeing the descriptor here would put the number back in circulation while a thread
        # is still reading from it, and something opened moments later could be handed it
        self.assertEqual(self.abort_as("linux"), 1)
        self.assertEqual(self.connection.sock.shutdown_calls, 1)
        self.assertFalse(self.connection.sock.detached)


if __name__ == "__main__":
    unittest.main()
