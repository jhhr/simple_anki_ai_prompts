"""Shared HTTP layer for the AI provider APIs.

Handles connection reuse, cancellation and — the main point of this module — reacting to the
rate-limit rejections each provider returns instead of trying to stay under a guessed
requests-per-minute ceiling. Callers just await a final response; the throttling is invisible
to them.
"""

import json
import logging
import random
import re
import socket
import sys
import threading
import time
import weakref
from datetime import datetime, timezone
from typing import Any, Optional

import requests  # type: ignore
from requests.adapters import HTTPAdapter  # type: ignore

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 5
DEFAULT_MAX_RETRY_WAIT_SECONDS = 120
# Ceiling on the exponential backoff used when a provider gives no hint about how long to wait
DEFAULT_RATE_LIMIT_WAIT_SECONDS = 60.0
# Cooldowns are slept off in slices this long so cancellation stays responsive
CANCEL_POLL_INTERVAL = 0.5

GEMINI = "gemini"
OPENAI = "openai"
ANTHROPIC = "anthropic"
TOGETHER = "together"


# --- Run-wide cancellation ----------------------------------------------------------------

# Cancellation used to rely on each op passing its cancel_state down to get_response, and in
# practice almost none of them did - so a cancelled run kept issuing API calls and collection
# queries from its abandoned threads for as long as there was work left. The run's cancelled
# state lives here instead and every request checks it whether or not the caller threaded
# anything through.
#
# It is per-run and per-thread rather than a single process-wide flag. A cancelled run's flag
# has to stay set for as long as its abandoned threads are alive, which is well past the point
# where the operation itself has ended - but work that is not part of that run must not inherit
# it. The single-note ops the editor hooks run are the case that matters: they call the same
# request path from the main thread, and with one shared flag a cancelled bulk run left them
# silently doing nothing for the rest of the session.
#
# So each run gets its own flag, and a thread is cancelled by the run it belongs to. Threads
# that belong to no run - the main thread handling an editor hook - are never cancelled.


class Run:
    """One bulk operation's cancelled flag, shared by every thread taking part in it."""

    __slots__ = ("cancelled",)

    def __init__(self) -> None:
        self.cancelled = threading.Event()


# The run the calling thread is taking part in, if any.
_thread_run = threading.local()

# The run currently in progress. Only used as a fallback for cancelling from a thread that is
# not itself part of the run; the thread-local is what every check reads.
_current_run: Optional[Run] = None


def begin_run() -> Run:
    """Start a bulk operation on this thread, and return its state.

    Pass the returned run to join_run in every worker thread the operation uses, so they are
    cancelled along with it.
    """
    global _current_run
    run = Run()
    _current_run = run
    _thread_run.run = run
    return run


def join_run(run: Run) -> None:
    """Enrol the calling thread in `run`, so cancelling the run stops its work too."""
    _thread_run.run = run


def end_run() -> None:
    """Leave the current run, once the operation has finished with this thread.

    The thread a bulk op runs on is Anki's, and goes back to a pool that later runs unrelated
    work - including our own single-note ops - so its membership must not outlive the op. The
    run's own worker threads stay enrolled: a cancelled run's abandoned threads must keep
    seeing the cancellation for as long as they are alive.
    """
    global _current_run
    run = getattr(_thread_run, "run", None)
    _thread_run.run = None
    if run is not None and _current_run is run:
        _current_run = None


def cancel_run() -> None:
    """Cancel the bulk operation this thread is part of.

    No further requests are issued, and the ones already in flight are aborted rather than
    left to run to completion in threads nothing is waiting for any more.

    The abort is process-wide rather than this run's own sockets, because connections are
    pooled and carry no run of their own. Nothing else can be holding one: Anki's progress
    dialog owns the UI while an operation runs, so no editor hook can start a request alongside
    it and there is never more than one operation in progress.
    """
    run = getattr(_thread_run, "run", None) or _current_run
    if run is None:
        logger.debug("Cancel requested with no run in progress")
    else:
        run.cancelled.set()
    aborted = abort_in_flight_requests()
    logger.info("Run cancelled, aborted %d in-flight request(s)", aborted)


def run_cancelled() -> bool:
    """True if the run this thread is taking part in has been cancelled."""
    run = getattr(_thread_run, "run", None)
    return run is not None and run.cancelled.is_set()


def is_cancelled(cancel_state: Optional[Any] = None) -> bool:
    """True if the run was cancelled, or the caller's own cancel state was set."""
    if run_cancelled():
        return True
    return cancel_state is not None and cancel_state.is_cancelled()


# --- Response classification -------------------------------------------------------------


class ResponseAction:
    """What post_with_retry should do with a response."""

    OK = "ok"
    RETRY = "retry"
    # Terminal: retrying cannot help (bad request, no credit, daily quota used up)
    FAIL = "fail"


# 429 is "too many requests" everywhere; Anthropic's 529 is "overloaded", which is the same
# message about the service as a whole rather than about this one request.
RATE_LIMIT_STATUSES = frozenset({429, 529})


def is_rate_limited(response: "Optional[requests.Response]") -> bool:
    """Whether a response is the provider telling us we are sending too much.

    Only these justify holding every task for the model back. A timeout, a dropped connection
    or a 500 is one request going wrong, and making the other fifty tasks wait it out is how a
    single slow request came to stall a whole run.
    """
    return response is not None and response.status_code in RATE_LIMIT_STATUSES


def _error_body(response: "requests.Response") -> dict:
    """Parse the JSON error body, returning an empty dict if it isn't JSON."""
    try:
        decoded = json.loads(response.text)
    except (ValueError, TypeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _error_object(body: dict) -> dict:
    """The body's `error` member, or an empty dict when it isn't the shape the providers use.

    Not everything answering at an API base URL is the provider itself - a proxy or a gateway
    in front of one may send a bare `{"error": "rate limit exceeded"}`. Reaching straight into
    that for a code or a message raised AttributeError out of classify_response, which turned a
    429 that should have been retried into a failure for that note.
    """
    error = body.get("error")
    return error if isinstance(error, dict) else {}


def parse_seconds_header(value: Optional[str]) -> Optional[float]:
    """Parse a Retry-After header value expressed in seconds."""
    if not value:
        return None
    try:
        return max(0.0, float(value.strip()))
    except ValueError:
        return None


def parse_rfc3339_reset(value: Optional[str]) -> Optional[float]:
    """Parse an RFC 3339 timestamp (Anthropic's *-reset headers) into seconds from now."""
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        reset_at = datetime.fromisoformat(text)
    except ValueError:
        return None
    if reset_at.tzinfo is None:
        reset_at = reset_at.replace(tzinfo=timezone.utc)
    return max(0.0, (reset_at - datetime.now(timezone.utc)).total_seconds())


_GO_DURATION_PART = re.compile(r"([0-9]*\.?[0-9]+)(ms|us|µs|ns|[smh])")
_GO_DURATION_UNITS = {
    "ns": 1e-9,
    "us": 1e-6,
    "µs": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
}


def parse_go_duration(value: Optional[str]) -> Optional[float]:
    """Parse OpenAI's Go-style duration strings, e.g. "1s", "6m0s", "20ms"."""
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    parts = _GO_DURATION_PART.findall(text)
    if not parts:
        return None
    # Reject strings with junk outside the matched parts, e.g. "6m0s extra"
    if "".join(f"{amount}{unit}" for amount, unit in parts) != text:
        return None
    total = 0.0
    for amount, unit in parts:
        total += float(amount) * _GO_DURATION_UNITS[unit]
    return max(0.0, total)


# Google returns retry hints as protobuf-JSON durations, always plain seconds like "34s"
_GOOGLE_DURATION = re.compile(r"^([0-9]*\.?[0-9]+)s$")


def parse_google_duration(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    match = _GOOGLE_DURATION.match(value.strip())
    if not match:
        return None
    return max(0.0, float(match.group(1)))


def _gemini_details(body: dict) -> list:
    details = _error_object(body).get("details")
    return details if isinstance(details, list) else []


def _gemini_is_daily_quota(body: dict) -> bool:
    """A per-day quota won't clear during this run, so retrying is pointless."""
    for detail in _gemini_details(body):
        if not isinstance(detail, dict):
            continue
        if not str(detail.get("@type", "")).endswith("google.rpc.QuotaFailure"):
            continue
        violations = detail.get("violations")
        if not isinstance(violations, list):
            continue
        for violation in violations:
            if not isinstance(violation, dict):
                continue
            quota_id = str(violation.get("quotaId", ""))
            if "PerDay" in quota_id:
                return True
    return False


def _gemini_retry_delay(body: dict) -> Optional[float]:
    for detail in _gemini_details(body):
        if not isinstance(detail, dict):
            continue
        if str(detail.get("@type", "")).endswith("google.rpc.RetryInfo"):
            delay = parse_google_duration(detail.get("retryDelay"))
            if delay is not None:
                return delay
    return None


def classify_response(provider: str, response: "requests.Response") -> tuple[str, Optional[float]]:
    """Decide what to do with a provider response.

    Returns (action, suggested_delay_seconds). The delay is None when the provider gave no
    hint; the caller then falls back to exponential backoff.
    """
    status = response.status_code
    if status == 200:
        return ResponseAction.OK, None

    headers = response.headers
    body = _error_body(response)

    if provider == ANTHROPIC:
        # 429 rate_limit_error, 529 overloaded_error, 500 api_error, 504 timeout_error
        if status in (429, 500, 502, 503, 504, 529):
            delay = parse_seconds_header(headers.get("retry-after"))
            if delay is None:
                # Fall back to whichever limit bucket resets soonest
                resets = [
                    parse_rfc3339_reset(headers.get(name))
                    for name in (
                        "anthropic-ratelimit-requests-reset",
                        "anthropic-ratelimit-input-tokens-reset",
                        "anthropic-ratelimit-output-tokens-reset",
                        "anthropic-ratelimit-tokens-reset",
                    )
                ]
                valid = [r for r in resets if r is not None]
                delay = min(valid) if valid else None
            return ResponseAction.RETRY, delay
        return ResponseAction.FAIL, None

    if provider == GEMINI:
        if status == 429:
            if _gemini_is_daily_quota(body):
                logger.error(
                    "Gemini daily quota exhausted, not retrying: %s",
                    _error_object(body).get("message", response.text),
                )
                return ResponseAction.FAIL, None
            return ResponseAction.RETRY, _gemini_retry_delay(body)
        if status in (500, 502, 503, 504):
            return ResponseAction.RETRY, _gemini_retry_delay(body)
        return ResponseAction.FAIL, None

    # OpenAI and Together share the same error shape
    if status == 429:
        error = _error_object(body)
        code = str(error.get("code") or error.get("type") or "")
        if code == "insufficient_quota":
            logger.error(
                "OpenAI-compatible API reports insufficient quota (billing), not retrying: %s",
                error.get("message", response.text),
            )
            return ResponseAction.FAIL, None
        delay = parse_seconds_header(headers.get("Retry-After"))
        if delay is None:
            resets = [
                parse_go_duration(headers.get(name))
                for name in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens")
            ]
            valid = [r for r in resets if r is not None]
            delay = max(valid) if valid else None
        return ResponseAction.RETRY, delay
    if status in (500, 502, 503, 504):
        return ResponseAction.RETRY, parse_seconds_header(headers.get("Retry-After"))
    return ResponseAction.FAIL, None


def preemptive_cooldown(provider: str, response: "requests.Response") -> Optional[float]:
    """On a successful response, check whether we just used up the last of a limit bucket.

    Returns how long to hold off before the next request to this model, or None when there is
    still headroom. Turns most would-be rejections into a wait.
    """
    headers = response.headers
    try:
        if provider == ANTHROPIC:
            remaining = headers.get("anthropic-ratelimit-requests-remaining")
            if remaining is not None and int(remaining) <= 0:
                return parse_rfc3339_reset(headers.get("anthropic-ratelimit-requests-reset"))
        elif provider in (OPENAI, TOGETHER):
            remaining = headers.get("x-ratelimit-remaining-requests")
            if remaining is not None and int(remaining) <= 0:
                return parse_go_duration(headers.get("x-ratelimit-reset-requests"))
    except (TypeError, ValueError):
        return None
    # Gemini does not return remaining-quota headers
    return None


# --- Rate limit tracking -----------------------------------------------------------------


class RateLimitTracker:
    """Tracks which models are currently in a rate-limit cooldown.

    Read and written from worker threads, so everything is under a lock. Keyed per
    provider+model because limits are enforced separately per model.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cooldown_until: dict[str, float] = {}
        # When each cooldown was installed, for telling a response that predates it from one
        # that can actually speak to whether the limit has cleared
        self._cooldown_set_at: dict[str, float] = {}

    def wait_time(self, key: str) -> float:
        """Seconds left on this model's cooldown, 0 if it's clear."""
        with self._lock:
            until = self._cooldown_until.get(key)
            if until is None:
                return 0.0
            remaining = until - time.monotonic()
            if remaining <= 0:
                del self._cooldown_until[key]
                self._cooldown_set_at.pop(key, None)
                return 0.0
            return remaining

    def note_rate_limited(self, key: str, delay: float) -> None:
        """Hold off further requests to this model for `delay` seconds."""
        with self._lock:
            now = time.monotonic()
            # Never shorten an existing cooldown
            self._cooldown_until[key] = max(now + delay, self._cooldown_until.get(key, 0.0))
            self._cooldown_set_at[key] = now

    def note_success(self, key: str, sent_at: Optional[float] = None) -> None:
        """Note a successful response, clearing the model's cooldown unless it is stale.

        `sent_at` is when that request was sent, and it decides whether the response has
        anything to say about the cooldown. A burst of tasks all have requests in flight when
        the first 429 comes back, and those requests were accepted before the limit was
        reached - so their 200s are no evidence that it has cleared. Clearing on one of them
        released every waiting task straight back into the same limit, where they spent their
        retries on the same rejection. Only a request that demonstrably started after the
        cooldown went up can clear it - a tie counts as stale, since time.monotonic on Windows
        moves in ~16ms steps and two things inside one step tell us nothing about their order.
        Passing no `sent_at` clears unconditionally.
        """
        with self._lock:
            set_at = self._cooldown_set_at.get(key)
            if set_at is not None and sent_at is not None and sent_at <= set_at:
                return
            self._cooldown_until.pop(key, None)
            self._cooldown_set_at.pop(key, None)

    def reset(self) -> None:
        with self._lock:
            self._cooldown_until.clear()
            self._cooldown_set_at.clear()


rate_limit_tracker = RateLimitTracker()


# --- Aborting requests in flight -----------------------------------------------------------

# A thread blocked in session.post() is waiting on a socket read, and nothing outside it can
# make that call return - not cancelling the asyncio task, not closing the Session (which only
# discards connections sitting idle in the pool, never one that is checked out and in use).
# That is why cancelling used to leave hundreds of threads running for minutes, still talking to
# the API and still holding their note and response buffers.
#
# Shutting the socket down does return the read immediately, so every connection registers
# itself here while it has a live socket. On cancellation we shut them all down at once; the
# waiting threads get a connection error, see that the run was cancelled, and exit.
_live_connections: "weakref.WeakSet[Any]" = weakref.WeakSet()
_connection_lock = threading.Lock()

# How many threads are inside session.post() right now, for telling "waiting on the API" apart
# from "busy with something else" when a cancelled run will not settle
_in_request_count = 0
_in_request_lock = threading.Lock()


def _count_request(delta: int) -> None:
    global _in_request_count
    with _in_request_lock:
        _in_request_count += delta


class _TrackedConnection:
    """Mixin registering a connection for the lifetime of its socket."""

    def connect(self) -> None:
        super().connect()  # type: ignore[misc]
        with _connection_lock:
            _live_connections.add(self)

    def close(self) -> None:
        with _connection_lock:
            _live_connections.discard(self)
        super().close()  # type: ignore[misc]


_tracked_pool_classes: "dict[type, type]" = {}


def _tracked_pool_class(pool_class: type) -> type:
    """Derive a pool class whose connections register themselves, from any pool class.

    Derived rather than hardcoded to the plain HTTP(S) pools because a proxied session does not
    use those: an HTTP proxy has its own pools and a SOCKS one has pools with an entirely
    different connection class. Subclassing whatever the manager was going to use adds the
    tracking without replacing the behaviour that gets the connection made in the first place.
    """
    tracked = _tracked_pool_classes.get(pool_class)
    if tracked is None:
        connection_class = pool_class.ConnectionCls  # type: ignore[attr-defined]
        tracked = type(
            f"_Tracked{pool_class.__name__}",
            (pool_class,),
            {
                "ConnectionCls": type(
                    f"_Tracked{connection_class.__name__}",
                    (_TrackedConnection, connection_class),
                    {},
                )
            },
        )
        _tracked_pool_classes[pool_class] = tracked
    return tracked


def _track_connections(manager: Any) -> None:
    """Make a pool manager hand out connections that can be aborted mid-request."""
    # The mapping is assigned per instance by PoolManager.__init__, so replacing it here only
    # affects this manager and never anything else in the process using requests
    manager.pool_classes_by_scheme = {
        scheme: _tracked_pool_class(pool_class)
        for scheme, pool_class in manager.pool_classes_by_scheme.items()
    }


class _TrackedAdapter(HTTPAdapter):
    """An HTTPAdapter whose connections can be aborted mid-request."""

    def init_poolmanager(self, *args: Any, **kwargs: Any) -> None:
        super().init_poolmanager(*args, **kwargs)
        _track_connections(self.poolmanager)

    def proxy_manager_for(self, *args: Any, **kwargs: Any) -> Any:
        # A proxied request never touches self.poolmanager: it goes through a separate manager
        # built here, with stock pool classes. Without this, a user behind a proxy has no
        # connection registered at all, so cancelling finds nothing to abort and the run sits
        # out the full request timeout instead.
        manager = super().proxy_manager_for(*args, **kwargs)
        if not getattr(manager, "_connections_tracked", False):
            _track_connections(manager)
            manager._connections_tracked = True
        return manager


def abort_in_flight_requests() -> int:
    """Force every live connection's socket to return, aborting requests in flight.

    Returns the number of sockets shut down. Safe to call at any time: a connection whose
    socket has been shut down is never reused, it is discarded and replaced on the next
    request.
    """
    with _connection_lock:
        connections = list(_live_connections)
        _live_connections.clear()

    # How many threads are actually inside a request matters as much as the abort itself: if it
    # is far below the number of tasks running, the rest are busy somewhere else entirely and
    # aborting requests was never going to be what stops them.
    logger.info(
        "Aborting requests: %d live connection(s), %d thread(s) inside a request",
        len(connections),
        _in_request_count,
    )

    aborted = 0
    for connection in connections:
        sock = getattr(connection, "sock", None)
        if sock is None:
            continue
        # Enough on its own on Unix, where it makes a blocked read return end-of-file
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            # Already closed, or never finished connecting
            pass
        if sys.platform != "win32":
            # Everywhere else the shutdown is the whole story, and the socket object is left
            # alone: whoever is reading it closes it on the way out. Freeing the descriptor
            # here instead would put the number back in circulation while a thread is still
            # reading from it, so a socket or file opened moments later could be handed the
            # same number and be read by that thread.
            aborted += 1
            continue
        # On Windows the shutdown is not enough: it leaves a thread already blocked in recv
        # sitting there, and sock.close() does not help either because http.client reads
        # through a makefile() wrapper, whose reference keeps close() from actually releasing
        # the descriptor. Detaching and closing the descriptor itself is what makes the pending
        # read return - it fails with "not a socket", which is exactly what we want it to do.
        # It is the reader's descriptor we are pulling out from under it, which is the risk
        # described above, and it is accepted here because the alternative is a cancelled run
        # whose threads stay blocked for the whole request timeout.
        try:
            fileno = sock.detach()
        except Exception as e:
            logger.debug("Could not detach socket while aborting: %s", e)
            continue
        if fileno == -1:
            continue
        try:
            socket.socket(sock.family, sock.type, sock.proto, fileno=fileno).close()
            aborted += 1
        except OSError as e:
            logger.debug("Could not close socket while aborting: %s", e)
    return aborted


# --- Sessions ----------------------------------------------------------------------------

_session_lock = threading.Lock()
_sessions: dict[str, "requests.Session"] = {}
_pool_size = 32


def set_connection_pool_size(size: int) -> None:
    """Size the connection pools to the concurrency ceiling.

    Called at the start of a bulk run. Existing sessions are dropped so the new size takes
    effect; in-flight requests keep their own connection alive until they finish.
    """
    global _pool_size
    with _session_lock:
        if size == _pool_size:
            return
        _pool_size = max(1, size)
        _sessions.clear()


def get_session(provider: str) -> "requests.Session":
    """One connection-pooled session per provider, created lazily."""
    with _session_lock:
        session = _sessions.get(provider)
        if session is None:
            session = requests.Session()
            # max_retries=0: retries are handled by post_with_retry, which knows how to read
            # each provider's rate-limit hints
            adapter = _TrackedAdapter(
                pool_connections=_pool_size,
                pool_maxsize=_pool_size,
                max_retries=0,
            )
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            _sessions[provider] = session
        return session


def close_all_sessions() -> None:
    """Drop every session and the idle connections it is holding.

    This does not touch requests in flight - a connection checked out of the pool survives its
    pool being closed. abort_in_flight_requests() is what stops those. Sessions are re-created
    lazily, so this is safe to call at any point.
    """
    with _session_lock:
        sessions = list(_sessions.values())
        _sessions.clear()
    for session in sessions:
        try:
            session.close()
        except Exception as e:
            logger.debug("Error closing session: %s", e)


# --- The request loop --------------------------------------------------------------------


def _sleep_cancellable(seconds: float, cancel_state: Optional[Any]) -> bool:
    """Sleep in short slices so cancellation isn't delayed. Returns False if cancelled."""
    deadline = time.monotonic() + seconds
    while True:
        if is_cancelled(cancel_state):
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return True
        time.sleep(min(CANCEL_POLL_INTERVAL, remaining))


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter, for when the provider gave no hint."""
    return min(DEFAULT_RATE_LIMIT_WAIT_SECONDS, float(2**attempt)) + random.uniform(0, 1)


def post_with_retry(
    provider: str,
    model: str,
    url: str,
    headers: dict,
    json_body: dict,
    timeout: float,
    cancel_state: Optional[Any] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_retry_wait: float = DEFAULT_MAX_RETRY_WAIT_SECONDS,
) -> Optional["requests.Response"]:
    """POST to a provider, waiting out rate limits and retrying transient failures.

    Blocking; expected to be called from a worker thread. Returns the final response — which
    may be a non-200 the caller should log — or None if the request was cancelled or every
    attempt failed to produce a response.
    """
    key = f"{provider}:{model}"
    session = get_session(provider)
    last_response: Optional["requests.Response"] = None

    for attempt in range(max_retries + 1):
        if is_cancelled(cancel_state):
            # Worth an info line: a burst of these says the tasks were nowhere near their
            # request when the cancel landed, and only reached it long afterwards
            logger.info("Skipping request to %s, the run was cancelled", key)
            return None

        # Another task may already have been rejected for this model; wait rather than
        # spending a request on the same rejection.
        cooldown = rate_limit_tracker.wait_time(key)
        if cooldown > 0:
            logger.debug("Waiting %.1fs on active cooldown for %s", cooldown, key)
            if not _sleep_cancellable(cooldown, cancel_state):
                return None

        # Noted before the request goes out: a 200 only says the limit has cleared if the
        # request was sent after the cooldown went up. See RateLimitTracker.note_success.
        sent_at = time.monotonic()
        try:
            _count_request(1)
            try:
                response = session.post(url, headers=headers, json=json_body, timeout=timeout)
            finally:
                _count_request(-1)
        except requests.exceptions.Timeout:
            logger.warning("Request to %s timed out (attempt %d)", key, attempt + 1)
            action, delay = ResponseAction.RETRY, None
            response = None
        except requests.exceptions.RequestException as e:
            # Includes connection errors, and the aborted-socket error raised when the
            # session is closed by a cancellation
            if is_cancelled(cancel_state):
                return None
            logger.warning("Request to %s failed (attempt %d): %s", key, attempt + 1, e)
            action, delay = ResponseAction.RETRY, None
            response = None
        else:
            last_response = response
            action, delay = classify_response(provider, response)

        # A blocking request can't be aborted from outside, so a cancelled operation's request
        # keeps running in its worker thread and lands here after everything else has moved on.
        # Drop the result: the caller then behaves as if the request failed and won't go on to
        # edit notes behind the back of the cleanup phase.
        if is_cancelled(cancel_state):
            logger.debug("Late response for %s discarded, operation was cancelled", key)
            return None

        if action == ResponseAction.OK and response is not None:
            rate_limit_tracker.note_success(key, sent_at=sent_at)
            hold_off = preemptive_cooldown(provider, response)
            if hold_off:
                logger.debug("Model %s is out of request quota, holding off %.1fs", key, hold_off)
                rate_limit_tracker.note_rate_limited(key, hold_off)
            return response

        if action == ResponseAction.FAIL:
            return last_response

        # Retryable
        if attempt >= max_retries:
            logger.error("Giving up on %s after %d attempts", key, attempt + 1)
            return last_response

        # A hint of zero is not a hint. An Anthropic *-reset header that has already passed, or
        # a Gemini retryDelay of "0s", would otherwise be taken at face value: all six attempts
        # fire back to back with no wait at all, and the cooldown the other tasks for this model
        # wait on is set to nothing.
        if delay is None or delay <= 0:
            delay = _backoff_delay(attempt)
        if delay > max_retry_wait:
            logger.error(
                "%s asked to wait %.0fs, above max_retry_wait_seconds (%.0fs) — giving up",
                key,
                delay,
                max_retry_wait,
            )
            return last_response

        status = response.status_code if response is not None else "no response"
        logger.warning("Retrying %s in %.1fs (status %s)", key, delay, status)
        # Only a rate-limit rejection is worth holding the whole model back for. Everything
        # else retryable - timeouts, dropped connections, 500s - is this request's problem, so
        # this task backs off on its own below and the others carry on.
        if is_rate_limited(response):
            rate_limit_tracker.note_rate_limited(key, delay)
        if not _sleep_cancellable(delay, cancel_state):
            return None

    return last_response
