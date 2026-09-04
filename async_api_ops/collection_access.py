"""Reading the collection from worker threads.

Ops run on a pool of worker threads and several of them search the collection per note. Two
things about that turned out to matter a lot when a run is cancelled.

The first is that Anki's backend serialises collection access behind one lock, so sixty threads
calling find_notes do not run sixty searches at once - they queue, and each waits out all the
ones ahead of it. Running them concurrently buys nothing.

The second is what that does to cancelling. A search already inside the backend cannot be
interrupted by anything: not cancelling the task, not raising in the thread, not any signal
Python offers. So whatever has been handed to the backend when the cancel arrives has to be
waited out - and the operation's own final write has to queue up behind all of it. Measured on
a cancelled make-meanings run, that was 46 whole-collection regex searches ahead of a one-note
write, and the write took 121 seconds to happen.

So collection access goes through here. Only one call is inside the backend at a time, and the
rest wait in a queue that cancelling empties instantly. That leaves exactly one search to wait
out instead of all of them, and it costs nothing during a normal run because those searches
were already taking turns.

Ops should use these instead of calling mw.col directly, so that cancellation keeps working
without every op having to think about it.
"""

import logging
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterable, Iterator, Sequence

from aqt import mw

from .api_client import run_cancelled

if TYPE_CHECKING:
    from anki.notes import Note, NoteId

logger = logging.getLogger(__name__)

# One at a time. The backend enforces this anyway; the point of doing it here is that our
# queue can be abandoned and the backend's cannot.
COLLECTION_CONCURRENCY = 1

# How often a waiting thread re-checks whether the run was cancelled
CANCEL_POLL_INTERVAL = 0.05

_collection_semaphore = threading.BoundedSemaphore(COLLECTION_CONCURRENCY)

# A thread that already holds the collection must not queue behind itself. With a limit of one
# that would be a deadlock rather than a slowdown, and it would be an easy thing for a future
# op to do by accident - fetching notes inside a loop that is already holding the collection.
_held = threading.local()

# Refusing on cancellation is right for the worker threads and wrong for the cleanup phase,
# which runs after a cancel precisely so the work already done gets saved. Some ops still have
# collection work to do there - resolving the ids of notes they added, for one - so the thread
# running cleanup is exempt.
_exempt = threading.local()


def begin_cleanup_phase() -> None:
    """Let this thread keep reading the collection even though the run was cancelled.

    Paired with end_cleanup_phase in a finally: the thread running this is reused for later
    operations, so leaving it exempt would quietly disarm cancellation for the next one.
    """
    _exempt.on = True


def end_cleanup_phase() -> None:
    _exempt.on = False


def _giving_up() -> bool:
    return run_cancelled() and not getattr(_exempt, "on", False)


class RunCancelled(Exception):
    """Raised in a worker thread that gave up because the run was cancelled.

    Ops do not need to catch this. It unwinds the op, the task treats it as an unsuccessful
    result, and the run carries on to save what it already had.
    """


@contextmanager
def collection_access(what: str) -> Iterator[None]:
    """Take the collection for the duration of one call, or give up if the run was cancelled."""
    if _giving_up():
        raise RunCancelled(what)

    if getattr(_held, "depth", 0):
        # Already ours; nesting is free
        _held.depth += 1
        try:
            yield
        finally:
            _held.depth -= 1
        return

    while not _collection_semaphore.acquire(timeout=CANCEL_POLL_INTERVAL):
        if _giving_up():
            raise RunCancelled(what)

    _held.depth = 1
    try:
        # Cancellation may have arrived while we were queueing. Checking again here is what
        # keeps a cancelled run from starting one more search it will have to wait out.
        if _giving_up():
            raise RunCancelled(what)
        yield
    finally:
        _held.depth = 0
        _collection_semaphore.release()


def find_notes(query: str) -> "Sequence[NoteId]":
    """mw.col.find_notes, serialised and abandonable."""
    with collection_access(f"find_notes: {query}"):
        return mw.col.find_notes(query)


def get_note(note_id: "NoteId") -> "Note":
    """mw.col.get_note, serialised and abandonable."""
    with collection_access("get_note"):
        return mw.col.get_note(note_id)


def get_notes(note_ids: "Iterable[NoteId]") -> "list[Note]":
    """Fetch several notes under a single turn with the collection.

    Taking and releasing per note would let every other waiting thread in between, turning one
    stretch of work into a long interleaved one.
    """
    ids = list(note_ids)
    if not ids:
        return []
    notes: "list[Note]" = []
    with collection_access(f"get_notes: {len(ids)} notes"):
        for note_id in ids:
            # One turn with the collection, but a bounded one. A broad search hands this
            # thousands of ids, and a batch that runs to the end no matter what is exactly the
            # stretch of uninterruptible backend work this module exists to avoid: only the
            # fetch already inside the backend has to be waited out, not all the rest.
            if _giving_up():
                raise RunCancelled(f"get_notes: gave up after {len(notes)} of {len(ids)} notes")
            notes.append(mw.col.get_note(note_id))
    return notes
