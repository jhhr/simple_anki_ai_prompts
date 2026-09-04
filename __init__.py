import os
import sys
import logging
import threading
import time
from datetime import datetime

from anki import hooks
from anki.notes import Note, NoteId
from aqt import gui_hooks
from aqt import mw
from aqt.browser import Browser
from aqt.qt import QAction, qconnect, QMenu

# Add the 'lib' directory to sys.path for module imports, modules will import from there
# so this needs to be done before any other imports
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
if lib_path not in sys.path:
    sys.path.append(lib_path)

# E402 - module level import not at top of file
from .utils import get_field_config  # noqa: E402
from .async_api_ops.diagnostics import WORKER_THREAD_PREFIX  # noqa: E402


from .async_api_ops.clean_meaning import (  # noqa: E402
    clean_meaning_in_note,
    clean_selected_notes,
)
from .async_api_ops.translate_field import (  # noqa: E402
    translate_selected_notes,
    translate_sentence_in_note,
)
from .async_api_ops.make_kanji_story import (  # noqa: E402
    make_stories_for_selected_notes,
    make_story_for_note,
)
from .async_api_ops.kanjify_sentence import (  # noqa: E402
    kanjify_selected_notes,
)
from .async_api_ops.extract_words import (  # noqa: E402
    extract_words_from_selected_notes,
    extract_words_in_note,
    extract_words_test_compare_from_selected_notes,
)
from .async_api_ops.migrate_compound_verbs import (  # noqa: E402
    migrate_compound_verbs_from_selected_notes,
)
from .async_api_ops.match_words_to_notes import (  # noqa: E402
    match_words_to_notes_from_selected,
    match_single_word_to_notes_from_selected,
)

from .async_api_ops.make_all_meanings import (  # noqa: E402
    make_meanings_selected_notes,
    merge_meanings_selected_notes,
)
from .async_api_ops.new_note_all_ops import (  # noqa: E402
    new_note_all_ops_selected_notes,
)
from .sync_local_ops.find_missing_matched_note_ids import (  # noqa: E402
    find_missing_matched_note_ids_selected_notes,
)
from .sync_local_ops.tag_notes_matched_status import (  # noqa: E402
    tag_notes_matched_status_from_selected,
)
from .sync_local_ops.deduplicate_existing_meaning_notes import (  # noqa: E402
    deduplicate_existing_meaning_notes_selected_notes,
)
from .sync_local_ops.make_fine_tuning_data import (  # noqa: E402
    make_kanjify_sentence_fine_tuning_data,
    make_extract_words_fine_tuning_data,
)


# Initialize root logger for the addon at module load
def setup_addon_logging():
    """Set up the root logger for this addon"""
    addon_logger = logging.getLogger(__name__.split(".")[0])  # Get root addon logger

    # Set initial level (will be updated from config)
    addon_logger.setLevel(logging.ERROR)

    # Prevent propagation to Anki's loggers
    addon_logger.propagate = False


setup_addon_logging()


# Marks the handlers this addon attaches, so they can be found and closed again
_ADDON_HANDLER_FLAG = "_simple_anki_ai_prompts_handler"


# How long to keep waiting for a run's threads before closing its log file anyway. Long enough
# for a cancelled run to finish unwinding, short enough that the file doesn't stay open for the
# session if something never exits.
_LOG_CLOSE_TIMEOUT_SECONDS = 300
_LOG_CLOSE_POLL_SECONDS = 2.0


def _addon_threads_alive() -> bool:
    """Whether any of this addon's worker threads is still running.

    A run's threads outlive the operation on purpose: a cancelled run cannot interrupt a
    request already in flight, so its threads unwind on their own afterwards - and what they
    log while doing it is the whole point of the cancellation diagnostics.
    """
    return any(
        thread.name.startswith(WORKER_THREAD_PREFIX) and thread.is_alive()
        for thread in threading.enumerate()
    )


def _close_handler_when_idle(handler: logging.Handler) -> None:
    """Close a detached handler, once nothing is still writing through it.

    Closing it straight away closed the file out from under a cancelled run's threads. They
    keep logging as they unwind, and a handler whose stream has been closed re-opens the file
    on the next record - so the descriptor this function exists to release was leaked after
    all, and the run's last diagnostics ended up somewhere nothing was looking.
    """

    def close_quietly() -> None:
        try:
            handler.close()
        except Exception:
            pass

    if not _addon_threads_alive():
        close_quietly()
        return

    def wait_and_close() -> None:
        deadline = time.monotonic() + _LOG_CLOSE_TIMEOUT_SECONDS
        while time.monotonic() < deadline and _addon_threads_alive():
            time.sleep(_LOG_CLOSE_POLL_SECONDS)
        close_quietly()

    threading.Thread(target=wait_and_close, name="sap_log_closer", daemon=True).start()


def close_previous_log_handlers(logger_instance: logging.Logger) -> None:
    """Detach any log handler this addon attached earlier, and close it once it is idle.

    A handler was added every time the browser context menu was built or a field lost focus,
    and none were ever removed. They accumulate for the lifetime of the session, so every log
    record gets written once per handler - and with several worker threads logging at once
    that turns into a great deal of redundant file I/O. It also keeps every previous log file
    open, which is why they can't be deleted until Anki is closed.

    Detaching is immediate; the close waits for the threads that may still be writing.

    Nothing detached here can belong to an operation still running. Every caller is a UI hook -
    building the browser context menu, unfocusing a field, adding a note - and Anki's progress
    dialog owns the UI while an operation is in progress, so none of them can fire until it has
    finished. Cancelling is the only thing the user can do meanwhile.
    """
    for handler in list(logger_instance.handlers):
        if getattr(handler, _ADDON_HANDLER_FLAG, False):
            logger_instance.removeHandler(handler)
            _close_handler_when_idle(handler)


def create_call_log_handler(function_name: str) -> logging.Handler:
    """Create a new file handler for a specific function call"""
    config = mw.addonManager.getConfig(__name__) or {}

    # Get log level from config
    log_level_str = config.get("log_level", "ERROR")
    log_level = getattr(logging, log_level_str.upper(), logging.ERROR)

    # Update the root addon logger's level to match config
    addon_logger = logging.getLogger(__name__.split(".")[0])
    addon_logger.setLevel(log_level)

    # Check if console logging is enabled
    log_to_console = config.get("log_to_console", False)

    if log_to_console:
        # Create console handler
        handler: logging.Handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        setattr(handler, _ADDON_HANDLER_FLAG, True)
        return handler

    # Create logs directory
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(addon_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Create unique log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"{function_name}_{timestamp}.log")

    # Create handler. delay=True so the file isn't opened (or created) until something is
    # actually logged - building the context menu shouldn't leave an empty log file behind.
    handler = logging.FileHandler(log_file, encoding="utf-8", delay=True)
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    setattr(handler, _ADDON_HANDLER_FLAG, True)

    return handler


# Function to be executed when the browser menus are initialized
def on_browser_will_show_context_menu(browser: Browser, menu: QMenu):
    handler = create_call_log_handler("add_note")
    logger = logging.getLogger(__name__)

    if handler:
        # Replace the previous run's handler rather than stacking another one on top
        close_previous_log_handlers(logger)
        logger.addHandler(handler)

    # Create a new action for the context menu
    meaning_action = QAction("Clean dictionary meaning", mw)
    translation_action = QAction("Translate sentence", mw)
    kanji_story_action = QAction("Generate kanji story", mw)
    component_words_action = QAction("Kanjify sentence", mw)
    extract_words_action = QAction("Extract words", mw)
    extract_words_test_compare_action = QAction("Test extract words prompt", mw)
    migrate_compound_verbs_action = QAction("Migrate compound verbs to prefix/suffix verbs", mw)
    match_words_action = QAction("Match extracted words to notes", mw)
    rematch_single_word_action = QAction("Rematch all single word to notes", mw)
    rematch_processed_single_word_action = QAction("Rematch processed single words to notes", mw)
    match_remaining_single_word_action = QAction(
        "Match remaining unprocessed single words to notes", mw
    )
    find_missing_matched_note_ids_action = QAction(
        "Find missing matched note ids for selected notes", mw
    )
    tag_notes_matched_status_action = QAction("Tag notes matched status", mw)
    deduplicate_existing_meaning_notes_action = QAction("Deduplicate existing meaning notes", mw)
    export_kanjify_ft_action = QAction("Export kanjify fine-tuning data", mw)
    export_extract_words_ft_action = QAction("Export extract-words fine-tuning data", mw)
    make_all_meanings_action = QAction("Generate all meanings for selected notes", mw)
    merge_meanings_action = QAction("Merge existing meanings for selected notes", mw)
    new_note_all_ops_action = QAction("Run all ops for new notes", mw)

    # Connect the action to the operation
    selected_nids = browser.selectedNotes()
    qconnect(
        meaning_action.triggered,
        lambda: clean_selected_notes(selected_nids, parent=browser),
    )
    qconnect(
        translation_action.triggered,
        lambda: translate_selected_notes(selected_nids, parent=browser),
    )
    qconnect(
        kanji_story_action.triggered,
        lambda: make_stories_for_selected_notes(selected_nids, parent=browser),
    )
    qconnect(
        component_words_action.triggered,
        lambda: kanjify_selected_notes(selected_nids, parent=browser),
    )
    qconnect(
        extract_words_action.triggered,
        lambda: extract_words_from_selected_notes(selected_nids, parent=browser),
    )
    qconnect(
        extract_words_test_compare_action.triggered,
        lambda: extract_words_test_compare_from_selected_notes(selected_nids, parent=browser),
    )
    qconnect(
        migrate_compound_verbs_action.triggered,
        lambda: migrate_compound_verbs_from_selected_notes(selected_nids, parent=browser),
    )
    qconnect(
        match_words_action.triggered,
        lambda: match_words_to_notes_from_selected(selected_nids, parent=browser),
    )
    qconnect(
        rematch_single_word_action.triggered,
        lambda: match_single_word_to_notes_from_selected(
            selected_nids, parent=browser, reprocess_words="both"
        ),
    )
    qconnect(
        rematch_processed_single_word_action.triggered,
        lambda: match_single_word_to_notes_from_selected(
            selected_nids, parent=browser, reprocess_words="only_processed"
        ),
    )
    qconnect(
        match_remaining_single_word_action.triggered,
        lambda: match_single_word_to_notes_from_selected(
            selected_nids, parent=browser, reprocess_words="only_unprocessed"
        ),
    )
    qconnect(
        make_all_meanings_action.triggered,
        lambda: make_meanings_selected_notes(selected_nids, parent=browser),
    )
    qconnect(
        merge_meanings_action.triggered,
        lambda: merge_meanings_selected_notes(selected_nids, parent=browser),
    )
    qconnect(
        new_note_all_ops_action.triggered,
        lambda: new_note_all_ops_selected_notes(selected_nids, parent=browser),
    )
    qconnect(
        find_missing_matched_note_ids_action.triggered,
        lambda: find_missing_matched_note_ids_selected_notes(selected_nids, parent=browser),
    )
    qconnect(
        tag_notes_matched_status_action.triggered,
        lambda: tag_notes_matched_status_from_selected(selected_nids, parent=browser),
    )
    qconnect(
        deduplicate_existing_meaning_notes_action.triggered,
        lambda: deduplicate_existing_meaning_notes_selected_notes(selected_nids, parent=browser),
    )
    qconnect(
        export_kanjify_ft_action.triggered,
        lambda: make_kanjify_sentence_fine_tuning_data(selected_nids, parent=browser),
    )
    qconnect(
        export_extract_words_ft_action.triggered,
        lambda: make_extract_words_fine_tuning_data(selected_nids, parent=browser),
    )

    ai_menu = menu.addMenu("AI helper")
    if ai_menu is None:
        logger.error("Error: AI helper menu could not be created.")
        return
    # Add the action to the browser's card context menu

    # Async ops
    ai_menu.addAction(meaning_action)
    ai_menu.addAction(translation_action)
    ai_menu.addAction(kanji_story_action)
    ai_menu.addAction(component_words_action)
    ai_menu.addAction(extract_words_action)
    ai_menu.addAction(extract_words_test_compare_action)
    ai_menu.addAction(migrate_compound_verbs_action)
    ai_menu.addAction(match_words_action)
    ai_menu.addAction(rematch_single_word_action)
    ai_menu.addAction(rematch_processed_single_word_action)
    ai_menu.addAction(match_remaining_single_word_action)
    ai_menu.addAction(make_all_meanings_action)
    ai_menu.addAction(merge_meanings_action)
    ai_menu.addAction(new_note_all_ops_action)
    ai_menu.addSeparator()
    # Sync ops
    ai_menu.addAction(find_missing_matched_note_ids_action)
    ai_menu.addAction(tag_notes_matched_status_action)
    ai_menu.addAction(deduplicate_existing_meaning_notes_action)
    ai_menu.addAction(export_kanjify_ft_action)
    ai_menu.addAction(export_extract_words_ft_action)


def run_op_on_field_unfocus(changed: bool, note: Note, field_idx: int):
    handler = create_call_log_handler("add_note")
    logger = logging.getLogger(__name__)

    if handler:
        # Replace the previous run's handler rather than stacking another one on top
        close_previous_log_handlers(logger)
        logger.addHandler(handler)

    note_type = note.note_type()
    if not note_type:
        return
    note_type_name = note_type["name"]
    config = mw.addonManager.getConfig(__name__)
    if not config:
        logger.error("Error: Missing addon configuration")
        return

    field_name = note_type["flds"][field_idx]["name"]
    cur_field_value = note[field_name]

    if note_type_name == "Kanji draw":
        story_field = get_field_config(config, "story_field", note_type)
        if field_name == story_field and cur_field_value == "":
            return make_story_for_note(config, note, {}, {})

    if note_type_name == "Japanese vocab note":
        translated_sentence_field = get_field_config(config, "translated_sentence_field", note_type)
        if field_name == translated_sentence_field and cur_field_value == "":
            return translate_sentence_in_note(config, note, {}, {})


def run_op_on_add_note(note: Note):
    handler = create_call_log_handler("add_note")
    logger = logging.getLogger(__name__)

    if handler:
        # Replace the previous run's handler rather than stacking another one on top
        close_previous_log_handlers(logger)
        logger.addHandler(handler)

    note_type = note.note_type()
    if not note_type:
        return
    note_type_name = note_type["name"]
    config = mw.addonManager.getConfig(__name__)
    if not config:
        logger.error("Error: Missing addon configuration")
        return

    if note_type_name == "Japanese vocab note":
        if note.has_tag("new_matched_jp_word"):
            # If the note has the tag, don't run the ops as this is happening within the
            # match_words_to_notes and causes some problems
            logger.info("Skipping ops for note with 'new_matched_jp_word' tag")
            return
        notes_to_update_dict: dict[NoteId, Note] = {}
        try:
            clean_meaning_in_note(config, note, {}, notes_to_update_dict)
            extract_words_in_note(config, note, {}, notes_to_update_dict)
        except Exception as e:
            logger.error(
                f"Error in clean_meaning_in_note or extract_words_in_note: {e}", exc_info=True
            )
        if notes_to_update_dict:
            updated_notes = list(notes_to_update_dict.values())
            # Filter out the added note itself from the updated notes
            updated_notes = [n for n in updated_notes if n.id != note.id]
            logger.info(f"Updating {len(updated_notes)} notes after adding new note")
            mw.col.update_notes(updated_notes)


# Register to card adding hook
hooks.note_will_be_added.append(lambda _col, note, _deck_id: run_op_on_add_note(note))

# hooks.note_will_be_added.append(lambda _col, note, _deck_id: translate_sentence_in_note(
# note, config=mw.addonManager.getConfig(__name__)))

# Register to context menu initialization hook
gui_hooks.browser_will_show_context_menu.append(on_browser_will_show_context_menu)

# Register to field unfocus hook
gui_hooks.editor_did_unfocus_field.append(run_op_on_field_unfocus)
