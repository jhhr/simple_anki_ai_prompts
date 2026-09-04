import logging
from typing import Union
from anki.notes import Note, NoteId
from anki.collection import Collection
from aqt import mw
from aqt.browser import Browser
from aqt.utils import showWarning
from collections.abc import Sequence

from .base_ops import (
    get_response,
    bulk_notes_op,
    selected_notes_op,
    AsyncTaskProgressUpdater,
)
from ..utils import get_field_config

logger = logging.getLogger(__name__)


def get_translated_field_from_model(config: dict[str, str], sentence: str) -> Union[str, None]:
    return_field = "english_sentence"
    # HTML-keeping prompt
    no_html_prompt = (
        f"sentence_to_translate_into_english: {sentence}\n\nIgnore any HTML in the"
        " sentence.\nReturn an HTML-free English translation of the sentence in a JSON string as"
        f' the value of the key "{return_field}".'
    )
    model = config.get("translate_sentence_model", "")
    result = get_response(model, no_html_prompt)
    if result is None:
        # If translation failed, return nothing
        return None
    try:
        return result[return_field]
    except KeyError:
        return None


def translate_sentence_in_note(
    config: dict,
    note: Note,
    notes_to_add_dict: dict[str, list[Note]],
    notes_to_update_dict: dict[NoteId, Note],
) -> bool:
    note_type = note.note_type()
    if not note_type:
        logger.error(f"note_type() call failed for note {note.id}")
        return False
    try:
        sentence_field = get_field_config(config, "sentence_field", note_type)
        translated_sentence_field = get_field_config(config, "translated_sentence_field", note_type)
    except Exception as e:
        logger.error(str(e))
        return False

    logger.debug(f"sentence_field in note: {sentence_field in note}")
    logger.debug(f"translated_sentence_field in note: {translated_sentence_field in note}")
    # Check if the note has the required fields
    if sentence_field in note and translated_sentence_field in note:
        logger.debug("note has fields")
        # Get the values from fields
        sentence = note[sentence_field]
        logger.debug(f"sentence: {sentence}")
        # Check if the value is non-empty
        if sentence:
            # Call API to get translation
            translated_sentence = get_translated_field_from_model(config, sentence)
            logger.debug(f"translated_sentence: {translated_sentence}")
            if translated_sentence is not None:
                # Update the note with the new value
                note[translated_sentence_field] = translated_sentence
                if note.id != 0 and note.id not in notes_to_update_dict:
                    notes_to_update_dict[note.id] = note
                return True
            return False
        return False
    else:
        logger.error("note is missing fields")
    return False


def bulk_translate_notes_op(
    col: Collection,
    notes: Sequence[Note],
    edited_nids: list[NoteId],
    progress_updater: AsyncTaskProgressUpdater,
    notes_to_add_dict: dict[str, list[Note]] = {},
    notes_to_update_dict: dict[NoteId, Note] = {},
):
    config = mw.addonManager.getConfig(__name__)
    if not config:
        showWarning("Missing addon configuration")
        return
    message = "Translating sentences"
    op = translate_sentence_in_note
    return bulk_notes_op(
        message,
        config,
        op,
        col,
        notes,
        edited_nids,
        progress_updater,
        notes_to_add_dict,
        notes_to_update_dict,
    )


def translate_selected_notes(nids: Sequence[NoteId], parent: Browser):
    progress_updater = AsyncTaskProgressUpdater(title="Async AI op: Translating sentences")
    done_text = "Updated translation"
    bulk_op = bulk_translate_notes_op
    return selected_notes_op(done_text, bulk_op, nids, parent, progress_updater)
