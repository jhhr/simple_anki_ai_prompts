import json
import logging
from typing import Union
from collections.abc import Sequence

from anki.notes import Note, NoteId
from anki.collection import Collection
from aqt import mw
from aqt.browser import Browser
from aqt.utils import showWarning

from .base_ops import (
    get_response,
    bulk_notes_op,
    selected_notes_op,
    AsyncTaskProgressUpdater,
)
from ..utils import get_field_config
from .extract_words import format_word_list_dict

logger = logging.getLogger(__name__)


def get_split_verbs_from_model(
    compound_verb: list,
    verbs_list: list,
    config: dict,
) -> Union[dict, None]:
    """Call the AI model to identify which entries in verbs_list are the prefix and suffix
    components of the given compound verb.

    Args:
        compound_verb: A word tuple for the compound verb, e.g. ["連れて行く", "つれていく"].
        verbs_list: The current list of verb tuples to search in.
        config: The addon configuration dict.

    Returns:
        A dict with keys "prefix_verb" and "suffix_verb", each being the exact matching
        entry from verbs_list, or null if not found. Returns None on API failure.
    """
    model = config.get("migrate_compound_verbs_model", config.get("extract_words_model", ""))
    if not model:
        logger.error("No model configured for migrate_compound_verbs_model or extract_words_model")
        return None

    compound_verb_json = json.dumps(compound_verb, ensure_ascii=False)
    verbs_json = json.dumps(verbs_list, ensure_ascii=False)

    prompt = f"""You are given a Japanese compound verb and a list of simple verbs.

The compound verb is: {compound_verb_json}
The simple verbs list is: {verbs_json}

A compound verb is formed by combining a prefix verb (the leading component) and a suffix verb (the trailing component).
Examples:
- 連れて行く has prefix verb 連れる and suffix verb 行く
- 飲み込む has prefix verb 飲む and suffix verb 込む
- 飛び出す has prefix verb 飛ぶ and suffix verb 出す

Your task:
1. Identify the prefix verb (the leading component) of the compound verb.
2. Identify the suffix verb (the trailing component) of the compound verb.
3. Find each component in the simple verbs list, matching by verb form.

Return a JSON object with exactly these two fields:
{{
  "prefix_verb": <the exact entry from the verbs list that matches the prefix component, or null if not found>,
  "suffix_verb": <the exact entry from the verbs list that matches the suffix component, or null if not found>
}}

Return only the JSON object, no other text."""

    return get_response(model, prompt)


def migrate_compound_verbs_in_note(
    config: dict,
    note: Note,
    notes_to_add_dict: dict,
    notes_to_update_dict: dict,
) -> bool:
    """For each compound verb in the note's word list, call the AI to identify its prefix and
    suffix verb components. Move matched entries from 'verbs' to 'prefix_verbs' and
    'suffix_verbs' respectively, and empty 'compound_verbs'.

    The verbs list is updated after each AI call so subsequent calls see the already-moved
    verbs removed, matching the instruction that iteration must be done synchronously.

    Args:
        config: The addon configuration dict.
        note: The note to process.
        notes_to_add_dict: Unused; present to match the bulk op interface.
        notes_to_update_dict: Dict to register notes that were modified.

    Returns:
        True if the note was modified, False otherwise.
    """
    note_type = note.note_type()
    if not note_type:
        logger.error(f"Missing note type for note {note.id}")
        return False
    try:
        word_list_field = get_field_config(config, "word_list_field", note_type)
    except Exception as e:
        logger.error(str(e))
        return False

    if not word_list_field or word_list_field not in note:
        return False

    raw = note[word_list_field]
    if not raw:
        return False

    try:
        word_lists = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse word_list JSON for note {note.id}: {e}")
        return False

    if not isinstance(word_lists, dict):
        return False

    compound_verbs = word_lists.get("compound_verbs", [])
    if not isinstance(compound_verbs, list) or not compound_verbs:
        return False

    # Work with mutable copies; these are updated after each AI call
    verbs: list = list(word_lists.get("verbs", []))
    prefix_verbs: list = list(word_lists.get("prefix_verbs", []))
    suffix_verbs: list = list(word_lists.get("suffix_verbs", []))

    changed = False
    for compound_verb in compound_verbs:
        if not isinstance(compound_verb, (list, tuple)) or len(compound_verb) < 2:
            logger.warning(f"Skipping invalid compound_verb entry: {compound_verb}")
            continue

        logger.debug(f"Processing compound verb {compound_verb} with verbs list: {verbs}")
        result = get_split_verbs_from_model(list(compound_verb), verbs, config)
        if not isinstance(result, dict):
            logger.warning(f"Model returned non-dict for compound verb {compound_verb}: {result}")
            continue

        prefix_verb = result.get("prefix_verb")
        suffix_verb = result.get("suffix_verb")

        # Move prefix verb from verbs to prefix_verbs
        if prefix_verb and isinstance(prefix_verb, (list, tuple)):
            prefix_verb_list = list(prefix_verb)
            for i, v in enumerate(verbs):
                if list(v) == prefix_verb_list:
                    prefix_verbs.append(verbs.pop(i))
                    changed = True
                    logger.debug(f"Moved prefix verb {prefix_verb} to prefix_verbs")
                    break
            else:
                logger.debug(
                    f"Prefix verb {prefix_verb} not found in verbs list for compound"
                    f" {compound_verb}"
                )

        # Move suffix verb from verbs to suffix_verbs
        if suffix_verb and isinstance(suffix_verb, (list, tuple)):
            suffix_verb_list = list(suffix_verb)
            for i, v in enumerate(verbs):
                if list(v) == suffix_verb_list:
                    suffix_verbs.append(verbs.pop(i))
                    changed = True
                    logger.debug(f"Moved suffix verb {suffix_verb} to suffix_verbs")
                    break
            else:
                logger.debug(
                    f"Suffix verb {suffix_verb} not found in verbs list for compound"
                    f" {compound_verb}"
                )

    if not changed:
        logger.debug(f"No compound verbs migrated for note {note.id}")
        return False

    word_lists["verbs"] = verbs
    word_lists["prefix_verbs"] = prefix_verbs
    word_lists["suffix_verbs"] = suffix_verbs
    word_lists["compound_verbs"] = compound_verbs

    note[word_list_field] = format_word_list_dict(word_lists)
    if note.id != 0 and note.id not in notes_to_update_dict:
        notes_to_update_dict[note.id] = note
    return True


async def bulk_migrate_compound_verbs_op(
    col: Collection,
    notes: Sequence[Note],
    edited_nids: list[NoteId],
    progress_updater: AsyncTaskProgressUpdater,
    notes_to_add_dict: dict[str, list[Note]],
    notes_to_update_dict: dict[NoteId, Note],
):
    """Bulk operation that migrates compound_verbs to prefix_verbs/suffix_verbs for each note.

    Runs synchronously so that the verbs list state is correct for each sequential AI call
    within a note.
    """
    config = mw.addonManager.getConfig(__name__)
    if not config:
        showWarning("Missing addon configuration")
        return
    message = "Migrating compound verbs to prefix/suffix verbs"
    return await bulk_notes_op(
        message,
        config,
        migrate_compound_verbs_in_note,
        col,
        notes,
        edited_nids,
        progress_updater,
        notes_to_add_dict,
        notes_to_update_dict,
    )


def migrate_compound_verbs_from_selected_notes(nids: Sequence[NoteId], parent: Browser):
    progress_updater = AsyncTaskProgressUpdater(
        title="AI op: Migrating compound verbs to prefix/suffix verbs"
    )
    done_text = "Migrated compound verbs to prefix/suffix verbs"
    bulk_op = bulk_migrate_compound_verbs_op
    return selected_notes_op(done_text, bulk_op, nids, parent, progress_updater)
