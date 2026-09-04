import logging
import json
import re
from pathlib import Path
from typing import Optional, Union, Sequence
from anki.notes import Note, NoteId
from anki.collection import Collection
from aqt import mw
from aqt.browser import Browser
from aqt.utils import showWarning
from collections.abc import Sequence

from .collection_access import (
    find_notes as col_find_notes,
    get_note as col_get_note,
)
from .base_ops import (
    get_response,
    bulk_notes_op,
    selected_notes_op,
    AsyncTaskProgressUpdater,
)
from ..sync_local_ops.mdx_dictionary import mdx_helper
from ..configuration import (
    MEANINGS_DICT_FILE,
    NO_DICTIONARY_ENTRY_TAG,
    MEANING_MAPPED_TAG,
    GeneratedMeaningsDictType,
    GeneratedMeaningType,
    EnAndJPSentence,
    WordAndSentences,
    MakeMeaningsResult,
)
from .make_all_meanings import (
    write_meanings_dict_to_file,
    make_meaning_dict_key,
    revise_meanings_for_word,
)

from ..utils import get_field_config
from ..html_stripping import strip_html

logger = logging.getLogger(__name__)


def get_sentences_for_note(
    config: dict[str, str],
    note: Note,
    exclude_self: bool = False,
) -> list[EnAndJPSentence]:
    note_type = note.note_type()
    if not note_type:
        logger.error(f"note_type() call failed for note {note.id}")
        return []
    word_list_field = get_field_config(config, "word_list_field", note_type)
    sentence_field = get_field_config(config, "sentence_field", note_type)
    translated_sentence_field = get_field_config(config, "translated_sentence_field", note_type)

    def make_en_and_jp_sentence(note: Note) -> EnAndJPSentence:
        return EnAndJPSentence(
            jp_sentence=strip_html(note[sentence_field]),
            en_sentence=strip_html(note[translated_sentence_field]),
        )

    cur_note_sentence = make_en_and_jp_sentence(note)
    if note.id == 0:
        # New note, can't search for others using its ID
        if exclude_self:
            return []
        return [cur_note_sentence]
    query = f'"{word_list_field}:*{note.id}*" -nid:{note.id}'
    logger.debug(f"Getting sentences for note {note.id} with query: {query}")
    other_sentence_note_ids = col_find_notes(query)
    other_sentences = [] if exclude_self else [cur_note_sentence]
    for onid in other_sentence_note_ids:
        onote = col_get_note(onid)
        if sentence_field in onote and onote[sentence_field] not in other_sentences:
            other_sentences.append(make_en_and_jp_sentence(onote))
    return other_sentences


def get_other_meaning_notes(
    config: dict[str, str],
    note: Note,
    notes_to_add_dict: Optional[dict[str, list[Note]]] = None,
    notes_to_update_dict: Optional[dict[NoteId, Note]] = None,
    allow_reupdate_existing: bool = False,
    include_pending_notes: bool = True,
) -> list[Note]:
    """Get notes in the same meaning group as ``note``, excluding ``note`` itself."""
    note_type = note.note_type()
    if not note_type:
        logger.error(f"note_type() call failed for note {note.id}")
        return []

    word_sort_field = get_field_config(config, "word_sort_field", note_type)
    word_normal_field = get_field_config(config, "word_normal_field", note_type)
    word_reading_field = get_field_config(config, "word_reading_field", note_type)
    word_field = get_field_config(config, "word_field", note_type)
    if (
        not word_sort_field
        or not word_normal_field
        or not word_reading_field
        or not word_field
        or word_reading_field not in note
        or word_normal_field not in note
        or word_field not in note
    ):
        logger.error(
            f"Could not build meaning-group query for note {note.id}, missing fields in note"
        )
        return []

    meaning_notes_query = (
        f"{word_sort_field}:re:m\\d+ -{word_sort_field}:re:x\\d+"
        f' -nid:{note.id} "{word_reading_field}:{note[word_reading_field]}"'
        f' ("{word_normal_field}:{note[word_normal_field]}" OR'
        f' "{word_field}:{note[word_field]}")'
    )
    other_meaning_note_ids = col_find_notes(meaning_notes_query)
    notes_to_update_dict = notes_to_update_dict or {}
    other_meaning_notes = [
        (
            col_get_note(onid)
            if allow_reupdate_existing or onid not in notes_to_update_dict
            else notes_to_update_dict[onid]
        )
        for onid in other_meaning_note_ids
    ]

    if include_pending_notes and notes_to_add_dict:
        m_pattern = re.compile(r"m\d+")
        x_pattern = re.compile(r"x\d+")
        for pending_notes in notes_to_add_dict.values():
            for pending_note in pending_notes:
                if id(pending_note) == id(note):
                    continue
                if word_sort_field not in pending_note:
                    continue
                pending_sort = pending_note[word_sort_field]
                if not m_pattern.search(pending_sort) or x_pattern.search(pending_sort):
                    continue
                if word_reading_field not in pending_note:
                    continue
                if pending_note[word_reading_field] != note[word_reading_field]:
                    continue
                if (
                    word_normal_field in pending_note
                    and pending_note[word_normal_field] == note[word_normal_field]
                ) or (word_field in pending_note and pending_note[word_field] == note[word_field]):
                    other_meaning_notes.append(pending_note)

    logger.debug(
        f"Other meaning notes count: {len(other_meaning_notes)}, query: {meaning_notes_query}"
    )
    return other_meaning_notes


UpdateAllMeaningsResultType = dict[NoteId, tuple[str, str]]


def update_all_meanings_for_word(
    config: dict[str, str],
    word: str,
    reading: str,
    existing_note_meanings_dict: dict[NoteId, WordAndSentences],
    jp_mdx_dict_entry: Union[str, None],
) -> UpdateAllMeaningsResultType:
    """
    Receive a list of current meanings and sentences for a word, and have an AI model rework the
    meanings to better fit the sentences, using the dictionary definitions as reference.

    :param config: Addon configuration dictionary.
    :param word: The word or phrase being defined.
    :param reading: The reading of the word or phrase.
    :param existing_note_meanings_dict: A dictionary mapping note IDs to their meanings and example sentences.
    :param jp_mdx_dict_entry: The dictionary entry for the word or phrase. None, if not available.
    :param generated_meanings: An entry for the word from the generated meanings json file. None,
            if not available.
    return: A dictionary mapping note IDs to tuples of (new_japanese_meaning, new_english_meaning).
    """
    if not existing_note_meanings_dict:
        return {}
    # Format current meanings and sentences for the prompt
    meanings_and_sentences = ""
    # Sort by note id, smallest to largest, to have a consistent order
    meanings_dict_items = list(existing_note_meanings_dict.items())
    meanings_dict_items.sort(key=lambda x: x[0])
    meaning_index_to_note_id = {}
    for i, (note_id, ws) in enumerate(meanings_dict_items):
        meaning_index_to_note_id[i] = note_id
        sentences_formatted = ""
        if len(ws["sentences"]) > 0:
            for sen in ws["sentences"]:
                sentences_formatted += f"- JP: {sen['jp_sentence']} -- EN: {sen['en_sentence']}\n"
        else:
            sentences_formatted = ""
        meanings_and_sentences += f"""---
Meaning index {i + 1}:
Japanese meaning: {ws['jp_meaning'] or '(empty)'}
English meaning: {ws['en_meaning'] or '(empty)'}

Sentences:
{sentences_formatted}
"""

    meaning_index_field = "meaning_index"
    jp_meaning_return_field = "jp_meaning"
    en_meaning_return_field = "en_meaning"
    dict_reference_return_field = "dictionary_reference"

    prompt = f"""{f'''Below is the dictionary entry for a word or phrase, along with currently used meanings for groups of sentences containing that word or phrase. Your task is to rework the meanings to better fit the usage in the sentences, using the dictionary entry as reference.
For each meaning, either extract the relevant parts from the dictionary entry and rephrase those to better fit the sentences. Follow these rules:
- DO NOT OVERFIT the definitions to the sentences. Especially when the number of examples is a single sentence. Pick as many meanings as possible than can broadly fit the theme of the sentences.
- If the dictionary entry describes two usage patterns for this word or phrase - for example, one literal and one figurative - those should become one meaning where each is described shortly.
- If there are more than two usage patterns for this word or phrase, describe the one used in the sentences.
- Aggressively shorten and simplify the picked meanings as much as possible, ideally into 1 sentence and at most 2 (if describing both a literal and figurative usage), with more complex meanings being allowed more explanation.
- Omit any example sentences included in the dictionary entry (often included within 「」 brackets).
''' if jp_mdx_dict_entry else f'''Below are currently used meanings for groups of sentences containing a certain word or phrase. Your task is to rework the meanings to better fit the usage in the sentences.
Follow these rules:
- DO NOT OVERFIT the definitions to the sentences. Especially when the number of examples is a mere 1-3 sentences. Aim for general definitions that broadly fit the theme of the sentences.
- If there are two usage patterns for this word or phrase - for example, one literal and one figurative - those should become one meaning where each is described shortly.
- If there are more than two usage patterns for this word or phrase, describe the one used in the sentences.
- Shorten and simplify the meanings as much as possible, ideally into 1 sentence and at most 2 (if describing both a literal and figurative usage), with more complex meanings being allowed more explanation.
'''}

Return a JSON object with one `meanings` field containing an array of objects. Each object corresponds to a meaning and has the following fields:
- "{meaning_index_field}": The 1-based index of an used meaning. IMPORTANT: This should match the index of the meaning in below list exactly.
- "{jp_meaning_return_field}": The reworked Japanese meaning.
- "{en_meaning_return_field}": The reworked English meaning.
{f'- "{dict_reference_return_field}": Repeat the parts of the dictionary entry that were used as reference for reworking the meaning.' if jp_mdx_dict_entry else ''}

You do not need to rework all meanings, only those that seem not to fit well with the sentences; the array can be of any length, including empty if all meanings are fine.

Word or phrase (and its reading):
{word} ({reading})
---{f'''
Dictionary entry:
{jp_mdx_dict_entry}
---
''' if jp_mdx_dict_entry else ''}
Current meanings and sentences:
{meanings_and_sentences}"""
    logger.debug(f"Prompt for updating meanings: {prompt}")

    array_properties = {
        meaning_index_field: {"type": "integer"},
        jp_meaning_return_field: {"type": "string"},
        en_meaning_return_field: {"type": "string"},
    }
    array_required = [
        meaning_index_field,
        jp_meaning_return_field,
        en_meaning_return_field,
    ]
    if jp_mdx_dict_entry:
        array_properties[dict_reference_return_field] = {"type": "string"}
        array_required.append(dict_reference_return_field)

    response_schema = {
        "type": "object",
        "properties": {
            "meanings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": array_properties,
                    "required": array_required,
                    "additionalProperties": False,
                },
            },
        },
        "required": ["meanings"],
        "additionalProperties": False,
    }

    model = config.get("word_meaning_model", "")
    result = get_response(model, prompt, response_schema=response_schema)
    if result is None:
        # Return nothing if the call failed
        return {}
    updated_meanings: dict[NoteId, tuple[str, str]] = {}
    if isinstance(result, dict) and "meanings" in result and isinstance(result["meanings"], list):
        for meaning_obj in result["meanings"]:
            try:
                assert (
                    isinstance(meaning_obj, dict)
                    and meaning_index_field in meaning_obj
                    and isinstance(meaning_obj[meaning_index_field], int)
                    and jp_meaning_return_field in meaning_obj
                    and isinstance(meaning_obj[jp_meaning_return_field], str)
                    and meaning_obj[jp_meaning_return_field].strip() != ""
                    and en_meaning_return_field in meaning_obj
                    and isinstance(meaning_obj[en_meaning_return_field], str)
                    and meaning_obj[en_meaning_return_field].strip() != ""
                )
            except AssertionError:
                logger.warning(f"Invalid meaning object in result: {meaning_obj}")
                continue
            meaning_index = meaning_obj[meaning_index_field] - 1
            meaning_note_id: NoteId | None = meaning_index_to_note_id.get(meaning_index)
            if meaning_note_id is not None:
                updated_meanings[meaning_note_id] = (
                    meaning_obj[jp_meaning_return_field],
                    meaning_obj[en_meaning_return_field],
                )
            else:
                logger.warning(f"Meaning index {meaning_index + 1} has no corresponding note ID")
    elif not isinstance(result, dict):
        logger.warning("Updated meanings result was not a dictionary")
    elif "meanings" not in result:
        logger.warning("Updated meanings result missing 'meanings' field")
    elif not isinstance(result["meanings"], list):
        logger.warning("Updated meanings 'meanings' field was not a list")

    logger.debug(f"Updated meanings: {updated_meanings}")
    return updated_meanings


MatchMeaningsResultType = dict[NoteId, tuple[str, str, int]]


def match_meanings_to_generated_meanings(
    config: dict[str, str],
    word: str,
    reading: str,
    existing_note_meanings_dict: dict[NoteId, WordAndSentences],
    all_generated_meanings_dict: GeneratedMeaningsDictType,
    depth: int = 0,
) -> MatchMeaningsResultType:
    """
    Receive a list of current meanings and sentences for a word, and have an AI model rework the
    meanings to better fit the sentences, using the dictionary definitions as reference.

    :param config: Addon configuration dictionary.
    :param word: The word or phrase being defined.
    :param reading: The reading of the word or phrase.
    :param existing_note_meanings_dict: A dictionary mapping note IDs to their meanings and example sentences.
    :param jp_mdx_dict_entry: The dictionary entry for the word or phrase. None, if not available.
    :param generated_meanings: An entry for the word from the generated meanings json file.
    :param depth: The recursion depth, used to stop endlessly looping the meanings revision.
    return: A dictionary mapping note IDs to tuples of (new_japanese_meaning, new_english_meaning,
        mapping_score).
    """
    if not existing_note_meanings_dict:
        return {}
    word_key = make_meaning_dict_key(word, reading)
    generated_meanings = all_generated_meanings_dict.get(word_key, None)
    if not generated_meanings:
        logger.error("match_meanings_to_generated_meanings called with missing generated meanings")
        return {}
    # Turn generated meanings to dict by note id for easy access
    generated_meanings_by_en_meaning: dict[str, GeneratedMeaningType] = {}
    for gm in generated_meanings:
        if "en_meaning" in gm and gm["en_meaning"] is not None:
            generated_meanings_by_en_meaning[gm["en_meaning"]] = gm

    # Format current meanings and sentences for the prompt
    meanings_and_sentences = ""
    # Sort by note id, smallest to largest, to have a consistent order
    meanings_dict_items = list(existing_note_meanings_dict.items())
    meanings_dict_items.sort(key=lambda x: x[0])
    meaning_index_to_note_id = {}
    ws_meanings_by_en_meaning: dict[str, NoteId] = {}
    some_meanings_already_mapped = False
    for i, (note_id, ws) in enumerate(meanings_dict_items):
        ws_meanings_by_en_meaning[ws["en_meaning"]] = ws
        # Is this note ID already mapped to a generated meaning?
        to_generated_meaning = generated_meanings_by_en_meaning.get(ws["en_meaning"])
        to_generated_meaning_index = None
        if to_generated_meaning:
            to_generated_meaning_index = generated_meanings.index(to_generated_meaning)
            some_meanings_already_mapped = True
        meaning_index_to_note_id[i] = note_id
        sentences_formatted = ""
        if len(ws["sentences"]) > 0:
            for sen in ws["sentences"]:
                sentences_formatted += f"- JP: {sen['jp_sentence']} -- EN: {sen['en_sentence']}\n"
        else:
            sentences_formatted = ""
        meanings_and_sentences += f"""---
Meaning index {i + 1}{f" (ALREADY MAPPED to possible meaning index {to_generated_meaning_index + 1})" if to_generated_meaning_index is not None else ""}:
Japanese meaning: {ws['jp_meaning'] or '(empty)'}
English meaning: {ws['en_meaning'] or '(empty)'}

Sentences:
{sentences_formatted}
"""

    used_meaning_index_field = "used_meaning_index"
    possible_meaning_index_field = "possible_meaning_index"
    mapping_score_field = "mapping_score"

    # Make dict for getting the possible meanings by index indicated by the AI call result
    possible_meanings_index_to_obj: dict[int, GeneratedMeaningType] = {}
    generated_meanings_formatted = ""
    for i, gm in enumerate(generated_meanings):
        jp_meaning = gm.get("jp_meaning", "").strip()
        en_meaning = gm.get("en_meaning", "").strip()
        possible_meanings_index_to_obj[i] = gm
        already_mapped = en_meaning in ws_meanings_by_en_meaning
        generated_meanings_formatted += f"""---
Possible meaning index {i + 1}{f" (ALREADY MAPPED)" if already_mapped else ""}:
Japanese meaning: {jp_meaning}
English meaning: {en_meaning}
"""

    prompt = f"""Below is a list all possible meanings for a certain word or phrase (which was generated from a selection of Japanese dictionary entries), along with currently used meanings for groups of sentences containing that word or phrase. Your task is to match the current meanings to the possible meanings, selecting the best fitting possible meaning for each current meaning.
Your task is to create a mapping from current meanings to possible meanings. Each current meaning must map to exactly one possible meaning.{f'''
- Some of the current meanings are already mapped to possible meanings. The possible meaning indexes they are mapped to are indicated and the already in use possible meanings are also marked. You CANNOT use those possible meanings for other current meanings; you must use one of the remaining possible meanings that is not yet mapped
- You MUST return a mapping for each not-already mappedcurrent meaning, even if you think its current meaning is already perfect.''' if some_meanings_already_mapped else '''
- You MUST return a mapping for current meaning, even if you think its current meaning is already perfect.'''}
- Return a score of 1 to 5 for how well each possible meaning fits the sentences for the current meaning, where 5=perfect fit,3=acceptable fit,1=unacceptable fit. This information is used to detect insufficient possible meanings.

Return a JSON object with one `meanings` field containing an array of objects. Each object corresponds to a meaning and has the following fields:
- "{used_meaning_index_field}": The 1-based index of an item in the CURRENT MEANINGS AND SENTENCES list below.
- "{possible_meaning_index_field}": The 1-based index of the possible meaning from the ALL POSSIBLE MEANINGS list below that should replace the current meaning.
- "{mapping_score_field}": An integer score from 1 to 5 indicating how well the possible meaning fits the sentences for the current meaning.


WORD OR PHRASE (READING):
{word} ({reading})

ALL POSSIBLE MEANINGS:
{generated_meanings_formatted}
---
CURRENT MEANINGS AND SENTENCES:
{meanings_and_sentences}"""
    logger.debug(f"Prompt for updating meanings: {prompt}")

    array_properties = {
        used_meaning_index_field: {"type": "integer"},
        possible_meaning_index_field: {"type": "integer"},
        mapping_score_field: {"type": "integer"},
    }
    array_required = [
        used_meaning_index_field,
        possible_meaning_index_field,
        mapping_score_field,
    ]

    response_schema = {
        "type": "object",
        "properties": {
            "meanings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": array_properties,
                    "required": array_required,
                    "additionalProperties": False,
                },
            },
        },
        "required": ["meanings"],
        "additionalProperties": False,
    }

    model = config.get("word_meaning_model", "")
    result = get_response(model, prompt, response_schema=response_schema)
    if result is None:
        # Return nothing if the call failed
        return {}
    updated_meanings: dict[NoteId, tuple[str, str]] = {}
    if not isinstance(result, dict):
        logger.warning("Updated meanings result was not a dictionary")
        return updated_meanings
    if "meanings" not in result:
        logger.warning("Updated meanings result missing 'meanings' field")
        return updated_meanings
    if not isinstance(result["meanings"], list):
        logger.warning("Updated meanings 'meanings' field was not a list")
        return updated_meanings
    if isinstance(result, dict) and "meanings" in result and isinstance(result["meanings"], list):
        for meaning_obj in result["meanings"]:
            if not isinstance(meaning_obj, dict):
                logger.warning(f"Invalid meaning object in result: {meaning_obj}")
                continue
            if used_meaning_index_field not in meaning_obj or not isinstance(
                meaning_obj[used_meaning_index_field], int
            ):
                logger.warning(
                    f"Invalid or missing '{used_meaning_index_field}' in meaning object:"
                    f" {meaning_obj}"
                )
                continue
            meaning_index = meaning_obj[used_meaning_index_field] - 1
            if meaning_index not in meaning_index_to_note_id:
                logger.error(
                    f"Invalid '{used_meaning_index_field}' value in meaning object, does not"
                    f" correspond to any current meaning: {meaning_obj}"
                )
                continue
            if possible_meaning_index_field not in meaning_obj or not isinstance(
                meaning_obj[possible_meaning_index_field], int
            ):
                logger.warning(
                    f"Invalid or missing '{possible_meaning_index_field}' in meaning object:"
                    f" {meaning_obj}"
                )
                continue
            possible_meaning_index = meaning_obj[possible_meaning_index_field] - 1
            if possible_meaning_index not in possible_meanings_index_to_obj:
                logger.error(
                    f"Invalid '{possible_meaning_index_field}' value in meaning object, does not"
                    f" correspond to any possible meaning: {meaning_obj}"
                )
                continue
            if mapping_score_field not in meaning_obj or not isinstance(
                meaning_obj[mapping_score_field], int
            ):
                logger.warning(
                    f"Invalid or missing '{mapping_score_field}' in meaning object: {meaning_obj}"
                )
                continue
            mapping_score = meaning_obj[mapping_score_field]
            if mapping_score < 1 or mapping_score > 5:
                logger.warning(
                    f"Invalid '{mapping_score_field}' value in meaning object, must be 1-5:"
                    f" {meaning_obj}"
                )
                continue
            meaning_note_id: NoteId = meaning_index_to_note_id.get(meaning_index)
            possible_meaning_obj = possible_meanings_index_to_obj.get(possible_meaning_index)
            updated_meanings[meaning_note_id] = (
                possible_meaning_obj["jp_meaning"],
                possible_meaning_obj["en_meaning"],
                mapping_score,
            )

    # If some meanings got a bad score, revise the meanings and try again
    existing_note_meanings_bad_score_ws: dict[NoteId, WordAndSentences] = {}
    for nid in updated_meanings.keys():
        # A threshold of 3 sometimes leads to a loop of revisions with the score given being 3 again
        # so we'll reduce the revision attempts to just 1 in such cases
        if updated_meanings[nid][2] <= 3:
            existing_note_meanings_bad_score_ws[nid] = existing_note_meanings_dict[nid]
    # Prepare the good meanings to return if no revision is done
    # We'll return meanings with scores >2 instead of >3 as the most common case of stopping
    # revision is that the score couldn't be improved beyond 3, 3 is still an acceptable mapping
    updated_good_meanings: MatchMeaningsResultType = {}
    for nid in updated_meanings.keys():
        # Only include meanings with good mapping score
        if updated_meanings[nid][2] > 2:
            updated_good_meanings[nid] = updated_meanings[nid]
    if len(existing_note_meanings_bad_score_ws) > 0 and depth == 0:
        logger.debug(
            f"Some meanings for word='{word}', reading='{reading}' got a bad mapping score,"
            f" revising meanings again, depth={depth}"
        )
        revise_result = revise_meanings_for_word(
            config,
            word,
            reading,
            existing_note_meanings_bad_score_ws,
            all_generated_meanings_dict,
        )
        if revise_result == MakeMeaningsResult.SUCCESS:
            # Re-run the matching with the revised meanings, the all_generated_meanings_dict
            # has been updated
            return match_meanings_to_generated_meanings(
                config,
                word,
                reading,
                existing_note_meanings_dict,
                all_generated_meanings_dict,
                depth=depth + 1,
            )
        # If some error happened during revise, return just the good mappings
        logger.debug(
            f"Error during meanings revision, returning only good mappings: {updated_good_meanings}"
        )
        return updated_good_meanings
    if depth >= 1:
        logger.debug(
            f"Max recursion depth reached for matching meanings for word='{word}',"
            f" reading='{reading}', returning good mappings: {updated_good_meanings}"
        )
        return updated_good_meanings

    logger.debug(f"Updated meanings: {updated_good_meanings}")
    return updated_good_meanings


def get_single_meaning_from_mdx_dict_entry(
    config: dict[str, str],
    word: str,
    reading: str,
    sentences: list[EnAndJPSentence],
    jp_mdx_dict_entry: str,
    prev_en_meaning: str = "",
):
    jp_meaning_return_field = "cleaned_meaning"
    en_meaning_return_field = "english_meaning"
    logger.debug(f"Getting single meaning with {len(sentences)} sentences")
    sentences_formatted = ""
    if len(sentences) > 1:
        for sen in sentences:
            sentences_formatted += f"- JP: {sen['jp_sentence']} -- EN: {sen['en_sentence']}\n"
    else:
        sentences_formatted = (
            f"JP: {sentences[0]['jp_sentence']} -- EN: {sentences[0]['en_sentence']}"
        )
    prompt = f"""Below, the dictionary entry for the word or phrase may contain multiple meanings. Your task is to either 1) extract the one meaning 2) or combine and rephrase meanings matching the usage of the word in the sentence{'s' if len(sentences) > 1 else ''}.

Selection criteria:
- DO NOT overfit the definition to the sentence{'s' if len(sentences) > 1 else ''}, but rather pick as many meanings as possible that can broadly fit the theme of the sentence{'s' if len(sentences) > 1 else ''}.
- Figurative uses of literal meanings should be one definition: When there is a pair of meanings for this word or phrase, one literal and one figurative, always pick both and shorten their respective descriptions. Do this even if the sentence{'s' if len(sentences) > 1 else ''} only uses one of the meanings.
- In case there is only a single meaning, return that.
- Exclude any dictionary entries from consideration that are for a different word. These may be included in the list of dictionary entries below due to matching the reading but not the word itself.
- If all dictionary entries appear to be for different words, generate a new meaning following the simplification omission rules below.{' Use the current english meaning as reference for the new meaning.' if prev_en_meaning.strip() != "" else ''}

Omission rules:
- Omit any example sentences the matching meaning(s) included (often included within 「」 brackets).
- Descriptions of animals and plants are often scientific. From these omit descriptions on their ecology and only describe their appearance and type of plant/animal with simple language.

Simplification rules:
- YOU MUST Shorten and simplify the meaning as much as possible, ideally into 1 sentence and at most 2 (if describing both a literal and figurative usage), with more complex meanings being allowed more explanation.

Formatting rules:
- Clean off meaning numberings and other notation leaving only a plain text description.

Additionally, but only if it seems necessary, reword the English dictionary definition to fit the Japanese one. The English definition should ideally simply list equivalent words, if there are some, and only explain in sentences when it's necessary.

Return a JSON object with two fields:
Return the extracted and possibly modified Japanese meaning as the value of the key "{jp_meaning_return_field}".
Return the possibly modified English meaning as the value of the key "{en_meaning_return_field}".

Word or phrase (and its reading):
{word} ({reading})
---
Sentence{'s' if len(sentences) > 1 else ''}:
{sentences_formatted}
{f'''---
Current English meaning:
{prev_en_meaning}''' if prev_en_meaning.strip() != "" else ""}
---
Japanese dictionary entry:
{jp_mdx_dict_entry}
"""
    logger.debug(f"Prompt for cleaning meaning: {prompt}")
    model = config.get("word_meaning_model", "")
    result = get_response(model, prompt)
    if result is None:
        # Return original dict_entry unchanged if the cleaning failed
        return jp_mdx_dict_entry, prev_en_meaning
    try:
        return result[jp_meaning_return_field], result[en_meaning_return_field]
    except KeyError:
        return jp_mdx_dict_entry, prev_en_meaning


def get_new_meaning_from_model(
    config: dict[str, str],
    word: str,
    reading: str,
    sentences: list[str],
    prev_en_meaning: str = "",
) -> tuple[str, str]:
    logger.debug(f"Getting new meaning with {len(sentences)} sentences")
    jp_meaning_return_field = "new_meaning"
    en_meaning_return_field = "english_meaning"
    sentences_formatted = ""
    if len(sentences) > 1:
        for sen in sentences:
            sentences_formatted += f"- {sen}\n"
    else:
        sentences_formatted = sentences[0]
    prompt = f"""Below {'is a sentence' if len(sentences) == 1 else 'are sentences each'} containing a certain word or phrase. Your task is to generate a short monolingual dictionary style definition of the general meaning used in the sentence by the word or phrase.

- Generally aim to for the definition to be a single sentence. If it is necessary to explain more, the maximum length should be 3 sentences.
- Do not overfit the definition to the sentence{'s' if len(sentences) > 1 else ''}, but rather aim for a general definition that fits the usage in {'each sentence' if len(sentences) > 1 else 'the sentence'}.
- If there are two usage patterns for this word or phrase - for example, one literal and one figurative - describe both shortly.
- If there are more than two usage patterns for this word or phrase, describe the one used in the sentence.
- The word itself should not be used in the definition.

The definition should be in the same language as the sentence. {'Use the current english meaning as reference for the new japanese meaning. Improve the english meaning  as well, if needed.' if prev_en_meaning.strip() != "" else 'Also, generate a very short English translation of the meaning, ideally a list of equivalent words or phrases but explaining further, if necessary.'}


Return a JSON object with two fields:
Return the meaning as the value of the key "{jp_meaning_return_field}".
Return the English translation as the value of the key "{en_meaning_return_field}".

Word or phrase (and its reading):
{word} ({reading})
{f'''---
Current English meaning:
{prev_en_meaning}''' if prev_en_meaning.strip() != "" else ""}
---
Sentence{'s' if len(sentences) > 1 else ''}:
{sentences_formatted}
"""
    logger.debug(f"Prompt for new meaning: {prompt}")
    model = config.get("word_meaning_model", "")
    result = get_response(model, prompt)
    if result is None:
        # Return nothing if the generating failed
        return "", ""

    new_meaning = ""
    en_meaning = ""
    try:
        new_meaning = result[jp_meaning_return_field]
    except KeyError:
        print(f"Error: '{jp_meaning_return_field}' not found in the result")

    try:
        en_meaning = result[en_meaning_return_field]
    except KeyError:
        print(f"Error: '{en_meaning_return_field}' not found in the result")
    return new_meaning, en_meaning


def clean_meaning_in_note(
    config: dict[str, str],
    note: Note,
    notes_to_add_dict: dict[str, list[Note]],
    notes_to_update_dict: dict[NoteId, Note],
    all_generated_meanings_dict: GeneratedMeaningsDictType,
    allow_update_all_meanings: Optional[bool] = True,
    allow_reupdate_existing: Optional[bool] = False,
    other_meaning_notes: Optional[Sequence[Note]] = None,
) -> bool:
    """
    Clean or update the meaning field in a given note using an AI model.

    :param config: Addon configuration dictionary.
    :param note: Note object to process.
    :param notes_to_add_dict: Dict for managing notes to add, not used atm.
    :param notes_to_update_dict: Dict for managing notes to update.
    :param all_generated_meanings_dict: Dict of all generated meanings
            for reuse across multiple calls. Provided to avoid doing file operations during
            async operations and to avoid doing file reading in every op.
    :param allow_update_all_meanings: Allows running the update_all_meanings_for_word function.
    :param allow_reupdate_existing: Allows re-updating notes that are already marked as updated.
    :param other_meaning_notes: Optional replacement list of notes with meanings for the
        update_all_meanings_for_word function.
    :return: True if the note was modified, False otherwise.
    """
    note_type = note.note_type()
    if not note_type:
        logger.error(f"note_type() call failed for note {note.id}")
        return False

    try:
        meaning_field = get_field_config(config, "meaning_field", note_type)
        english_meaning_field = get_field_config(config, "english_meaning_field", note_type)
        word_field = get_field_config(config, "word_field", note_type)
        word_reading_field = get_field_config(config, "word_reading_field", note_type)
        sentence_field = get_field_config(config, "sentence_field", note_type)
        new_note_id_field = get_field_config(config, "new_note_id_field", note_type)
    except KeyError as e:
        logger.error(str(e))
        return False

    logger.debug(f"cleaning meaning in note {note.id}")

    # Check if the note has the required fields
    if (
        meaning_field in note
        and english_meaning_field in note
        and word_field in note
        and word_reading_field in note
        and sentence_field in note
        and new_note_id_field in note
    ):
        logger.debug(
            f"allow_update_all_meanings: {allow_update_all_meanings}, allow_reupdate_existing:"
            f" {allow_reupdate_existing}, note id: {note.id}, provided other_meaning_notes:"
            f" {other_meaning_notes is not None}"
        )
        mdx_helper.load_mdx_dictionaries_if_needed(
            config, show_progress=True, finish_progress=False
        )
        pick_dictionary = config.get("mdx_pick_dictionary", "all")
        # Get dictionary entry from mdx helper
        jp_mdx_dict_entry = mdx_helper.get_definition_text(
            word=note[word_field],
            reading=note[word_reading_field],
            pick_dictionary=pick_dictionary,
        )
        if note.id > 0 and note.id in notes_to_update_dict:
            note = notes_to_update_dict[note.id]
        if not jp_mdx_dict_entry:
            # Set tag to find notes without dictionary entry later
            note.add_tag(NO_DICTIONARY_ENTRY_TAG)
            if note.id > 0 and note.id not in notes_to_update_dict:
                notes_to_update_dict[note.id] = note

        word_key = make_meaning_dict_key(note[word_field], note[word_reading_field])
        word_generated_meanings = all_generated_meanings_dict.get(word_key, None)

        all_meaning_notes: list[Note] = [note]
        if other_meaning_notes is None and allow_update_all_meanings:
            other_meaning_notes = get_other_meaning_notes(
                config=config,
                note=note,
                notes_to_add_dict=notes_to_add_dict,
                notes_to_update_dict=notes_to_update_dict,
                allow_reupdate_existing=allow_reupdate_existing,
                include_pending_notes=True,
            )
            all_meaning_notes = other_meaning_notes + [note]

        meaning_sentences_dict = {
            (n.id if n.id != 0 else int(n[new_note_id_field])): WordAndSentences(
                jp_meaning=n[meaning_field],
                en_meaning=n[english_meaning_field],
                sentences=get_sentences_for_note(config, n),
            )
            for n in all_meaning_notes
        }

        def update_all_meanings_from_result_dict(
            update_dict: Union[UpdateAllMeaningsResultType, MatchMeaningsResultType],
        ) -> bool:
            any_changed_inner = False
            for n in all_meaning_notes:
                note_key = n.id if n.id != 0 else int(n[new_note_id_field])
                if note_key in update_dict:
                    mapping_score = None
                    # MatchMeaningsResultType
                    if len(update_dict[note_key]) == 3:
                        new_jp_meaning, new_en_meaning, mapping_score = update_dict[note_key]
                    else:
                        # UpdateAllMeaningsResultType
                        new_jp_meaning, new_en_meaning = update_dict[note_key]
                    prev_en_meaning = n[english_meaning_field]
                    prev_jp_meaning = n[meaning_field]
                    n[meaning_field] = new_jp_meaning
                    n[english_meaning_field] = new_en_meaning
                    n.add_tag("updated_jp_meaning")
                    if mapping_score is not None:
                        n.add_tag(f"meaning_mapping_score::{mapping_score}")
                        n.add_tag(MEANING_MAPPED_TAG)
                    # Meanings should be changing if mapping was done, so we don't check
                    # mapping_score here again
                    if new_jp_meaning != prev_jp_meaning or new_en_meaning != prev_en_meaning:
                        any_changed_inner = True
                        if n.id > 0 and n.id not in notes_to_update_dict:
                            notes_to_update_dict[n.id] = n
            return any_changed_inner

        if len(all_meaning_notes) > 1 and not word_generated_meanings:
            if not allow_reupdate_existing and note.id in notes_to_update_dict:
                logger.debug(
                    f"Skipping note {note.id} as it's already marked as updated by a previous op"
                )
                return False

            updated_meanings_dict = update_all_meanings_for_word(
                config,
                note[word_field],
                note[word_reading_field],
                meaning_sentences_dict,
                jp_mdx_dict_entry,
            )
            any_changed = update_all_meanings_from_result_dict(updated_meanings_dict)
            return any_changed
        elif word_generated_meanings:
            logger.debug(
                f"Using generated meanings for word {word_key} in note {note.id} to update meanings"
                f" generated meanings: {word_generated_meanings}"
            )
            # Check if all notes with this word have already been mapped to a generated meaning
            # Convert the list to a dict for easier checking
            generated_meanings_by_en_meaning: dict[str, GeneratedMeaningType] = {}
            for gm in word_generated_meanings:
                en_meaning = gm.get("en_meaning", None)
                if en_meaning is not None:
                    generated_meanings_by_en_meaning[en_meaning] = gm

            # Check if all notes are already mapped, mapping is determined by note's english meaning
            # matching exactly a generated meaning's english meaning
            all_mapped = True
            for n in all_meaning_notes:
                # Take this opportunity and update the note tags
                if n[english_meaning_field] in generated_meanings_by_en_meaning and not n.has_tag(
                    MEANING_MAPPED_TAG
                ):
                    n.add_tag(MEANING_MAPPED_TAG)
                    if n.id > 0 and n.id not in notes_to_update_dict:
                        notes_to_update_dict[n.id] = n

                if n[english_meaning_field] not in generated_meanings_by_en_meaning:
                    all_mapped = False
                    break
            if all_mapped:
                logger.debug(
                    f"All notes for word {word_key} already mapped to generated meanings, skipping"
                )
                return False
            if not allow_reupdate_existing and note.id in notes_to_update_dict:
                logger.debug(
                    f"Skipping note {note.id} as it's already marked as updated by a previous op"
                )
                return False

            updated_meanings_dict = match_meanings_to_generated_meanings(
                config,
                note[word_field],
                note[word_reading_field],
                meaning_sentences_dict,
                all_generated_meanings_dict,
            )
            any_changed = update_all_meanings_from_result_dict(updated_meanings_dict)
            return any_changed

        prev_en_meaning = note[english_meaning_field]
        word = note[word_field]
        reading = note[word_reading_field]
        sentences = get_sentences_for_note(config, note)
        # Check if the value is non-empty
        if jp_mdx_dict_entry:
            # Call API to get single meaning from the raw dictionary entry
            new_jp_meaning, new_en_meaning = get_single_meaning_from_mdx_dict_entry(
                config, word, reading, sentences, jp_mdx_dict_entry, prev_en_meaning
            )

            # Update the note with the new value
            note[meaning_field] = new_jp_meaning
            note[english_meaning_field] = new_en_meaning
            # Return success, if the we changed something
            if new_jp_meaning != jp_mdx_dict_entry or new_en_meaning != prev_en_meaning:
                if note.id > 0 and note.id not in notes_to_update_dict:
                    notes_to_update_dict[note.id] = note
                return True
            return False
        else:
            # If there's no dict_entry, we'll let a model generate one from scratch
            new_meaning, en_meaning = get_new_meaning_from_model(
                config, word, reading, sentences, prev_en_meaning
            )
            note[meaning_field] = new_meaning
            note[english_meaning_field] = en_meaning
            if new_meaning != "" or en_meaning != "":
                if note.id > 0 and note.id not in notes_to_update_dict:
                    notes_to_update_dict[note.id] = note
                return True
            return False

    else:
        logger.error("note is missing fields")
    return False


def bulk_clean_notes_op(
    col: Collection,
    notes: Sequence[Note],
    edited_nids: list,
    progress_updater: AsyncTaskProgressUpdater,
    notes_to_add_dict: dict[str, list[Note]],
    notes_to_update_dict: dict[NoteId, Note],
):
    config = mw.addonManager.getConfig(__name__)
    if not config:
        showWarning("Missing addon configuration")
        return
    message = "Cleaning meaning"

    media_path = Path(mw.pm.profileFolder(), "collection.media")
    all_meanings_dict_path = Path(media_path, MEANINGS_DICT_FILE)
    all_generated_meanings_dict: GeneratedMeaningsDictType = {}
    if all_meanings_dict_path.exists():
        with open(all_meanings_dict_path, "r", encoding="utf-8") as f:
            all_generated_meanings_dict = json.load(f)

    def op(
        config: dict[str, str],
        note: Note,
        notes_to_add_dict: dict[str, list[Note]],
        notes_to_update_dict: dict[NoteId, Note],
    ) -> bool:
        nonlocal all_generated_meanings_dict
        return clean_meaning_in_note(
            config,
            note,
            notes_to_add_dict,
            notes_to_update_dict,
            all_generated_meanings_dict,
        )

    def on_end():
        nonlocal all_generated_meanings_dict
        # Write updated meanings dictionary to file after successful operation
        write_meanings_dict_to_file(all_generated_meanings_dict)

    return bulk_notes_op(
        message,
        config,
        op,
        col,
        notes,
        edited_nids,
        progress_updater,
        notes_to_add_dict=notes_to_add_dict,
        notes_to_update_dict=notes_to_update_dict,
        on_end=on_end,
    )


def clean_selected_notes(nids: Sequence[NoteId], parent: Browser):
    progress_updater = AsyncTaskProgressUpdater(title="Async AI op: Cleaning meanings")
    done_text = "Updated meaning"
    bulk_op = bulk_clean_notes_op
    return selected_notes_op(done_text, bulk_op, nids, parent, progress_updater)
