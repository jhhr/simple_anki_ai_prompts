import json
import logging
from collections.abc import Sequence
from pathlib import Path

from anki.collection import Collection
from anki.notes import Note, NoteId
from aqt import mw
from aqt.browser import Browser
from aqt.utils import showWarning

from ..configuration import KANJI_STORY_COMPONENT_WORDS_LOG
from ..utils import get_field_config
from .base_ops import (
    AsyncTaskProgressUpdater,
    bulk_notes_op,
    get_response,
    selected_notes_op,
)

logger = logging.getLogger(__name__)


KANJI_STORY_DEFAULT_TEMPERATURE = 0.8


def get_component_words_section(components: list[str], components_dict: dict[str, str]) -> str:
    component_words = ", ".join(
        [components_dict.get(component, component) for component in components]
    )
    return (
        f"  component_radicals_or_kanji: {components}\n  words_to_use_in_story_for_components:"
        f" {component_words}"
    )


def get_kanji_story_from_model(
    config: dict[str, str],
    kanji: str,
    components: str,
    current_story: str,
) -> str:
    media_path = Path(mw.pm.profileFolder(), "collection.media")
    # Get stored dict of words used for component in the kanji_story_component_words.log file
    with open(Path(media_path, KANJI_STORY_COMPONENT_WORDS_LOG), "r", encoding="utf-8") as f:
        try:
            component_words_dict = json.loads(f.read())
        except json.JSONDecodeError as e:
            print(f"Error reading component words dict: {e}")
            component_words_dict = {}

    return_field = "new_story"
    response_schema = {
        "type": "object",
        "properties": {
            return_field: {"type": "string"},
        },
        "required": [return_field],
        "additionalProperties": False,
    }

    prompt = (
        f"kanji: {kanji}"
        f"\n{get_component_words_section(components.split(','), component_words_dict)}"
    )
    if current_story:
        prompt += f"\ncurrent_story_in_japanese: {current_story}"
    prompt += (
        "\n"
        "\nThe kanji is made up of the radicals or kanji listed above."
        "\nFor each of those, there are words that can be used to refer to them in the mnemonic"
        " story for the kanji."
    )
    prompt += (
        "\nKeep the the current mnemonic story otherwise the same but complete it for the kanji"
        " using those words (if it isn't already using them)."
        if current_story
        else "\nCome up with a new mnemonic story in Japanese for the kanji using those words."
    )
    prompt += (
        "\n 1) You can inflect the component words to fit the sentence better."
        "\n 2) The story should be very short - a single sentence - and include all the"
        " components and then a word for the kanji itself."
        "\n 3) The story should be written in hiragana only."
        "\n 4) Each word and particle should be separated by a space to make it easier to read."
        "\n 5) The component words should be wrapped in <i> tags and the kanji word in <b> tags."
        "\n 5a) Ideally the kanji word be a single word, usually a kunyomi reading. but if there"
        " is no usable kunyomi reading, use a compound word."
        "\n 5b) For a compound word wrap the part where the kanji is used in <b> tags."
        "\n 6) If there are no words for a component, invent a word that fits the component in a"
        " memorable way."
        "\n"
        "\nIMPORTANT: The story doesn't need to be strictly grammatically correct, it can take"
        " poetic license to achieve the necessary word order and brevity. It doesn't need to be"
        " grammatically complex, sophisticated or even make sense. "
        "Instead, it should focus on being memorable by connecting the component words with the"
        " example word in a direct (though potentially fantastical) way that is easy to remember"
        " and visualize."
        "\nYOU MUST use the component words in the exact order they are listed in the components"
        " list, and then the kanji word at the end."
        "\n"
        "\nExamples of stories for other kanji:"
        "\n  kanji: 裾"
        f"\n{get_component_words_section(['衤', '居'], component_words_dict)}"
        "\n  story: <i>ころも</i> の なか に <i>いる</i> と、 ぬけた <b>すそ</b> が ひろがる。"
        "\n"
        "\n  kanji: 熱"
        f"\n{get_component_words_section(['廾', '灬'], component_words_dict)}"
        "\n  story: <i>どろだんご</i> が <i>れっか</i>したら、たかい <b>ねつ</b> が できる"
        "\n"
        "\n  kanji: 柳"
        f"\n{get_component_words_section(['木', '卯'], component_words_dict)}"
        "\n  story: <i>き</i> が <i>うさぎの みみ</i> の ように しなやか、<b>やなぎ</b>"
        "\n"
        "\n  kanji: 捩"
        f"\n{get_component_words_section(['扌', '戻'], component_words_dict)}"
        "\n  story: <i>て</i> が <i>もどせない</i>、そんなに <b>よじっている</b>。"
        "\n"
        "\n  kanji: 移"
        f"\n{get_component_words_section(['禾', '多'], component_words_dict)}"
        "\n  story: <i>のぎ</i> が <i>おおくて</i>、それ を くら に <b>うつして</b>みましょう。"
        "\n"
        "\n  kanji: 侶"
        f"\n{get_component_words_section(['亻', '呂'], component_words_dict)}"
        "\n  story: <i>ひと</i> の <i>せぼね</i> は あばらぼね の はん<b>りょ</b> だ。"
        "\n"
        "\n  kanji: 宮"
        f"\n{get_component_words_section(['宀', '呂'], component_words_dict)}"
        "\n  story: <i>したぎ かんむり</i> の したに <i>せぼね</i> の ような はしら が きゅうでんの"
        " いりぐちに たった"
        "\n"
        "\n  kanji: 蹴"
        f"\n{get_component_words_section(['足', '就'], component_words_dict)}"
        "\n  story: <i>あし</i> の <i>しゅうしょく</i> は もの を <b>ける</b> こと。"
        "\n"
        "\n  kanji: 諭"
        f"\n{get_component_words_section(['言', '俞'], component_words_dict)}"
        "\n  story: <i>いいたい</i> こと を <i>いやしの こぶね</i> を こぎ ながら つたえる と、"
        "せんちょう が しずに してくれと <b>さとした</b>"
        "\n"
        "\n kanji: 嘘"
        f"\n{get_component_words_section(['口', '虚'], component_words_dict)}"
        "\n  story: <i>くち</i> から でる <i>むなしい</i> <b>うそ</b>..."
        "\n"
        "\n kanji: 勇"
        f"\n{get_component_words_section(['マ', '男'], component_words_dict)}"
        "\n  story: <i>ま！</i> <i>おとこらしい！</i> <b>いさましい</b> <b>ゆう</b>き を もっている ね"
        "\n"
        "\n kanji: 鯉"
        f"\n{get_component_words_section(['魚', '里'], component_words_dict)}"
        "\n  story: <i>さかな</i>, <i>さと</i> の いけ で はっしゃぐ、<b>こい</b> だ。"
        "\n"
        f'\nReturn the new story in a JSON string as the value of the key "{return_field}".'
    )
    model = config.get("kanji_story_model", "")
    config_temp = config.get("kanji_story_temperature", None)
    if config_temp is not None:
        try:
            temperature = float(config_temp)
        except ValueError:
            logger.error(
                "Invalid kanji story temperature in config: %s. Using default temperature: %f",
                config_temp,
                KANJI_STORY_DEFAULT_TEMPERATURE,
            )
            temperature = KANJI_STORY_DEFAULT_TEMPERATURE
    else:
        temperature = KANJI_STORY_DEFAULT_TEMPERATURE

    result = get_response(
        model,
        prompt,
        response_schema=response_schema,
        temperature=temperature,
    )
    if result is None:
        # Return original story unchanged if the cleaning failed
        return current_story
    try:
        return result[return_field]
    except KeyError:
        return current_story


def make_story_for_note(
    config: dict[str, str],
    note: Note,
    notes_to_add_dict: dict[str, list[Note]],
    notes_to_update_dict: dict[NoteId, Note],
) -> bool:
    model = note.note_type()
    if not model:
        logger.error("Missing note type for note", note.id)
        return False

    try:
        components_field = get_field_config(config, "components_field", model)
        kanji_field = get_field_config(config, "kanji_field", model)
        story_field = get_field_config(config, "story_field", model)
    except Exception as e:
        print(e)
        return False

    logger.debug(f"making story in note {note.id}")
    logger.debug(f"components_field in note {components_field in note}")
    logger.debug(f"kanji_field in note {kanji_field in note}")
    logger.debug(f"story_field in note {story_field in note}")
    # Check if the note has the required fields
    if components_field in note and kanji_field in note and story_field in note:
        logger.debug("note has fields")
        # Get the values from fields
        components = note[components_field]
        kanji = note[kanji_field]
        current_story = note[story_field]
        # Check if the value is non-empty
        if components:
            new_story = get_kanji_story_from_model(config, kanji, components, current_story)

            # Update the note with the new value
            note[story_field] = new_story
            # Return success, if the we changed something
            if new_story != current_story:
                if note.id != 0 and note.id not in notes_to_update_dict:
                    notes_to_update_dict[note.id] = note
                return True
            return False
        return False

    else:
        logger.error("note is missing fields")
    return False


def bulk_make_stories_op(
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
    message = "Updated stories"
    op = make_story_for_note
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


def make_stories_for_selected_notes(nids: Sequence[NoteId], parent: Browser):
    progress_updater = AsyncTaskProgressUpdater(title="Async AI op: Making kanji stories")
    done_text = "Updated stories"
    bulk_op = bulk_make_stories_op
    return selected_notes_op(done_text, bulk_op, nids, parent, progress_updater)
