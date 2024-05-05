from aqt.browser import Browser
from collections.abc import Sequence
from aqt import mw
from anki.notes import Note, NoteId

from .base_ops import (
    get_response_from_chat_gpt,
    bulk_notes_op,
    selected_notes_op,
)

DEBUG = True

def get_single_meaning_from_chat_gpt(vocab, sentence, dict_entry):
    return_field = "cleaned_meaning"
    prompt = """word: {vocab}
    sentence: {sentence}
    dictionary_entry_for_word: {dict_entry}

    The dictionary entry may contain multiple meanings for the word.
    Extract the one meaning matching the usage of the word in the sentence.
    Omit any example sentences the matching meaning included.
    If the meaning is more than four sentences long, shorten it to explain the basics only.
    Otherwise, keep an already short meaning as-is.
    In case there is only meaning, return that.

    Return the extracted meaning in a JSON string as the value of the key \"{return_field}\".
    """.format(
        vocab=vocab, sentence=sentence, dict_entry=dict_entry, return_field=return_field
    )
    result = get_response_from_chat_gpt(prompt, return_field)
    if result is None:
        # Return original dict_entry unchanged if the cleaning failed
        return dict_entry
    return result


def generate_meaning_from_chatGPT(vocab, sentence):
    return_field = "new_meaning"
    prompt = """word: {vocab}
    sentence: {sentence}

    Genarate a short dictionary-style explanation of the usage pattern fitting how the word is used in the sentence.
    The explanation should be in the same language as the sentence.
    Generally aim to explain the meaning shortly in a single sentence.
    If it is necessary to explain more, the maximum length should be 3 sentences.

    Return the meaning in a JSON string as the value of the key \"{return_field}\".
    Translate the meaning you generated into English as the value of the key "english_meaning".
    """.format(
        vocab=vocab, sentence=sentence, dict_entry=dict_entry, return_field=return_field
    )
    result = get_response_from_chat_gpt(prompt, return_field)
    if result is None:
        # Return original dict_entry unchanged if the cleaning failed
        return dict_entry
    return result


def clean_meaning_in_note(note: Note, config):
    model = mw.col.models.get(note.mid)
    meaning_field = config["meaning_field"][model["name"]]
    word_field = config["word_field"][model["name"]]
    sentence_field = config["sentence_field"][model["name"]]
    if DEBUG:
        print("cleaning meaning in note", note.id)
        print("meaning_field in note", meaning_field in note)
        print("word_field in note", word_field in note)
        print("sentence_field in note", sentence_field in note)
    # Check if the note has the required fields
    if meaning_field in note and word_field in note and sentence_field in note:
        if DEBUG:
            print("note has fields")
        # Get the values from fields
        dict_entry = note[meaning_field]
        word = note[word_field]
        sentence = note[sentence_field]
        # Check if the value is non-empty
        if dict_entry:
            # Call API to get single meaning from the raw dictionary entry
            modified_meaning_jp = get_single_meaning_from_chat_gpt(
                word, sentence, dict_entry
            )

            # Update the note with the new value
            note[meaning_field] = modified_meaning_jp
            # Return success, if the we changed something
            if modified_meaning_jp != dict_entry:
                return True
            return False

        return False
    elif DEBUG:
        print("note is missing fields")
    return False


def bulk_clean_notes_op(col, notes: Sequence[Note], editedNids: list):
    config = mw.addonManager.getConfig(__name__)
    message = "Cleaning meaning"
    op = clean_meaning_in_note
    return bulk_notes_op(message, config, op, col, notes, editedNids)


def clean_selected_notes(nids: Sequence[NoteId], parent: Browser):
    title = "Cleaning done"
    done_text = "Updated meaning"
    bulk_op = bulk_clean_notes_op
    return selected_notes_op(title, done_text, bulk_op, nids, parent)
