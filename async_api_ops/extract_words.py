import json
import re
import logging
from collections import Counter
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
from ..configuration import (
    RawOneMeaningWordType,
    RawMultiMeaningWordType,
    OneMeaningMatchedWordType,
)

logger = logging.getLogger(__name__)

EXTRACT_WORDS_TEST_MATCH_TAG = "1-extract-words-test-match"
EXTRACT_WORDS_TEST_NO_MATCH_TAG = "1-extract-words-test-no-match"


def normalize_word_tuple_for_test_comparison(
    word_tuple: Sequence,
) -> Union[RawOneMeaningWordType, RawMultiMeaningWordType, None]:
    """Normalize one word tuple for test comparison.

    Matched tuples can include metadata (sort word and note id) that is not produced by the
    extraction prompt, so we strip that metadata for comparison.
    """
    if not isinstance(word_tuple, (list, tuple)) or len(word_tuple) < 2:
        return None
    word = word_tuple[0]
    reading = word_tuple[1]
    if not isinstance(word, str) or not isinstance(reading, str) or not word or not reading:
        return None

    # Keep a meaning index if present; otherwise compare by word + reading only.
    if len(word_tuple) >= 3:
        meaning_or_meta = word_tuple[2]
        try:
            meaning_number = int(str(meaning_or_meta).strip())
            return (word, reading, meaning_number)
        except ValueError:
            # Non-numeric 3rd element is treated as metadata-like and ignored.
            return (word, reading)

    return (word, reading)


def normalize_word_list_dict_for_test_comparison(word_lists: dict) -> dict[str, list[tuple]]:
    """Normalize a word list dict for deterministic order-insensitive comparison."""
    if not isinstance(word_lists, dict):
        return {}

    normalized: dict[str, list[tuple]] = {}
    for key, value in word_lists.items():
        if not isinstance(key, str):
            continue
        if not isinstance(value, list):
            normalized[key] = []
            continue

        normalized_tuples: list[tuple] = []
        for raw_tuple in value:
            normalized_tuple = normalize_word_tuple_for_test_comparison(raw_tuple)
            if normalized_tuple is not None:
                normalized_tuples.append(tuple(normalized_tuple))

        # Sort for stable debug output; comparison itself is order-insensitive.
        normalized[key] = sorted(
            normalized_tuples,
            key=lambda t: tuple(str(x) for x in t),
        )

    return normalized


def compare_normalized_word_lists_for_test(
    original_word_lists: dict[str, list[tuple]],
    test_word_lists: dict[str, list[tuple]],
) -> tuple[bool, dict[str, dict[str, list[tuple]]]]:
    """Compare normalized word lists per key and per tuple, ignoring list order."""
    all_keys = sorted(set(original_word_lists.keys()) | set(test_word_lists.keys()))
    diffs: dict[str, dict[str, list[tuple]]] = {
        "missing_in_test": {},
        "extra_in_test": {},
    }

    for key in all_keys:
        original_counter = Counter(tuple(x) for x in original_word_lists.get(key, []))
        test_counter = Counter(tuple(x) for x in test_word_lists.get(key, []))

        missing_in_test = sorted(
            list((original_counter - test_counter).elements()),
            key=lambda t: tuple(str(x) for x in t),
        )
        extra_in_test = sorted(
            list((test_counter - original_counter).elements()),
            key=lambda t: tuple(str(x) for x in t),
        )

        if missing_in_test:
            diffs["missing_in_test"][key] = missing_in_test
        if extra_in_test:
            diffs["extra_in_test"][key] = extra_in_test

    is_match = not diffs["missing_in_test"] and not diffs["extra_in_test"]
    return is_match, diffs


def word_tuple_sort_key(
    word_tuple: Sequence,
) -> tuple[
    Union[str, None], Union[str, None], Union[str, None], Union[int, None], Union[int, None]
]:
    """
    Sort key for word tuples. This is used to sort the words in the word list.
    The sort order is:
    1. word
    2. reading
    3. check sort_word and note_id, if they are present and sort them before those that lack them
    4. meaning_number, if present, should be sorted after the above
    """
    word, reading, meaning_number, sort_word, note_id = None, None, None, None, None
    if len(word_tuple) == 2:
        word, reading = word_tuple
    elif len(word_tuple) == 3:
        word, reading, meaning_number = word_tuple
    elif len(word_tuple) == 4:
        word, reading, sort_word, note_id = word_tuple
    # at least word and reading must be present
    if not word or not reading:
        # Some invalid data, sort last
        return (None, None, None, None, None)
    if sort_word is None:
        sort_word = ""
    if note_id is None:
        note_id = 0
    else:
        # If note_id is present, it should be a number, so convert it to int
        try:
            note_id = int(str(note_id).strip())
        except ValueError:
            logger.warning(f"Invalid note_id {note_id} for word {word}, setting to 0")
            note_id = 0
    if meaning_number is None:
        meaning_number = 0
    else:
        # If meaning_number is present, it should be a number, so convert it to int
        try:
            meaning_number = int(str(meaning_number).strip())
        except ValueError:
            logger.warning(f"Invalid meaning_number {meaning_number} for word {word}, setting to 0")
            meaning_number = 0
    # Return a tuple that can be used for sorting
    return (word, reading, sort_word, note_id, meaning_number)


def compared_word_lists(
    cur_word_list: list[tuple],
    new_word_list: list[tuple],
) -> list[Union[RawOneMeaningWordType, RawMultiMeaningWordType, OneMeaningMatchedWordType]]:
    """
    Compare two word lists and return a list of words that combine both lists, keeping all previous
    words and only adding new words that are not already in the current list.
    """
    if not cur_word_list:
        return new_word_list
    if not new_word_list:
        return cur_word_list
    # Make sets of the whole tuples in the lists
    cur_set = set(tuple(word) for word in cur_word_list)
    new_set = set(tuple(word) for word in new_word_list)
    # Get all tuples that are only in the new set
    added_set = new_set - cur_set
    # check each new tuple for validity
    added_list: list[Union[RawOneMeaningWordType, RawMultiMeaningWordType]] = []
    # If the current set already contains words with note id set we'll get duplicates from the
    # new set. Remove duplicates by checking for matching word + reading in the list with note_id
    seen_with_note_id = set(
        (w[0], w[1])
        for w in cur_set
        if (len(w) == 4 and w[3] is not None) or (len(w) == 5 and w[4] is not None)
    )
    for word_tuple in added_set:
        word, reading, meaning_number, sort_word, note_id = None, None, None, None, None

        if len(word_tuple) == 2:
            word, reading = word_tuple
        elif len(word_tuple) == 3:
            word, reading, meaning_number = word_tuple
        elif len(word_tuple) == 4:
            word, reading, sort_word, note_id = word_tuple
        elif len(word_tuple) == 5:
            word, reading, meaning_number, sort_word, note_id = word_tuple
        else:
            logger.warning(f"Word tuple with invalid length {word_tuple} in added_set, skipping")
            continue

        # at least word and reading must be present
        if word and reading:
            # sort_word and note_id should not be present in the added_set, any new words that had
            # should've matched an existing word in the current set, so a new word that has
            # them is invalid and was probably hallucinated by the model. If this happens a lot
            # the prompt's instructions should be adjusted as the model should be returning
            # existing words sort sort_word and note_id as-is
            if sort_word is not None or note_id is not None:
                logger.warning(
                    f"Invalid new word tuple with sort_word/note_id preset {word_tuple} in"
                    " added_set, skipping"
                )
                continue
            elif (word, reading) not in seen_with_note_id:
                # Only add the word if it's not already present in the current set with a note_id
                if meaning_number is not None:
                    added_list.append((word, reading, meaning_number))
                else:
                    # if meaning_number is not present, this is allowed
                    added_list.append((word, reading))
        else:
            logger.warning(
                f"Invalid new word tuple with missing word/reading {word_tuple} in added_set,"
                " skipping"
            )
            continue
    # Return the sorted combined list of current words and new words
    # Ensure all elements are tuples
    combined_list = [tuple(w) for w in cur_word_list] + [tuple(w) for w in added_list]

    combined_list.sort(key=word_tuple_sort_key)
    return combined_list


def format_word_list_dict(word_lists: dict) -> str:
    """Format a word list dict with one list element per row, per-category line. Returns '{}' for empty."""
    if not word_lists:
        return "{}"
    return (
        "{\n"
        + ",\n".join(
            f'  "{k}": ['
            + (f"\n    " if v else "")
            + ",\n    ".join(json.dumps(item, ensure_ascii=False) for item in v)
            + (f"\n  " if v else "")
            + "]"
            for k, v in word_lists.items()
        )
        + "\n}"
    )


def _format_test_diffs(diffs: dict) -> str:
    """Format the test comparison diff dict for debug logging."""
    parts = []
    for outer_key in ("missing_in_test", "extra_in_test"):
        inner = diffs.get(outer_key, {})
        inner_str = format_word_list_dict(inner)
        # Indent all lines except the opening brace so they nest under the key
        inner_lines = inner_str.split("\n")
        if len(inner_lines) > 1:
            indented = inner_lines[0] + "\n" + "\n".join("  " + line for line in inner_lines[1:])
        else:
            indented = inner_str
        parts.append(f'  "{outer_key}": {indented}')
    return "{\n" + ",\n".join(parts) + "\n}"


def word_lists_str_format(
    word_lists: dict,
) -> Union[str, None]:
    """
    Convert the word list dict into the same format used in the prompt
    """
    if not word_lists:
        return None
    return format_word_list_dict(word_lists)


def get_extract_words_prompt(sentence: str) -> str:
    return f"""Below is a Japanese sentence that contains furigana in brackets after kanji words. Your task is to examine each word and phrase in the sentence, categorize each into either nouns, proper nouns, numbers, counters, verbs, prefix verbs (leading verb component of compound verb constructions), suffix verbs (trailing verb component of compound verb constructions), adjectives, adverbs, adjectivals, particles (and copula), conjunctions, pronouns, suffixes, prefixes, idiomatic expressions or common phrases and 4-kanji idioms (yojijukugo). You will convert convert inflected words into their dictionary forms. When two or more words are both homophones and homographs a number is added to indicate that they are different meanings.

More details on the categorization
- Compound words, expressions or aphorisms should be listed as well, along with their components. That is, if "XYZ" is such a sequence and "XY" and "Z" are valid words, include "XYZ","XY" and "Z" in the result.
- However, don't list compound words that do not form a significantly different meaning from their components. For example, from 委員会議長[いいんかいぎちょう] the words to list would be just 委員会 ("committee") and 議長 ("chairman") as the compound is simply "committee chairman" and thus perfectly described by the two components.
- Compound verbs that should be split are only those where the prefix is in 1) noun form, e.g. 切り替える --> ["切る~","きる"] in prefix_verbs, ["~替える","かえる"] in suffix_verbs or 2) a ます stem form, e.g. 受け入れる --> ["受ける~","うける"] in prefix_verbs, ["~入れる","いれる"] in suffix_verbs.
- Compound verbs that should not be split are 1) those where the prefix is in て-form, e.g. 応えてあげる --> ["応える","こたえる"] and ["あげる","あげる"] in verbs. 2) where the prefix is not a verb e.g. a <noun>を<verb> construction like 尻餅を突く --> ["尻餅","しりもち"] in nouns and ["突く","つく"] in verbs.
- Any compound verbs with 過ぎる as the suffix, do not list the compound verb, list both the prefix and 過ぎる as normal verbs.
- Don't list verbs いる, される or しまう when they occur as auxiliary verbs in verbs inflected forms, e.g. 食べている, 行かせる.
- する verbs are to be listed as nouns and the する verb ignored.
- Avoid listing words ending in particles or copula, as this would create many variants of the same word. The exceptions would be when the copula/particle-added form is overwhelmingly more common than the word being used without the particle/copula. For example, with the particle に, the adverb 共に is overhelmingly more common over the plain noun form 共, so whenever 共に occurs, 共に and not 共 should be listed. Only, if 共 were to occur alone (not as part of a compound word), it should be listed.
- Don't list verbs in たい, たくない, せる or other non-base forms, except when such a form has a special meaning. Examples of special meanings, 食えない "shrewd" vs literal "cannot eat", 唸らせる "to impress" vs literal "to make someone groan". Example of non-special meaning: 齧らせる is simply "to make someone bite"
- Don't list adjectives in さ form, list them in their い-form. Avoid listing adjectives in く-form as well, excpect when they have a meaning that isn't merely adverbial; for example, 大きく has the meaning "on a grand scale / extensively"
- Don't list nouns which may take the genitive case の with the particle, list them in their plain form. For example, 上の should be listed as just 上.
- Don't list noun form verbs, list them as verbs. For example. For example 冷やかし should be listed as 冷やかす.
- Only list proper nouns a single time, ignoring their component nouns.
- Don't list words in two categories, e.g. an adjective yojijukugo should only be listed in yojijukugo. Exception: a verb may appear in both verbs and prefix_verbs or suffix_verbs if it occurs both standalone and as a component of a compound verb construction in the same sentence.
- List 4-kanji idioms only once as well, disregarding 2-kanji words that they may contain.
- Otherwise ignore any HTML that may be in the text, leaving any HTML out of the word lists.
- A word occuring twice or more with the same kanji form and reading needs to considered for homonymity. If it is a used in the same meaning, the word should be listed just once. If the meanings differ, the word listed once for each different meaning, with a 1-based index number included to differentiate them. For example, 行く as "physically move to a place" vs "participate in an activity" vs "reach a point (in an activity, not physical place)".
- Note, homonym listing of individual words can only be done, if a word actually occurs more than once.
- Additionally, a word occuring twice with the same meaning but, for some reason in kanji form and in hiragana, should result in one entry using the kanji form.
- Ensure that you use the correct base reading for words, not a rendaku or otherwise altered reading. For example, 中 used as a suffix can often be じゅう but the base reading is ちゅう.

This example includes handling a compound verb that is not split into prefix and suffix verbs.
Example sentence 1: 私[わたし]も<b> 連[つ]れて 行[い]って</b><k> 下[くだ]さい</k>。
Example results 1:
{{
  "nouns": [],
  "proper_nouns": [],
  "numbers": [],
  "counters": [],
  "verbs": [["連れる","つれる"],["行く","いく"]],
  "prefix_verbs": [],
  "suffix_verbs": [],
  "compound_verbs": [],
  "adjectives": [],
  "adverbs": [],
  "adjectivals": [],
  "particles": [["も","も"]],
  "conjunctions": [],
  "pronouns",[["私","わたし"]],
  "suffixes": [],
  "prefixes": [],
  "expressions": [["下さい","ください"]],
  "yojijukugo": []
}}

This example includes handling a compound verb that is split into prefix and suffix verbs.
Example sentence 2: 坂[さか]を<k> 登[のぼ]り切[き]ったら</k>、 向[むこ]うに<b>開豁[かいかつ]</b>に<k> 拓[ひら]けた</k> 海[うみ]が<k> 見下[みお]ろせた</k>のだ。
Example results 2:
{{
  "nouns": [["向こう","むこう"],["坂","さか"],["海","うみ"]],
  "proper_nouns": [],
  "numbers": [],
  "counters": [],
  "verbs": [["開ける","ひらける"]],
  "prefix_verbs": [["登る","のぼる"],["見る","みる"]],
  "suffix_verbs": [["下ろす","おろす"],["切る","きる"]],
  "compound_verbs": [["登り切る","のぼりきる"],["見下ろす","みおろす"]],
  "adjectives": [],
  "adverbs": [],
  "adjectivals": [["開豁","かいかつ"]],
  "particles": [["が","が"],["に","に"],["の","の"],["を","を"]],
  "conjunctions": [],
  "pronouns": [],
  "suffixes": [],
  "prefixes": [],
  "expressions": [],
  "yojijukugo": []
}}

This example includes する verb handling and adverbial adjective handling:
Example sentence 2: <k> 彼[あ]の</k> 飛行機[ひこうき]は<b> 間[ま]も<k> 無[な]く</k></b> 着陸[ちゃくりく]<k> 為[し]ます</k>ね。
Example results 3:
{{
  "nouns": [["飛行機","ひこうき"],["各陸","ちゃくりく"],["間","ま"]],
  "proper_nouns": [],
  "numbers": [],
  "counters": [],
  "verbs": [["為る","する"]],
  "prefix_verbs": [],
  "suffix_verbs": [],
  "compound_verbs": [],
  "adjectives": [["無い","ない"]],
  "adverbs": [["間も無く","まもなく"]],
  "adjectivals": [["彼の","あの"]],
  "particles": [["は","は"],["ね","ね"]],
  "conjunctions": [],
  "pronouns": [],
  "suffixes": [],
  "prefixes": [],
  "expressions": [],
  "yojijukugo": []
}}

This example includes long expression handling with all its individual components added:
Example sentence 3:  <k> 此[こ]れ</k>は 正[まさ]に 天高[てんたか]く 馬肥[うまこ]ゆる 秋[あき]と 言[い]った<k> 物[も]ん</k>だな。
Example result 4:
{{
  "nouns": [["天","てん"],["馬","うま"],["秋","あき"],["物","もの"]],
  "proper_nouns": [],
  "numbers": [],
  "counters": [],
  "verbs": [["肥える","こえる"],["言う","いう"]],
  "prefix_verbs": [],
  "suffix_verbs": [],
  "compound_verbs": [],
  "adjectives": [["高い","たかい"]],
  "adjectivals": [],
  "adverbs": [["正に","まさに"]],
  "particles": [["に","に"],["と","と"],["だ","だ"]],
  "conjunctions": [],
  "pronouns": [["此れ","これ"]],
  "suffixes": [],
  "prefixes": [],
  "expressions": [["天高く馬肥ゆる秋","てんたかくうまこゆるあき"]],
  "yojijukugo": [],
}}

This example includes yojijukugo handling:
Example sentence 4: 昭和[しょうわ]10 年[ねん](1935 年[ねん]) 頃[ごろ]から、<b>八紘一宇[はっこういちう]</b><k> 等[など]</k>のスローガンが 掲[かか]げられる<k> 様[よう]に</k><k> 成[な]った</k>。
Example result 5:
{{
  "nouns": [["昭和","しょうわ"],["年","ねん"],["スローガン","すろーがん"],["様","よう"]],
  "proper_nouns": [],
  "numbers": [],
  "counters": [],
  "verbs": [["掲げる","かかげる"],["成る","なる"]],
  "prefix_verbs": [],
  "suffix_verbs": [],
  "compound_verbs": [],
  "adjectives": [],
  "adjectivals": [],
  "adverbs": [["頃","ごろ"]],
  "particles": [["から","から"],["等","など"],["の","の"],["が","が"],["に","に"]],
  "conjunctions": [],
  "pronouns": [],
  "suffixes": [],
  "prefixes": [],
  "expressions": [],
  "yojijukugo": [["八紘一宇","はっこういちう"]]
}}

This example includes proper noun handling:
Example sentence 5: <b> 不甲斐[ふがい]ない</b> 里樹[りしゅ]<k> 様[さま]</k>の 侍女[じじょ]<k> 達[たち]</k>を 阿多[ああでぅお]<k> 様[さま]</k>の 侍女[じじょ]<k> 達[たち]</k>が<k> 諫[いさ]めていた</k>。
Example result 6:
{{
  "nouns": [["侍女","じじょ"]],
  "proper_nouns": [["里樹","りしゅ"],["阿多","ああでぅお"]],
  "numbers": [],
  "counters": [],
  "verbs": [["諫める","いさめる"]],
  "prefix_verbs": [],
  "suffix_verbs": [],
  "compound_verbs": [],
  "adjectives": [["不甲斐ない","ふがいない"]],
  "adverbs": [],
  "adjectivals": [],
  "particles": [["の","の"],["を","を"],["が","が"]],
  "conjunctions": [],
  "pronouns": [],
  "suffixes": [["様","さま"],["達","たち"]],
  "prefixes": [],
  "expressions": [],
  "yojijukugo": []
}}

This example includes prefix handling:
Example sentence 6: <k> 危[あや]うく</k><b>某[ぼう]</b> 業者[ぎょうしゃ]の 甘言[かんげん]に 騙[だま]され、 大損[おおそん]<k> 為[す]る</k><k> 所[ところ]</k>でした。
Example result 7:
{{
  "nouns": [["業者","ぎょうしゃ"],["甘言","かんげん"],["大損","おおそん"],["所","ところ"]],
  "proper_nouns": [],
  "numbers": [],
  "counters": [],
  "verbs": [["騙す","だます"],["為る","する"]],
  "prefix_verbs": [],
  "suffix_verbs": [],
  "compound_verbs": [],
  "adjectives": [],
  "adverbs": [],
  "adjectivals": [],
  "particles": [["の","の"],["に","に"]],
  "conjunctions": [],
  "pronouns": [],
  "suffixes": [],
  "prefixes": [["某","ぼう"]],
  "expressions": [],
  "yojijukugo": []
}}

This example includes suffix handling:
Example sentence 7: <k> 一[ひと]つ</k>の 仕事[しごと]に<b> 於[お]いて</b> 困難[こんなん] 性[せい]の 尺度[しゃくど]で、 仕事[しごと]の 遂行[すいこう] 能力[のうりょく]が、<k> 其[そ]の</k> 頂上[ちょうじょう]を 越[こ]えない 場合[ばあい]は、 何時[いつ]まで 待[ま]っても 解決[かいけつ]<k> 為[し]ない</k>。
Example results 8:
{{
  "nouns": [["仕事","しごと"],["困難","こんなん"],["性","せい"],["尺度","しゃくど"],["遂行","すいこう"],["能力","のうりょく"],["頂上","ちょうじょう"],["場合","ばあい"],["解決","かいけつ"]],
  "proper_nouns": [],
  "numbers": [],
  "counters": [],
  "verbs": [["越える","こえる"],["待つ","まつ"],["為る","する"]],
  "prefix_verbs": [],
  "suffix_verbs": [],
  "compound_verbs": [],
  "adjectives": [],
  "adverbs": [["何時","いつ"]],
  "adjectivals": [],
  "particles": [["に","に"],["で","で"],["が","が"],["を","を"],["は","は"],["まで","まで"]],
  "conjunctions": [["於いて","おいて"]],
  "pronouns": [],
  "suffixes": [["一つ","ひとつ"],["性","せい"]],
  "prefixes": [],
  "expressions": [],
  "yojijukugo": []
}}

This example includes counter and number handling:
Example sentence 8: 二<b>隻[せき]</b>の 船[ふね]が 同時[どうじ]に 沈[しず]んだ。
Example result 9:
{{
  "nouns": [["船","ふね"],["同時","どうじ"]],
  "proper_nouns": [],
  "numbers": [["二","に"]],
  "counters": [["隻","せき"]],
  "verbs": [["沈む","しずむ"]],
  "prefix_verbs": [],
  "suffix_verbs": [],
  "compound_verbs": [],
  "adjectives": [],
  "adverbs": [],
  "adjectivals": [],
  "particles": [["の","の"],["が","が"],["に","に"]],
  "pronouns": [],
  "suffixes": [],
  "prefixes": [],
  "expressions": [["同時に","どうじに"]],
  "yojijukugo": []
}}

This example includes homonym handling whent the word (行く) is used twice times with different meanings:
Example sentence 9: 最近[さいきん] 行[い]ったデート、どのベースまで 行[い]けた？
Example result 10:
{{
  "nouns": [["デート","でーと"],["ベース","ベーす"]],
  "proper_nouns": [],
  "numbers": [],
  "counters": [],
  "verbs": [["行く","いく",1],["行く","いく",2]],
  "prefix_verbs": [],
  "suffix_verbs": [],
  "compound_verbs": [],
  "adjectives": [],
  "adverbs": [["最近","さいきん"]],
  "adjectivals": [["どの","どの"]],
  "particles": [["まで","まで"]],
  "pronouns": [],
  "suffixes": [],
  "prefixes": [],
  "expressions": [],
  "yojijukugo": []
}}

This example includes honomym handling when the word's (言う) occurrence is the same meaning:
Example sentence 11: そう 言[い]えば、 昨日[きのう]なにいった？
Example results 11:
{{
  "nouns": [["昨日","きのう"]],
  "proper_nouns": [],
  "numbers": [],
  "counters": [],
  "verbs": [["言う","いう"]],
  "prefix_verbs": [],
  "suffix_verbs": [],
  "compound_verbs": [],
  "adjectives": [],
  "adverbs": [],
  "adjectivals": [],
  "particles": [],
  "pronouns": [["何","なに"]],
  "suffixes": [],
  "prefixes": [],
  "expressions": [["そう言えば","そういえば"]],
  "yojijukugo": []
}}

This example includes number handling with month and day counters:
Example sentence 12: <k> 例えば[たとえば]</k>、イギリスや 香港[ほんこん]では3 月[がつ]1 日[にち]に 加齢[かれい]<k> 為[さ]れ</k>、 日本[にっぽん]やニュージーランドでは2 月[がつ]28 日[にち]に 加齢[かれい]<k> 為[さ]れる</k>。 日本[にっぽん]でグレゴリオ 暦[れき]を 採用[さいよう]<k> 為[する]</k> 際[さい]、2 月[がつ]29 日[にち]を<b> 閏[うるう] 日[び]</b>と 定[さだ]めた。
Example results 12:
{{
  "nouns": [["加齢","かれい"],["グレゴリオ暦","ぐれごりおれき"],["暦","れき"],["採用","さいよう"],["際","さい"],["閏日","うるうび"],["閏","うるう"],["日","ひ"]],
  "proper_nouns": [["イギリス","いぎりす"],["香港","ほんこん"],["日本","にっぽん"],["ニュージーランド","にゅーじーらんど"]],
  "numbers": [["3","さん"],["1","いち"],["2","に"]],
  "counters": [["月","がつ"],["日","にち"]],
  "verbs": [["定める","さだめる"],["為れる","される"]],
  "prefix_verbs": [],
  "suffix_verbs": [],
  "compound_verbs": [],
  "adjectives": [],
  "adverbs": [["例えば","たとえば"]],
  "adjectivals": [],
  "particles": [["や","や"],["で","で"],["は","は"],["に","に"],["を","を"],["と","と"]],
  "conjunctions": [],
  "pronouns": [],
  "suffixes": [],
  "prefixes": [],
  "expressions": [],
  "yojijukugo": []
}}

This example includes expression handling with all its individual components added:
Example sentence 13: <b> 鳥肌[とりはだ]</b>が 立[た]つ<k> 位[くらい]</k><k> 痺[しび]れる</k> 演奏[えんそう] 聴[き]かせて<k> 遣[や]っから</k>
Example results 13:
{{
  "nouns": [["鳥肌","とりはだ"], ["演奏","えんそう"], ["位","くらい"]],
  "proper_nouns": [],
  "numbers": [],
  "counters": [],
  "verbs": [["立つ","たつ"], ["痺れる","しびれる"], ["聴く","きく"], ["遣る","やる"]],
  "prefix_verbs": [],
  "suffix_verbs": [],
  "compound_verbs": [],
  "adjectives": [],
  "adverbs": [],
  "adjectivals": [],
  "particles": [["が","が"], ["から","から"]],
  "conjunctions": [],
  "pronouns": [],
  "suffixes": [],
  "prefixes": [],
  "expressions": [["鳥肌が立つ","とりはだがたつ"]],
  "yojijukugo": []
}}

This example includes expression handling when not all its components are added:
Example sentence 14: 私[わたし]<k> 達[たち]</k>は 今[いま] 生徒会[せいとかい]に<b> 頭[あたま]ごなし</b>に 出展[しゅってん] 拒否[きょひ]<k> 為[さ]れている</k> 状況[じょうきょう]で 。
Example results 14:
{{
  "nouns": [["生徒会","せいとかい"], ["頭","あたま"], ["出展","しゅってん"], ["拒否","きょひ"], ["状況","じょうきょう"]],
  "proper_nouns": [],
  "numbers": [],
  "counters": [],
  "verbs": [["為れる","される"]],
  "prefix_verbs": [],
  "suffix_verbs": [],
  "compound_verbs": [],
  "adjectives": [],
  "adverbs": [["今","いま"]],
  "adjectivals": [],
  "particles": [["は","は"], ["に","に"], ["で","で"]],
  "conjunctions": [],
  "pronouns": [["私","わたし"]],
  "suffixes": [["達","たち"]],
  "prefixes": [],
  "expressions": [["頭ごなし","あたまごなし"]],
  "yojijukugo": []
}}

Example sentence 15: 私[わたし]には 生徒会[せいとかい]の<b> 総意[そうい]</b>を 覆[くつがえ]す<k> 様[よう]な</k> 力[ちから]は<k> 有[あ]りません</k>よ！ 前[まえ]も 言[い]いましたが<k> 先[ま]ずは</k> 証拠[しょうこ]！
Example results 15:
{{
  "nouns": [["生徒会","せいとかい"], ["総意","そうい"], ["力","ちから"], ["前","まえ"], ["証拠","しょうこ"], ["様","よう"]],
  "proper_nouns": [],
  "numbers": [],
  "counters": [],
  "verbs": [["覆す","くつがえす"], ["有る","ある"], ["言う","いう"]],
  "prefix_verbs": [],
  "suffix_verbs": [],
  "compound_verbs": [],
  "adjectives": [],
  "adverbs": [["先ず","まず"]],
  "adjectivals": [],
  "particles": [["に","に"], ["は","は"], ["の","の"], ["を","を"], ["も","も"], ["よ","よ"]],
  "conjunctions": [],
  "pronouns": [["私","わたし"]],
  "suffixes": [],
  "prefixes": [],
  "expressions": [],
  "yojijukugo": []
}}

Example sentence 16: 人一倍[ひといちばい]<b> 照[て]れ 屋[や]</b>だった 父[ちち]は、 酒[さけ]<k> 無[な]し</k>には 人[ひと]と 話[はなし]も<k> 出来[でき]</k>なかった。
Example results 16:
{{
  "nouns": [["人","ひと"], ["無し","なし"], ["照れ屋","てれや"], ["父","ちち"], ["話","はなし"], ["酒","さけ"]],
  "proper_nouns": [],
  "numbers": [],
  "counters": [],
  "verbs": [["出来る","できる"]],
  "prefix_verbs": [],
  "suffix_verbs": [],
  "compound_verbs": [],
  "adjectives": [],
  "adverbs": [["人一倍","ひといちばい"]],
  "adjectivals": [],
  "particles": [["だ","だ"], ["は","は"], ["に","に"], ["と","と"], ["も","も"]],
  "conjunctions": [],
  "pronouns": [],
  "suffixes": [],
  "prefixes": [],
  "expressions": [],
  "yojijukugo": []
}}

Example sentence 17: どんな 手段[しゅだん]を<k> 持[も]って</k>も、<k> 俺[おれ]</k>を 殺[ころ]さずには<b> 措[お]かない</b> 気[き]でいるに<k> 違[ちが]い 無[な]い</k>。
Example results 17:
{{
  "nouns": [["手段","しゅだん"], ["気","き"], ["違い","ちがい"]],
  "proper_nouns": [],
  "verbs": [["持つ","もつ"], ["措く","おく"], ["殺す","ころ"], ["殺す","ころす"]],
  "prefix_verbs": [],
  "suffix_verbs": [],
  "compound_verbs": [],
  "adjectives": [["無い","ない"]],
  "adverbs": [],
  "adjectivals": [["どんな","どんな"]],
  "particles": [["で","で"], ["に","に"], ["も","も"], ["を","を"]],
  "pronouns": [["俺","おれ"]],
  "suffixes": [],
  "expressions": [["に違いない","にちがいない"]],
  "yojijukugo": [],
  "numbers": [],
  "counters": [],
  "conjunctions": [],
  "prefixes": []
}}

This example includes multiple expression handling:
Example sentence 18: 見[み]た 目[め]は 私[わたし]より<k> 余程[よっぽど]</k> 悪役[あくやく] 令嬢[れいじょう]っぽい。 声[こえ] 高々[たかだか]に<b> 配役[はいやく]</b>ミスを 主張[しゅちょう]<k> 為[し]たい</k>。
example results 18:
{{
  "nouns": [["ミス","みす"], ["主張","しゅちょう"], ["令嬢","れいじょう"], ["声","こえ"], ["悪役","あくやく"], ["目","め"], ["配役","はいやく"]],
  "proper_nouns": [["私","わたし"]],
  "verbs": [["為る","する"], ["見る","みる"]],
  "prefix_verbs": [],
  "suffix_verbs": [],
  "compound_verbs": [],
  "adjectives": [],
  "adverbs": [["余程","よっぽど"], ["高々","たかだか"]],
  "adjectivals": [],
  "particles": [["に","に"], ["は","は"], ["より","より"], ["を","を"]],
  "pronouns": [["私","わたし"]],
  "suffixes": [["っぽい","っぽい"]],
  "expressions": [["声高々","こえたかだか"], ["悪役令嬢","あくやくれいじょう"], ["見た目","みため"], ["配役ミス","はいやくみす"]],
  "yojijukugo": [],
  "numbers": [],
  "counters": [],
  "conjunctions": [],
  "prefixes": []
}}

This example includes compound verb handling where only 仕舞う in 竦んで仕舞う is considered a suffix verb. However, because it can be used with any verb, 竦む is listed under "verbs" not "prefix_verbs" and the compound itself is omitted:
Example sentence 19: <k> 蛇[ヘビ]</k>を 見[み]て 足[あし]が<b><k> 竦[すく]んで</k><k> 仕舞[しま]った</k></b>。
Example results 19:
{{
  "nouns": [["蛇","へび"], ["足","あし"]],
  "proper_nouns": [],
  "verbs": [["見る","みる"], ["竦む","すくむ"]],
  "prefix_verbs": [],
  "suffix_verbs": [["仕舞う","しまう"]],
  "compound_verbs": [],
  "adjectives": [],
  "adverbs": [],
  "adjectivals": [],
  "particles": [["が","が"], ["を","を"]],
  "pronouns": [],
  "suffixes": [],
  "expressions": [],
  "yojijukugo": [],
  "numbers": [],
  "counters": [],
  "conjunctions": [],
  "prefixes": []
}}

This example includes not splitting a yojijukugo (自由自在) into its components:
Example sentence 20: 自由自在[じゆうじざい]な 人物[じんぶつ]、 大空[おおぞら]を<b>翔[かけ]る</b> 奔馬[ほんば]だ。
Example results 20:
{{
  "nouns": [["人物","じんぶつ"], ["大空","おおぞら"], ["奔馬","ほんば"]],
  "proper_nouns": [],
  "verbs": [["翔る","かける"]],
  "prefix_verbs": [],
  "suffix_verbs": [],
  "compound_verbs": [],
  "adjectives": [],
  "adverbs": [],
  "adjectivals": [],
  "particles": [["な","な"], ["を","を"]],
  "pronouns": [],
  "suffixes": [],
  "expressions": [],
  "yojijukugo": [["自由自在","じゆうじざい"]],
  "numbers": [],
  "counters": [],
  "conjunctions": [],
  "prefixes": []
}}

This example includes suffix handling with correct base reading:
Example sentence 21:  世界中[せかいじゅう]の 言語[げんご]は<k> 幾[いく]つ</k>かの<b> 類型[るいけい]</b>に 分類[ぶんるい]<k> 為[さ]れる</k>。
Example results 21:
{{
  "nouns": [["世界","せかい"], ["分類","ぶんるい"], ["言語","げんご"], ["類型","るいけい"]],
  "proper_nouns": [],
  "verbs": [["為れる","される"]],
  "prefix_verbs": [],
  "suffix_verbs": [],
  "compound_verbs": [],
  "adjectives": [],
  "adverbs": [],
  "adjectivals": [],
  "particles": [["に","に"], ["の","の"], ["は","は"]],
  "pronouns": [],
  "suffixes": [["中","ちゅう"]],
  "expressions": [],
  "yojijukugo": [],
  "numbers": [],
  "counters": [],
  "conjunctions": [],
  "prefixes": []
}}

This example includes homonym handling with the word (方) used twice with different meanings:
Example sentence 22: 藤堂[とうどう]さん 貴方[あなた]そろそろ 軽音部[けいおんぶ]の 方[ほう]に 向[む]かった<b> 方[ほう]が 良[い]い</b>んじゃないの？
Example results 22:
{{
  "nouns":[["方","ほう",1],["方","ほう",2]],
  "proper_nouns":[["藤堂","とうどう"]],
  "numbers":[],
  "counters":[],
  "verbs":[["向かう","むかう"]],
  "prefix_verbs":[],
  "suffix_verbs":[],
  "compound_verbs": [],
  "adjectives":[["良い","いい"]],
  "adverbs":[["そろそろ","そろそろ"]],
  "adjectivals":[],
  "particles":[["の","の"],["に","に"],["が","が"],["で","で"],["は","は"]],
  "conjunctions":[],
  "pronouns":[["貴方","あなた"]],
  "suffixes":[["さん","さん"]],
  "prefixes":[],
  "expressions":[["方が良い","ほうがいい"]],
  "yojijukugo":[]
}}

This example includes including noun form verbs as just verbs:
Example sentence 23: <k> 籤[くじ]</k> 入[い]りカプセル<k> 掬[すく]い</k>だよ！ 一等[いっとう]は<b> 甘[あま]やかし</b><k> 御[お]</k> 姉[ねえ]さんか 罵倒[ばとう]<k> 御[お]</k> 姉[ねえ]さんの 耳元[みみもと]<k> 囁[ささや]き</k>だよ
Example results 23:
{{
  "nouns": [["籤","くじ"], ["カプセル","かぷせる"], ["一等","いっとう"], ["罵倒","ばとう"], ["姉","あね"], ["耳元","みみもと"], ["耳","みみ"], ["元","もと"]],
  "proper_nouns": [],
  "numbers": [],
  "counters": [],
  "verbs": [["入る","はいる"], ["掬う","すくう"], ["甘やかす","あまやかす"], ["囁く","ささやく"]],
  "prefix_verbs": [],
  "suffix_verbs": [],
  "compound_verbs": [],
  "adjectives": [],
  "adverbs": [],
  "adjectivals": [],
  "particles": [["だ","だ"], ["よ","よ"], ["は","は"], ["か","か"], ["の","の"]],
  "conjunctions": [],
  "pronouns": [],
  "suffixes": [["さん","さん"]],
  "prefixes": [["御","お"]],
  "expressions": [],
  "yojijukugo": []
}}


Return only the JSON formatted result containing all properties with at least empty arrays. Values inside the arrays must be arrays of two strings, or two strings and one number for multi-meaning words.

The sentence to process: {sentence}
"""


def get_extracted_words_from_model(
    sentence: str, current_lists: Union[str, None], config: dict[str, str]
) -> Union[dict, None]:
    # current_lists_addition = ""
    # if current_lists:
    # If current_lists is not empty, add the portion to the instructions to examine it
    #         current_lists_addition = f"""The sentence has been processed before and the current word lists are shown below. Your task should be to consider only whether more words should be added to any lists and whether any words may have been categorized differently from the current instructions below - the instructions when you processed these last time may have been different.

    # Instructions on modifying the current lists:
    # - Linked words are word arrays that have 4 elements: 3 strings and one positive or negative long integer may only be moved to another list, not modified and definitely not removed.
    #   - The most common case is that the linked multi-meaning word's 3rd string is the same as the 1st string. For example ["上る","のぼる",上る", 1378555077520]
    #   - When there are more than one multi-meaning words in this form will have the two first strings be identical but the 3rd string and final integer will differ. For example: ["控える","おさえる"] and [控える","おさえる"]
    # - Generally you should only add more words, not remove any. Removal can be considered for compound verbs or expressions that are sufficiently accounted for by their individual components that are already listed in other word categories. Or, if there appears to be too many multi-meaning words, when one less meaning may suffice to account for each usage of the word. However, if the multi-meaning words are already linked, they should not be touched.
    # - If there is a case of a pair or more of homophone+homograph words occurring in the sentence but the current list does not list the word enough times, the to-be-added additional multi-meaning words' meaning index number depends on whether the current words are linked or not.
    #   a. If there is only a single 2-element non-linked word, you should modify it to add the meaning index number to it, starting from 1, and add new word(s) with meanings number incrementing from there. For example, there being ["上がる","あがる"] only but 2 meanings of 上がる used in the sentence --> the result would contain ["上がる"] and ["上がる"]
    #   b. If there is more than one 2-element word - which should contain meaning numbers already - continue adding more words with meaning numbers beginning from the highest index + 1 of the current words. For example, ["上がる"] and ["上がる"] being present but 3 meanings of 上がる are used in the sentence --> add one more, so ["上がる"]
    #   c. If there are any 4-element linked words, the meaning numbers you use for the new word or words you add do not need count the linked word(s). For example, there is ["当て","あて"] and 2 meanings of 当て used --> only add ["当て"]. If there is ["当て","あて"] and ["当て"] but 3 meanings of 当て used --> only add ["当て"]

    # Current word lists: {current_lists}

    # """
    prompt = get_extract_words_prompt(sentence)
    model = config.get("extract_words_model", "")
    result = get_response(model, prompt, max_output_tokens=6000)
    if result is None:
        logger.error("Failed to get a response from the API.")
        # If the prompt failed, return nothing
        return None
    return result


def extract_words_in_note(
    config: dict,
    note: Note,
    notes_to_add_dict: dict[str, list[Note]],
    notes_to_update_dict: dict[NoteId, Note],
) -> bool:
    note_type = note.note_type()
    if not note_type:
        logger.error(f"Missing note type for note {note.id}")
        return False
    try:
        word_extraction_sentence_field = get_field_config(
            config, "word_extraction_sentence_field", note_type
        )
        word_list_field = get_field_config(config, "word_list_field", note_type)
    except Exception as e:
        logger.error(str(e))
        return False

    ignore_current_word_lists = config.get("ignore_current_word_lists", False)

    logger.debug(
        f"word_extraction_sentence_field in note: {word_extraction_sentence_field in note}"
    )
    logger.debug(f"word_list_field in note: {word_list_field in note}")
    # Check if the note has the required fields
    if word_extraction_sentence_field in note and word_list_field in note:
        logger.debug("note has fields")
        # Get the values from fields
        sentence = note[word_extraction_sentence_field]
        logger.debug(f"sentence: {sentence}")
        # Check if the value is non-empty
        if sentence:
            # Remove text within <i> tags, as it is not relevant for word extraction
            sentence = re.sub(r"<i>.*?</i>", "", sentence, flags=re.DOTALL)
            current_word_lists_raw = note[word_list_field]
            logger.debug(
                f"current_word_lists_raw: '{current_word_lists_raw}', ignore_current_word_lists:"
                f" {ignore_current_word_lists}"
            )
            current_word_lists = None
            # Reformat the current word lists to the same format so formatting differences do not
            # cause issues
            if current_word_lists_raw and not ignore_current_word_lists:
                try:
                    current_word_lists = word_lists_str_format(json.loads(current_word_lists_raw))
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON from current word lists: {e}")
                    return False
                logger.debug("Calling API with sentence")
            else:
                logger.debug("Calling API with sentence and no current word lists")

            word_lists = get_extracted_words_from_model(sentence, current_word_lists, config)
            logger.debug(f"result from API: {word_lists}")

            if word_lists is not None:
                logger.debug(f"word_list_json: {word_lists}")

                # Update the note with the new values
                if not isinstance(word_lists, dict):
                    logger.warning("API response is not a dictionary, resetting to empty")
                    word_lists = {}
                logger.debug(f"new_list: {word_lists}")
                # compare and add the new words to the current word list
                try:
                    cur_word_lists = json.loads(note[word_list_field])
                except json.JSONDecodeError:
                    cur_word_lists = {}
                if not isinstance(cur_word_lists, dict):
                    cur_word_lists = {}
                logger.debug(f"cur_word_lists: {cur_word_lists}")
                # Compare each matching key in the new and current word lists
                for key in word_lists:
                    if key in cur_word_lists:
                        logger.debug(f"Comparing word lists for key {key}")
                        # Compare the current and new word lists
                        cur_word_list = cur_word_lists[key]
                        new_word_list = word_lists[key]
                        combined_list = compared_word_lists(cur_word_list, new_word_list)
                        logger.debug(f"Combined list for key {key}: {combined_list}")
                        # Update the current word list with the combined list
                        cur_word_lists[key] = combined_list
                    else:
                        logger.debug(f"Adding new key {key} to word lists")
                        # If the key is not present in the current word list, add it
                        # Use the compare func to validate the new list elements
                        cur_word_lists[key] = compared_word_lists([], word_lists[key])
                # Finally, update the field value in the note
                note[word_list_field] = format_word_list_dict(cur_word_lists)
                if note.id != 0 and note.id not in notes_to_update_dict:
                    notes_to_update_dict[note.id] = note
                return True
            return False
        return False
    else:
        logger.error("note is missing fields")
    return False


def extract_words_test_compare_in_note(
    config: dict,
    note: Note,
    notes_to_add_dict: dict[str, list[Note]],
    notes_to_update_dict: dict[NoteId, Note],
) -> bool:
    """Run extract-words prompt and compare with current list, tagging match/no-match only."""
    note_type = note.note_type()
    if not note_type:
        logger.error(f"Missing note type for note {note.id}")
        return False

    try:
        word_extraction_sentence_field = get_field_config(
            config, "word_extraction_sentence_field", note_type
        )
        word_list_field = get_field_config(config, "word_list_field", note_type)
    except Exception as e:
        logger.error(str(e))
        return False

    ignore_current_word_lists = config.get("ignore_current_word_lists", False)

    if word_extraction_sentence_field not in note or word_list_field not in note:
        logger.error("note is missing fields")
        return False

    sentence = note[word_extraction_sentence_field]
    if not sentence:
        return False
    sentence = re.sub(r"<i>.*?</i>", "", sentence, flags=re.DOTALL)

    current_word_lists_raw = note[word_list_field]
    current_word_lists_for_prompt = None
    if current_word_lists_raw and not ignore_current_word_lists:
        try:
            current_word_lists_for_prompt = word_lists_str_format(
                json.loads(current_word_lists_raw)
            )
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from current word lists: {e}")
            return False

    test_word_lists = get_extracted_words_from_model(
        sentence,
        current_word_lists_for_prompt,
        config,
    )
    if test_word_lists is None:
        return False
    if not isinstance(test_word_lists, dict):
        logger.warning("Extract words test response is not a dictionary")
        test_word_lists = {}

    try:
        current_word_lists = json.loads(current_word_lists_raw) if current_word_lists_raw else {}
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding current word lists for note {note.id}: {e}")
        return False
    if not isinstance(current_word_lists, dict):
        current_word_lists = {}

    normalized_original = normalize_word_list_dict_for_test_comparison(current_word_lists)
    normalized_test = normalize_word_list_dict_for_test_comparison(test_word_lists)
    is_match, diffs = compare_normalized_word_lists_for_test(normalized_original, normalized_test)

    note.remove_tag(EXTRACT_WORDS_TEST_MATCH_TAG)
    note.remove_tag(EXTRACT_WORDS_TEST_NO_MATCH_TAG)
    note.add_tag(EXTRACT_WORDS_TEST_MATCH_TAG if is_match else EXTRACT_WORDS_TEST_NO_MATCH_TAG)

    if not is_match:
        logger.debug(
            "Extract words test mismatch for note %s\nDiff summary:\n%s\nOriginal normalized:\n%s"
            "\nTest normalized:\n%s",
            note.id,
            _format_test_diffs(diffs),
            format_word_list_dict(normalized_original),
            format_word_list_dict(normalized_test),
        )

    if note.id != 0 and note.id not in notes_to_update_dict:
        notes_to_update_dict[note.id] = note

    return True


def bulk_extract_from_notes_op(
    col: Collection,
    notes: Sequence[Note],
    edited_nids: list[NoteId],
    progress_updater: AsyncTaskProgressUpdater,
    notes_to_add_dict: dict[str, list[Note]],
    notes_to_update_dict: dict[NoteId, Note],
):
    config = mw.addonManager.getConfig(__name__)
    if not config:
        showWarning("Missing addon configuration")
        return
    message = "Extracting words"
    op = extract_words_in_note
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


def extract_words_from_selected_notes(nids: Sequence[NoteId], parent: Browser):
    progress_updater = AsyncTaskProgressUpdater(title="Async AI op: Extracting words")
    done_text = "Updated word lists"
    bulk_op = bulk_extract_from_notes_op
    return selected_notes_op(done_text, bulk_op, nids, parent, progress_updater)


def bulk_extract_words_test_compare_from_notes_op(
    col: Collection,
    notes: Sequence[Note],
    edited_nids: list[NoteId],
    progress_updater: AsyncTaskProgressUpdater,
    notes_to_add_dict: dict[str, list[Note]],
    notes_to_update_dict: dict[NoteId, Note],
):
    config = mw.addonManager.getConfig(__name__)
    if not config:
        showWarning("Missing addon configuration")
        return
    model = config.get("extract_words_model", "")
    logger.debug(f"Model for extract words test compare: {model}")
    message = "Testing extract words prompt"
    op = extract_words_test_compare_in_note
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


def extract_words_test_compare_from_selected_notes(nids: Sequence[NoteId], parent: Browser):
    progress_updater = AsyncTaskProgressUpdater(title="Async AI op: Testing extracted words prompt")
    done_text = "Tagged extract words test results"
    bulk_op = bulk_extract_words_test_compare_from_notes_op
    return selected_notes_op(done_text, bulk_op, nids, parent, progress_updater)
