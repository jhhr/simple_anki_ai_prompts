import re
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


K_WORD_REC = re.compile(r"<k>([^<]*)</k>")
K_INNER_REC = re.compile(r"( [\d々\u4e00-\u9faf\u3400-\u4dbf]+\[[^[]*\][^>[]*?)")

SO_WITHOUT_NO = re.compile(r"<k> 其\[そ\]</k>の?")
SO_INSIDE_NO = re.compile(r" <k> 其\[その\]</k>")

K_INNER_REVERSING_REC = re.compile(r" [\d々\u4e00-\u9faf\u3400-\u4dbf]+\[([^[]*)\]([^>[]*?)")


def make_k_word_replacer(sentence: str):
    # Check if the match can be found in the original sentence
    k_words_in_sentence = False

    def inner_k_word_replacer(match: re.Match[str]) -> str:
        nonlocal k_words_in_sentence
        word = match.group(1)
        if word in sentence:
            # remove <k> tags if the word was kanjified in the original sentence
            k_words_in_sentence = True
            return word
        else:
            # Keep the <k> word unchanged
            return match.group(0)

    def k_word_replacer(match: re.Match[str]) -> str:
        # the match is of the form <k>...</k> which may contain multiple furigana words
        res = K_INNER_REC.sub(inner_k_word_replacer, match.group(1))
        if k_words_in_sentence:
            return res
        else:
            return match.group(0)

    return k_word_replacer


def inner_k_word_reversing_replacer(match: re.Match[str]) -> str:
    furigana = match.group(1)
    okurigana = match.group(2) or ""
    return f"{furigana}{okurigana}"


def k_word_reversing_replacer(match: re.Match[str]) -> str:
    # the match is of the form <k>...</k> which may contain multiple furigana words
    return K_INNER_REVERSING_REC.sub(inner_k_word_reversing_replacer, match.group(1))


B_TAGS_REC = re.compile(r"<b>|</b>")

NUMBER_FURI_REC = re.compile(r"(\d+)\[([^\]]+)\]([あ-ん]*)")


def make_number_furi_replacer(sentence: str):
    def number_furi_replacer(match: re.Match[str]) -> str:
        # Check if the original sentence has the number with furigana or not
        number = match.group(1)
        okurigana = match.group(3) or ""
        whole_match = match.group(0)
        if whole_match in sentence:
            # It has furigana too, keep the number with furigana
            return whole_match
        else:
            # it doesn't, so return just the number and okurigana
            return number + okurigana

    return number_furi_replacer


KANJIFIED_SENTENCE_RETURN_FIELD = "kanjified_sentence"
KANJIFY_SENTENCE_DEFAULT_TEMPERATURE = 0.1


def get_kanjify_sentence_prompt(sentence: str) -> str:
    return f"""Below is a sentence in Japanese that includes furigana in brackets after kanji words. Your task is to convert the sentence into a fully kanjified version, where all hiragana and katakana words are replaced with their kanji equivalents"

Carefully examine each and every word written in hiragana or katakana and determine whether it can be written in kanji.
Common words that always written in hiragana should be kanjified. Examples of common words in kanjified form (not to be considered an exhaustive list of what to kanjify!): 此[こ]れ, 無[な]い,出来[でき]る, 御前[おまえ], 愈々[いよいよ]
The result should be text where only foreign words or names in katakana, particles and words for which no kanjified form exists or should be used (see edge cases below), are left in hiragana or katakana.

Content modification rules:
- To signify the changes made to the text, wrap each kanjified word in <k> tags with a space before the kanji: "これ" becomes "<k> 此[こ]れ</k>". Include the okurigana of verbs and adjectives within the <k> tags, for example "まわってた" becomes "<k> 回[まわ]ってた</k>".
- <k> tags should wrap a contiguous sequence of kanji conversions, stopping on a word that was already in kanji. For example "とうもろこし" becomes "<k> 玉蜀黍[とうもろこし]</k>" and "タンパク 質[しつ]" becomes "<k> 蛋白[たんぱく]</k> 質[しつ]". Make sure add spaces when there's multiple furigana words within one <k> tag, for example "つきあい" becomes "<k> 付[つ]き 合[あ]い</k>".
- Keep any existing HTML tags in the sentence as they are. Added <k> tags should be placed inside existing tags, leaving them outermost so that multiple <k> tags can, if necessary, be within the existing tag. For example "<b>これ見よがしに</b>" becomes "<b><k> 此[こ]れ</k> 見[み]よがしに</b>".
- Preserve the katakana of kanjifiable words by including in the furigana. For example, "バカ" becomes "<k> 馬鹿[バカ]</k>".
- Maintain the original colloquial shortenings when kanjifying. For example when もの (者 or 物) is written as もん: バカもん becomes "<k> 馬鹿者[バカもん]</k>", そんなもんなんか becomes "<k> 其[そ]んな 物[もん]</k>なんか". Note, that these kind of shortenings should not be considered <gikun> cases.

Policy on edge cases (not be considered an exhaustive list, but can be used as a guideline for cases not listed here):

Do not kanjify:
- あげる when meant as "to give". あげる is written in hiragana specifically to indicate that it is different from 揚げる, 上げる or 挙げる
- ない auxiliary in negated verbs. ない is not considered 無い written in hiragana, but rather a grammatical auxiliary.
- いる and いく as an auxiliary verb in verb conjugation, for example 書いている, 食べていく, etc. These auxiliaries should be considered a part of a verb's conjugation and, thus, its okurigana, and so enclosed within the <k> tag of a kanjified verb.
- なんか when it is acting more as a particle or filler and could removed without (significant) loss of meaning, for example 今日なんか暑いですね
- もう used purely as exclamatory particle, for example もう！ or もう、やめてよ！
- もっと as it is not truly component in もっとも which does have a kanjified form as 最も or 尤も
- そんな, こんな, あんな, どんな
- The expression として should be considered to not contain する and should be left as is.
― Gikun-type, reading-as-meaning conversions. Only perform kanjification, if there is some evidence of usage in dictionaries.

Do kanjify:
- ない when used as a standalone word, including conjugated forms like なかった, なくて. 無い is even currently used in modern text, but is simply often written in hiragana.
- いる and いく when is used as a standalone verb, for example 彼は家にいる, あっちにいく
- する, even in suru-verbs. Historically, suru-verbs were written with 為る so this is a valid kanjification.
- なんか when it is clearly a contraction of なにか its removal would change the questioning meaning of a phrase, for example なんか食べたい
- やすい as used in verbs like 食べやすい, 書きやすい, etc. This is a a form of 易い
- the honorific prefix お
- よう as 様[よう] in all its forms, ような, ように, ようだ, etc.
- The pluralizing suffix ら as 等. Like in 彼ら, そいつら etc.
- Romaji numbers, keep them as is though furigana can be added without adding <k> tags. For example, １つ (no furigana) becomes "１[ひと]つ". 10分[ぷん] becomes "10分[じゅっぷん]". 1000[せん]円[えん] stays as is.


Important final checks:
- MAKE SURE TO NOT TO OMIT ANY PARTICLES OR COPULA FROM THE KANJIFIED SENTENCE

# Examples to illustrate the conversions:
Example sentence 1: 「これでもちゃんと 皆[みな]さんのことを 考[かんが]えてるつもりなんですよ！」<br>「なんかいよいよお 前[まえ]も 完全[かんぜん]に 内政[ないせい] 官[かん]だな。」
Kanjified example 1: 「 <k> 此[こ]れ</k>でもちゃんと 皆[みな]さんの<k> 事[こと]</k>を 考[かんが]えてる<k> 積[つも]り</k>なんですよ！」<br>「なんか<k> 愈々[いよいよ]</k><k> 御[お]</k> 前[まえ]も 完全[かんぜん]に 内政[ないせい] 官[かん]だな。」

Example sentence 2: ナツキ 殿[どの]ですよね？ 兄[あに]から 聞[き]いています。その 数々[かずかず]のうわさも<b>かねがね</b>。
Kanjified example 2: ナツキ 殿[どの]ですよね？ 兄[あに]から 聞[き]いています。<k> 其[そ]</k>の 数々[かずかず]の<k> 噂[うわさ]</k>も<b><k> 兼々[かねがね]</k></b>。

Example sentence 3: ズボンのすそを<b>まくって</b> 作業[さぎょう]をした。
Kanjified example 3: <k> 洋袴[ズボン]</k>の<k> 裾[すそ]</k>を<b><k> 捲[まく]って</k></b> 作業[さぎょう]を<k> 為[し]た</k>。

Example sentence 4: おかげで　この 守銭奴[しゅせんど]の 性[しょう] 悪天使[あくてんし]に ぼったくられたぜ。
Kanjified example 4: <k> 御蔭[おかげ]</k>で<k> 此[この]</k> 守銭奴[しゅせんど]の 性[しょう] 悪天使[あくてんし]にぼっ<k> 手繰[たく]られた</k>ぜ。

Example sentence 5: 毎日[まいにち]一キロ 以上[いじょう] 水泳[すいえい]をしてきただけのことはあって、 彼[かれ]は九十 歳[さい]の 今[いま]もかくしゃくとしている。
Kanjified example 5: 毎日[まいにち] 一[いち]キロ 以上[いじょう] 水泳[すいえい]を<k> 為[し]て</k><k> 来[き]た</k>だけの<k> 事[こと]</k>は<k> 有[あ]って</k>、 彼[かれ]は 九十[きゅうじゅう] 歳[さい]の 今[いま]も<k> 矍鑠[かくしゃく]</k>として<k> 居[い]る</k>。

Example sentence 6: 俺はちっぽけでどうしようもないろくでなしですよ。
Kanjified example 6: 俺[おれ]は<k> 小[ち]</k>っぽけで<k> 如何[どう]</k><k> 仕様[しよう]</k>も<k> 無[な]い</k><k> 碌[ろく]</k>で<k> 無[な]し</k> ですよ。

Example sentence 7: <b>しのごの</b> 言[い]わずさっさと 手[て]を 貸[か]せ。
Kanjified example 7: <b><k> 四[し]</k>の<k> 五[ご]</k>の</b> 言[い]わずさっさと 手[て]を 貸[か]せ。

Example sentence 8: <ul><li>いろいろなもので 遊[あそ]んでいるうちに 家[いえ]の 中[なか]は<span style="font-weight: bold;">めちゃくちゃ</span>になっていた。ふと 気[き]が 付[つ]けば 時計[とけい]は11 時[じ]を 指[さ]していた。</li></ul><br>
Kanjified example 8: <ul><li><k> 色々[いろいろ]</k>な<k> 物[もの]</k>で 遊[あそ]んでいる<k> 内[うち]に</k> 家[いえ]の 中[なか]は<span style="font-weight: bold;"><k> 滅茶苦茶[めちゃくちゃ]</k></span>に<k> 成[な]っていた</k>。<k> 不図[ふと]</k> 気[き]が 付[つ]けば 時計[とけい]は11 時[じ]を 指[さ]していた。</li></ul><br>

Example sentence 9: しかもけっして 食欲[しょくよく]がないからではなかったのだ。また、 彼[かれ]の 口[くち]にもっと 合[あ]うような 別[べつ]な 食[た]べものをもってくるのだろうか。 妹[いもうと]が 自分[じぶん]でそうしてくれないだろうか。
Kanjified example 9: <k> 然[しか]も</k><k> 決[け]っして</k> 食欲[しょくよく]が<k> 無[な]い</k>からでは<k> 無[な]かった</k>のだ。<k> 又[また]</k>、 彼[かれ]の 口[くち]にもっと<k> 合[あ]う</k><k> 様[よう]な</k><k> 別[べつ]</k>な<k> 食[た]べ物[もの]</k>を<k> 持[も]って</k><k> 来[く]る</k>のだろうか。 妹[いもうと]が 自分[じぶん]で<k> 然[そ]う</k><k> 為[し]て</k><k> 呉[く]れない</k>だろうか。

Example sentence 10: 私[わたし]は 毎朝[まいあさ]<b>コーヒー</b>を 飲[の]みます。
Kanjified example 10: 私[わたし]は 毎朝[まいあさ]<b><k> 珈琲[コーヒー]</k></b>を 飲[の]みます。

Example sentence 11: <b> 上司[じょうし]</b>に 相談[そうだん]してみます。
Kanjified example 11: <b> 上司[じょうし]</b>に 相談[そうだん]<k> 為[し]て</k><k> 見[み]ます</k>。

Example sentence 12: <b> 未来[みらい]</b>は 誰[だれ]にも 分[わ]からない。
Kanjified example 12: <b> 未来[みらい]</b>は 誰[だれ]にも 分[わ]からない。

Example sentence 13: お 坊[ぼう]さんが 鐘[かね]を<b> 鳴[な]らして</b>いますね。
Example kanjified 13: <k> 御[お]</k> 坊[ぼう]さんが 鐘[かね]を<b> 鳴[な]らして</b>いますね。

Example sentence 14: 彼[かれ]らは<b> 裸[はだか]</b>のつきあいをしているよ。
Kanjified example 14: 彼[かれ]らは<b> 裸[はだか]</b>の<k> 付[つ]き 合[あ]い</k>を<k> 為[し]ている</k>よ。

Example sentence 15: <b> 作業[さぎょう]</b>するにはもっと 広[ひろ]いスペースが 必要[ひつよう]だ。
Example kanjified 15: <b> 作業[さぎょう]</b><k> 為[す]る</k>にはもっと 広[ひろ]いスペースが 必要[ひつよう]だ。

Example sentence 16: 娘[むすめ]が 初[はじ]めて<b> 寝返[ねがえ]り</b>しました。
Example kanjified 16: 娘[むすめ]が 初[はじ]めて<b> 寝返[ねがえ]り</b><k> 為[し]</k>ました。

Example sentence 17: <b>とにかく</b> 現場[げんば]へ 行[い]ってみましょう。
Example kanjified 17: <b><k> 兎[と]に 角[かく]</k></b> 現場[げんば]へ 行[い]って<k> 見[み]ましょう</k>。

Example sentence 18: <i>なっ…バカもん！ 麦[むぎ]のあとは 放牧[ほうぼく]だ。</i> 切[き]り 株[かぶ]を 長[なが]めに 残[のこ]しておくのはあとで 家畜[かちく]に 食[く]わせるためなんだぞ。
Example kanjified 18: <i>なっ…<k> 馬鹿者[バカもん]</k>！ 麦[むぎ]の<k> 後[あと]</k>は 放牧[ほうぼく]だ。</i> 切[き]り 株[かぶ]を 長[なが]めに 残[のこ]しておくのは<k> 後[あと]</k>で 家畜[かちく]に<b> 食[く]わせる</b><k> 為[ため]</k>なんだぞ。

Example sentence 19: あの 逃[に]げ 方[かた]は 一番[いちばん]マズい。 背中[せなか]を<b> 蹴[け]って</b>くれと 言[い]ってるようなものだ
Example kanjified 19: <k> 彼[あの]</k> 逃[に]げ 方[かた]は 一番[いちばん]<k> 不味[マズ]い</k>。 背中[せなか]を<b> 蹴[け]って</b><k> 呉[く]れ</k>と 言[い]ってる<k> 様[よう]な</k><k> 物[もの]</k>だ

Example sentence 20: 北[きた]アメリカでは １つの 家[いえ]に １つ<b>ないし</b> ２つの 車庫[しゃこ]があるのはよくあることだ。
Example kanjified 20: 北アメリカでは １[ひと]つの 家[いえ]に １[ひと]つ<b><k> 乃至[ないし]</k></b> ２[ふた]つの 車庫[しゃこ]が<k> 有[あ]る</k>のは<k> 良[よ]く</k><k> 有[あ]る</k><k> 事[こと]</k>だ。

Example sentence 21: それに 触[ふ]れた 銀[ぎん]の<b>カバ</b>ノキも 見[み]えます。
Example kanjified 21: <k> 其[そ]れ</k>に 触[ふ]れた 銀[ぎん]の<b><k> 樺[カバ]</k></b>ノ<k> 木[キ]</k>も 見[み]えます。

Example sentence 22: それらの 作品[さくひん]には<b>優劣[ゆうれつ]</b>をつけがたい。
Kanjified example 22: <k> 其[そ]れ 等[ら]</k>の 作品[さくひん]には<b><k> 優劣[ゆうれつ]</k></b>を<k> 付[つ]け 難[がた]い</k>。

Return a JSON string with the following key-value pairs:
 "{KANJIFIED_SENTENCE_RETURN_FIELD}": The fully kanjified sentence.

The sentence to process: {sentence}
"""


def get_kanjified_sentence_from_model(
    config: dict[str, str],
    sentence: str,
) -> Union[list[str], None]:
    prompt = get_kanjify_sentence_prompt(sentence)
    model = config.get("kanjify_sentence_model", "")
    config_temp = config.get("kanjify_sentence_temperature", None)
    if config_temp is not None:
        try:
            temperature = float(config_temp)
        except ValueError:
            logger.error(
                "Invalid temperature value in config: %s. Using default temperature: %f",
                config_temp,
                KANJIFY_SENTENCE_DEFAULT_TEMPERATURE,
            )
            temperature = KANJIFY_SENTENCE_DEFAULT_TEMPERATURE
    else:
        temperature = KANJIFY_SENTENCE_DEFAULT_TEMPERATURE
    result = get_response(model, prompt, temperature=temperature)
    if result is None:
        logger.error("Failed to get a response from the API.")
        # If the prompt failed, return nothing
        return None
    try:
        return [result[KANJIFIED_SENTENCE_RETURN_FIELD]]
    except KeyError:
        logger.error(f"Key '{KANJIFIED_SENTENCE_RETURN_FIELD}' not found in the result.")
        return None


MAX_ATTEMPTS = 5


def kanjify_sentence_in_note(
    config: dict[str, str],
    note: Note,
    notes_to_add_dict: dict[str, list[Note]],
    notes_to_update_dict: dict[NoteId, Note],
    attempt: int = 1,
) -> bool:
    model = note.note_type()
    if not model:
        logger.error("Missing note type for note %s", note.id)
        return False
    try:
        furigana_sentence_field = get_field_config(config, "furigana_sentence_field", model)
        kanjified_sentence_field = get_field_config(config, "kanjified_sentence_field", model)
    except Exception as e:
        logger.error("Error getting field config: %s", e)
        return False

    logger.debug("furigana_sentence_field in note: %s", furigana_sentence_field in note)
    logger.debug("kanjified_sentence_field in note: %s", kanjified_sentence_field in note)
    # Check if the note has the required fields
    if furigana_sentence_field in note and kanjified_sentence_field in note:
        # Get the values from fields
        sentence = note[furigana_sentence_field]
        logger.debug("sentence: %s", sentence)
        # Check if the value is non-empty
        if sentence:
            # Clean any <b> tags from the sentence
            # sentence = sentence.replace("<b>", "").replace("</b>", "")
            logger.debug("cleaned sentence: %s", sentence)
            # Call API to get translation
            result = get_kanjified_sentence_from_model(config, sentence)
            logger.debug("result from API: %s", result)
            if result is not None:
                [kanjified_sentence] = result
                logger.debug("kanjified_sentence: %s", kanjified_sentence)
                # Clean up common mistakes by the AI
                # Sometimes kanjifying する results in 為[し]る
                kanjified_sentence = kanjified_sentence.replace(
                    "<k> 為[し]る</k>", "<k> 為[す]る</k>"
                )
                # Sometimes when kanjifying その it leaves out the の --> <k> 其[そ]</k>
                kanjified_sentence = SO_WITHOUT_NO.sub("<k> 其[そ]の</k>", kanjified_sentence)
                # Or puts the の inside the furigana tags -->  <k> 其[その]</k>
                kanjified_sentence = SO_INSIDE_NO.sub(" <k> 其[そ]の</k>", kanjified_sentence)
                # Sometimes it wraps the sentence in 「」when the original sentence wasn't
                if not (sentence.startswith("「") and sentence.endswith("」")) and (
                    kanjified_sentence.startswith("「") and kanjified_sentence.endswith("」")
                ):
                    # Remove the wrapping
                    kanjified_sentence = kanjified_sentence[1:-1]

                # It may unnecessarily wrap words in <k> tags that were already kanjified in the
                # original sentence

                k_word_replacer = make_k_word_replacer(sentence)
                kanjified_sentence = K_WORD_REC.sub(k_word_replacer, kanjified_sentence)

                # Clean double spaces
                kanjified_sentence = kanjified_sentence.replace("  ", " ")
                # Clean extra space before <k> tags
                kanjified_sentence = kanjified_sentence.replace(" <k> ", "<k> ")

                # Update the note with the new values
                note[kanjified_sentence_field] = kanjified_sentence

                # Check if reversing the kanjification results in the original sentence and tag
                # the note if not
                reversed_sentence = B_TAGS_REC.sub("", kanjified_sentence)
                reversed_sentence = K_WORD_REC.sub(k_word_reversing_replacer, reversed_sentence)
                number_furi_replacer = make_number_furi_replacer(sentence)
                reversed_sentence = NUMBER_FURI_REC.sub(number_furi_replacer, reversed_sentence)
                # Remove all whitespace as the comparison often fails due to trivial differences
                reversed_sentence = re.sub(r"\s", "", reversed_sentence)

                cleaned_sentence = B_TAGS_REC.sub("", sentence)
                cleaned_sentence = re.sub(r"\s", "", cleaned_sentence)
                if cleaned_sentence != reversed_sentence:
                    note.add_tag("kanjify_sentence_mismatch")
                    # try again until MAX_ATTEMPTS is reached
                    print(
                        f"Reversed sentence does not match original:\n{reversed_sentence}\n"
                        f"{cleaned_sentence}"
                    )
                    if attempt < MAX_ATTEMPTS:
                        logger.debug(
                            "Reversed sentence does not match original. Attempt %d of %d",
                            attempt,
                            MAX_ATTEMPTS,
                        )
                        return kanjify_sentence_in_note(
                            config, note, notes_to_add_dict, notes_to_update_dict, attempt + 1
                        )
                elif note.has_tag("kanjify_sentence_mismatch"):
                    note.remove_tag("kanjify_sentence_mismatch")
                if note.id != 0 and note.id not in notes_to_update_dict:
                    notes_to_update_dict[note.id] = note
                return True
            return False
        return False
    else:
        logger.error("note is missing fields")
    return False


def bulk_kanjify_notes_op(
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
    message = "Kanjifying sentences"
    op = kanjify_sentence_in_note
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


def kanjify_selected_notes(nids: Sequence[NoteId], parent: Browser):
    progress_updater = AsyncTaskProgressUpdater(title="Async AI op: Kanjifying sentences")
    done_text = "Updated kanjified sentences"
    bulk_op = bulk_kanjify_notes_op
    return selected_notes_op(done_text, bulk_op, nids, parent, progress_updater)
