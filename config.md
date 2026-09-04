# Config

## General

- `log_level`: Default is "ERROR". Possible values from less logging to more: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
- `log_to_console` Default is `true`. If false, logs to files in the logs/ dir in the addon folder

## models

Needs to be one of the models that supports structured output

### OpenAI

- `gpt-4o` and `gpt-4o-*` models
- `gpt-4` and `gpt-4-*` models
- `gpt-3.5-turbo`

### Google

- `gemini-2.5-pro`
- `gemini-2.5-flash`
- `gemini-2.0-flash` (default, free)

## sentence_field / meaning_field / word_field

### model

Define which model to use for each task

- `word_meaning_model`
- `kanji_story_model`
- `translate_sentence_model`
- `kanjify_sentence_model`
- `extract_words_model`
- `match_words_model`

### temperature

- `kanjify_sentence_temperature`: Default is `0.1`. Passed through to provider temperature controls for kanjification requests only.
  Use a low value to reduce variation in kanji choice while still allowing the model a small amount of flexibility. `0.0` is supported by the major providers here, but is not guaranteed to be fully deterministic and can sometimes be more brittle than a very low non-zero setting.

### rate limits

Nothing to configure. Requests go out as fast as the concurrency limit allows; when a provider
rejects one for exceeding its rate limit, that model is put on a short cooldown and the request
is retried automatically. The wait comes from the provider's own response — Anthropic's
`retry-after` header, OpenAI's `Retry-After`, Gemini's `RetryInfo.retryDelay` — falling back to
exponential backoff when none is given.

Errors that retrying can't fix are not retried: an exhausted OpenAI billing quota
(`insufficient_quota`) or a used-up Gemini per-day quota fails immediately and is logged.

- `max_request_retries`: Default `5`. How many times to retry a request that failed for a
  retryable reason (rate limit, provider overload, timeout, connection error).
- `max_retry_wait_seconds`: Default `120`. If a provider asks to wait longer than this, the
  request is abandoned rather than stalling the whole run. Raise it if you regularly hit long
  token-per-minute cooldowns and would rather wait them out.

### memory use and concurrency

How many notes/words are processed at once is what determines memory use, and it is sized
automatically: available RAM divided by what one task of that op actually costs. A tablet gets a
lower limit than a desktop, and a heavy op gets a lower limit than a light one, with no
configuration. While a run is going, the limit is lowered if free memory gets low and raised
again when it recovers. The progress dialog shows the current value.

The per-task cost is measured, not guessed. Notes are processed in windows, and between windows
nothing is in flight, which gives a clean baseline — memory that has accumulated over the run so
far is absorbed into it, so only growth caused by the concurrent tasks counts. The largest
measurement in a run wins, since what has to fit in RAM is the peak. The result is remembered per
op in `user_files/memory_estimates.json` and blended with the previous value, so the first run of
an op learns what it costs and later runs start out sized correctly. Deleting that file just
means the ops get measured again from the default guess.

While nothing is configured there is also a backstop of 256 concurrent tasks, which only exists to
stop a very cheap op on a very empty machine from opening an absurd number of connections at once.
Memory is meant to be what limits concurrency in practice: with a couple of GB of budget, ops
costing more than about 8 MB per task are limited by memory rather than by the backstop.

- `max_concurrent_requests`: Default `0` (automatic). Any value above `0` takes the place of that
  backstop, whether it is below 256 or above it, but does **not** turn off the memory-based
  adjustment — the limit still drops under memory pressure, and what memory allows still caps it
  if that is lower than your value. So a large value raises what a run *may* grow to rather than
  what it will get, and a small one holds a device below what its free RAM would otherwise permit.
  Where free memory cannot be probed at all there is nothing to adapt against, and your value is
  used exactly as given (the default there is a static 8).
- `memory_limit`: Default `0` (disabled). Set a number for MB (for example `900`) or a percent
  string of total RAM (for example `"15%"`). Once the Anki process RSS passes that cap, the addon
  starts backing off concurrency. A value of `0` leaves this hard cap off.

The progress dialog shows the current limit, free memory, and the measured cost per task once a
window has completed.

### request_timeout

Default `180`. Seconds to wait for a single API response before giving up on that attempt.
Timeouts are retried, subject to `max_request_retries`.

## config fields per note type name

Add the fields by note type like this. You can set multiple different note types. You can't set
multiple fields per note type though.

```json
{
  "note type name A": {
    "meaning_field": "note A meaning field",
    "word_field": "note A word field",
    "sentence_field": "note A sentence field",
    "insert_deck": "My deck::sub deck 1"
  },
  "note type name B": {
    "meaning_field": "note B meaning field",
    "word_field": "note B word field",
    "sentence_field": "note B sentence field",
    "translation_field": "note B translation field",
    "insert_deck": "My deck::sub deck""
  },
  ...etc
}
```

You need to define

- for cleaning/generating word meanings:
  1. `meaning_field`
  2. `english_meaning_field`
  3. `word_field`
  4. `word_reading_field`
  5. `sentence_field`
  - `mdx_filenames`: array of filenames (with `.mdx` extension) for dictionary files in the addon's `user_files/` folder.
    - The `user_files/` folder is created automatically if it does not exist. Example: `["dict1.mdx", "dict2.mdx"]`
  - `mdx_pick_dictionary`: one of "all", "first", "shortest", "longest"
- for translating sentences
  1. `sentence_field`
  2. `translated_sentence_field`
- for generating kanji stories:
  1. `kanji_field`
  2. `kanji_story_field`
- for kanjifying sentences:
  1. `furigana_sentence_field`
  2. `kanjified_sentence_field`
- for extracting words:
  1. `word_extraction_sentence_field`
  2. `word_list_field`
- for matching extracted words:
  1. `word_list_field`
  2. `word_kanjified_field`
  3. `word_normal_field`
  4. `word_reading_field`
  5. `word_sort_field`
  6. `meaning_field`
  7. `english_meaning_field`
  8. `part_of_speech_field`
  9. `new_note_id_field`
  10. `insert_deck` (optional) Used when generating TSVs for inserting new notes. If omitted, the
      file will simply not specify the deck

## optipnal specification

### `extract_words` operatation

- `ignore_current_word_lists`: (default: false) don't pass the current field to the prompt, making the model recreate the result from scratch

### `match_words_model` operation

-`word_lists_to_process` to select what parts of speech you collect:
    - `nouns`: default = yes
    - `proper_nouns`: default = no
    - `verbs`: default = yes
    - `compound_verbs`: default = yes
    - `adjectives`: default = yes
    - `adverbs`: default = yes
    - `adjectivals`: default = yes
    - `particles`: default = no
    - `pronouns`: default = yes
    - `suffixes`: default = yes
    - `expressions`: default = yes
    - `yojijukugo`: default = yes

- `replace_existing_matched_words`: (default: false) overwrite previously processed matched words?
