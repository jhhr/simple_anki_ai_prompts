import json
import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable, Coroutine, Any, NamedTuple, Union
from functools import partial

from anki.notes import Note, NoteId
from anki.collection import Collection, OpChanges
from aqt import mw
from aqt.browser import Browser
from aqt.operations import CollectionOp
from aqt.utils import tooltip
from collections.abc import Sequence

from .api_client import (
    ANTHROPIC,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_RETRY_WAIT_SECONDS,
    GEMINI,
    OPENAI,
    TOGETHER,
    begin_run,
    cancel_run,
    close_all_sessions,
    end_run,
    is_cancelled,
    join_run,
    post_with_retry,
    rate_limit_tracker,
    run_cancelled,
    set_connection_pool_size,
)
from .collection_access import RunCancelled, begin_cleanup_phase, end_cleanup_phase
from .concurrency import TASK_QUEUE_DEPTH, ConcurrencyGate, max_possible_concurrency
from .diagnostics import (
    clear_cancel_time,
    diagnostic_level,
    dump_thread_stacks,
    note_cancel_time,
    seconds_since_cancel,
    start_cancel_watchdog,
)

from ..utils import get_field_config, print_error_traceback

from ..make_notes_tsv import make_tsv_from_notes, import_tsv_file

logger = logging.getLogger(__name__)

MAX_TOKENS_VALUE = 8000
# Shortest gap between progress dialog redraws. Redraws run on Anki's main thread, so this is
# what keeps a burst of finishing tasks from starving the UI.
PROGRESS_UPDATE_INTERVAL = 0.15
DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a helpful assistant for processing Japanese text. You are a"
    " superlative expert in the Japanese language and its writing system."
    " You are designed to output JSON."
)

OPENAI_FIXED_TEMPERATURE_MODEL_PREFIXES = (
    "gpt-5",
    "o1",
    "o3",
    "o4",
)

ANTHROPIC_FIXED_TEMPERATURE_MODEL_PREFIXES = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-mythos-preview",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
)


def log_phase(label: str, started: float, **extra) -> float:
    """Log how long a shutdown or cleanup step took, and return a fresh start marker.

    Cancellation problems show up as one of these steps blocking on something, and which step
    it is narrows the cause down immediately. Debug logging shows them for every run; once a
    run has been cancelled they are logged at info level too, since that is the case where
    knowing which step is slow actually matters and a cancelled run only emits a handful.
    """
    now = time.monotonic()
    level = diagnostic_level()
    if logger.isEnabledFor(level):
        details = "".join(f" {k}={v}" for k, v in extra.items())
        since_cancel = seconds_since_cancel()
        if since_cancel is not None:
            details += f" since_cancel={since_cancel:.1f}s"
        logger.log(level, "[phase] %s took %.3fs%s", label, now - started, details)
    return now


class CancelState:
    """Shared state for cancellation that can be accessed across threads."""

    def __init__(self):
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def is_cancelled(self):
        return self._cancelled


def get_response(
    model: str,
    prompt: str,
    cancel_state: Optional[CancelState] = None,
    instructions: Optional[str] = None,
    response_schema: Optional[dict] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    json_result_corrector: Optional[Callable[[str], str]] = None,
) -> Union[dict, None]:
    """Get a response from the appropriate model based on the configuration.

    Args:
        model: The model to use for the request.

    Returns:
        A dict containing the parsed JSON response, or None if there was an error.
    """
    if model.startswith("gemini"):
        return get_response_from_gemini(
            model,
            prompt,
            cancel_state=cancel_state,
            instructions=instructions,
            response_schema=response_schema,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            json_result_corrector=json_result_corrector,
        )
    elif model.startswith("gpt") or model.startswith("o3") or model.startswith("o1"):
        return get_response_from_openai(
            model,
            prompt,
            cancel_state=cancel_state,
            instructions=instructions,
            response_schema=response_schema,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            json_result_corrector=json_result_corrector,
        )
    elif model.startswith("claude") or model.startswith("anthropic"):
        return get_response_from_anthropic(
            model,
            prompt,
            cancel_state=cancel_state,
            instructions=instructions,
            response_schema=response_schema,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            json_result_corrector=json_result_corrector,
        )
    elif "/" in model:
        return get_response_from_together(
            model,
            prompt,
            cancel_state=cancel_state,
            instructions=instructions,
            response_schema=response_schema,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            json_result_corrector=json_result_corrector,
        )
    else:
        logger.error(f"Unsupported model: {model}")
        return None


def post_to_api(
    provider: str,
    model: str,
    url: str,
    headers: dict,
    data: dict,
    config: dict,
    cancel_state: Optional[CancelState] = None,
):
    """Send a request to a provider, waiting out any rate limits it reports.

    Blocking, and always called from a worker thread. Returns the final response, or None if
    the request was cancelled or never got a response.
    """
    return post_with_retry(
        provider=provider,
        model=model,
        url=url,
        headers=headers,
        json_body=data,
        timeout=config.get("request_timeout", 300),
        cancel_state=cancel_state,
        max_retries=int(config.get("max_request_retries", DEFAULT_MAX_RETRIES)),
        max_retry_wait=float(config.get("max_retry_wait_seconds", DEFAULT_MAX_RETRY_WAIT_SECONDS)),
    )


def decode_json_result(json_str: str):
    logging.debug("json_result", json_str)
    try:
        result = json.loads(json_str)
        logger.debug(
            "Parsed result from json: %s", json.dumps(result, ensure_ascii=False, indent=2)
        )
        return result
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON response, json_result: {json_str}")
        return None
    except ValueError as ve:
        logger.error(f"Failed to parse JSON response - ValueError: {ve}")
        if "integer string conversion" in str(ve):
            logger.error("Large integer detected in JSON response")
        logger.error(f"json_result: {json_str}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing JSON: {e}")
        logger.error(f"json_result: {json_str}")
        return None


def clean_response_schema_for_gemini(schema: dict) -> dict:
    """Cleans the response schema to be compatible with Gemini's expected format.

    Args:
        response_schema: The original response schema.

    Returns:
        A cleaned response schema compatible with Gemini.
    """
    # Clean "additionalProperties" from array items and objects to avoid Gemini rejecting the schema
    if isinstance(schema, dict):
        if schema.get("type") == "array" and "items" in schema:
            items = schema["items"]
            if isinstance(items, dict):
                if "additionalProperties" in items:
                    del items["additionalProperties"]
                # Recursively clean items
                schema["items"] = clean_response_schema_for_gemini(items)
        elif schema.get("type") == "object" and "properties" in schema:
            if "additionalProperties" in schema:
                del schema["additionalProperties"]
            for key, value in schema["properties"].items():
                schema["properties"][key] = clean_response_schema_for_gemini(value)
    return schema


def openai_supports_custom_temperature(model: str) -> bool:
    return not model.startswith(OPENAI_FIXED_TEMPERATURE_MODEL_PREFIXES)


def anthropic_supports_custom_temperature(model: str) -> bool:
    normalized_model = model
    if model.startswith("anthropic/"):
        normalized_model = model.split("/", 1)[1]
    return not normalized_model.startswith(ANTHROPIC_FIXED_TEMPERATURE_MODEL_PREFIXES)


def anthropic_response_indicates_unsupported_temperature(response_text: str) -> bool:
    text = response_text.lower()
    return "temperature" in text and (
        "deprecated for this model" in text or "not supported for this model" in text
    )


def get_response_from_gemini(
    model: str,
    prompt: str,
    cancel_state: Optional[CancelState] = None,
    instructions: Optional[str] = None,
    response_schema: Optional[dict] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    json_result_corrector: Optional[Callable[[str], str]] = None,
) -> Union[dict, None]:
    """Get a response from Google's Gemini API.

    Args:
        prompt: The prompt to send to the API.

    Returns:
        A dict containing the parsed JSON response, or None if there was an error.
    """
    logger.debug(f"Gemini call, model: {model}")

    if is_cancelled(cancel_state):
        return None

    # Create the request body
    data: dict[str, Any] = {
        "contents": [
            {
                # "role": "user",
                "parts": [
                    {"text": prompt},
                ],
            },
        ],
        "system_instruction": {
            "parts": [
                {
                    "text": (
                        instructions
                        if instructions
                        else (
                            "You are a helpful assistant for processing Japanese text. You are a"
                            " superlative expert in the Japanese language and its writing system."
                            " You are designed to output JSON."
                        )
                    )
                },
            ]
        },
        "generationConfig": {
            "responseMimeType": "application/json",
            # maxOutputTokens includes both thinking and output so it needs to be large enough
            "maxOutputTokens": max_output_tokens or MAX_TOKENS_VALUE,
            "thinkingConfig": {"thinkingBudget": 6000},
        },
    }
    if max_output_tokens is not None:
        logger.debug("Using max_output_tokens %d", max_output_tokens)
    if temperature is not None:
        data["generationConfig"]["temperature"] = temperature
        logger.debug("Using temperature %s", temperature)
    if response_schema:
        response_schema = clean_response_schema_for_gemini(response_schema)
        data["generationConfig"]["responseSchema"] = response_schema
        logger.debug(
            "Using response schema %s", json.dumps(response_schema, ensure_ascii=False, indent=2)
        )

    headers = {
        "Content-Type": "application/json",
        # "x-goog-api-key": google_api_key,
    }

    config = mw.addonManager.getConfig(__name__)
    if config is None:
        print("No configuration found for the addon.")
        return None
    google_api_key = config.get("google_api_key", "")

    # Make the API call
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={google_api_key}"
    )
    response = post_to_api(
        provider=GEMINI,
        model=model,
        url=url,
        headers=headers,
        data=data,
        config=config,
        cancel_state=cancel_state,
    )
    if response is None:
        return None

    if response.status_code != 200:
        logger.error(f"Error: {response.status_code}, {response.text}")
        return None

    try:
        decoded_json = json.loads(response.text)
        # Extract content from Gemini response structure
        content_text = decoded_json["candidates"][0]["content"]["parts"][0]["text"]
    except json.JSONDecodeError as je:
        logger.error(f"Error decoding JSON: {je}")
        logger.error("response %s", response.text)
        return None
    except KeyError as ke:
        logger.error(f"Error extracting content: {ke}")
        logger.error("response %s", response.text)
        return None

    # Extract the JSON from the response
    json_result = extract_json_string(content_text)

    result = decode_json_result(json_result)
    if not result and json_result_corrector:
        json_result = json_result_corrector(json_result)
        result = decode_json_result(json_result)
    return result


def get_response_from_openai(
    model: str,
    prompt: str,
    cancel_state: Optional[CancelState] = None,
    instructions: Optional[str] = None,
    response_schema: Optional[dict] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    json_result_corrector: Optional[Callable[[str], str]] = None,
) -> Union[dict, None]:
    logger.debug("OpenAI call, model %s", model)

    if is_cancelled(cancel_state):
        return None

    # Use max_completion_tokens instead of max_tokens for o3

    messages = [
        {
            "role": "system",
            "content": (
                instructions
                if instructions
                else (
                    "You are a helpful assistant for processing Japanese text. You are a"
                    " superlative expert in the Japanese language and its writing system. You are"
                    " designed to output JSON."
                )
            ),
        },
        {"role": "user", "content": prompt},
    ]
    config = mw.addonManager.getConfig(__name__)
    if config is None:
        logger.error("No configuration found for the addon.")
        return None
    openai_api_key = config.get("openai_api_key", "")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}",
    }

    data: dict[str, Any] = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": messages,
    }
    if any(model.startswith(m) for m in ["o3", "o1", "gpt-5"]):
        # This is total completion tokens, including both reasoning and output
        data["max_completion_tokens"] = max_output_tokens or MAX_TOKENS_VALUE
    else:
        # This is for GPT models and only limits output, not reasoning
        data["max_tokens"] = max_output_tokens or MAX_TOKENS_VALUE
    if temperature is not None:
        if openai_supports_custom_temperature(model):
            data["temperature"] = temperature
            logger.debug("Using temperature %s", temperature)
        else:
            logger.debug(
                "Skipping custom temperature %s for model %s because this OpenAI chat"
                " model family only accepts the default temperature.",
                temperature,
                model,
            )
    if response_schema:
        data["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "schema_name",
                "schema": response_schema,
                "strict": True,
            },
        }
        logger.debug(
            "Using response schema %s", json.dumps(response_schema, ensure_ascii=False, indent=2)
        )

    # Make the API call
    response = post_to_api(
        provider=OPENAI,
        model=model,
        url="https://api.openai.com/v1/chat/completions",
        headers=headers,
        data=data,
        config=config,
        cancel_state=cancel_state,
    )
    if response is None:
        return None

    if response.status_code != 200:
        logger.error(f"Error: {response.status_code}, {response.text}")
        return None

    try:
        decoded_json = json.loads(response.text)
        content_text = decoded_json["choices"][0]["message"]["content"]
    except json.JSONDecodeError as je:
        logger.error(f"Error decoding JSON: {je}")
        logger.error("response %s", response.text)
        return None
    except KeyError as ke:
        logger.error(f"Error extracting content: {ke}")
        logger.error("response %s", response.text)
        return None

    # Extract the cleaned meaning from the response
    json_result = extract_json_string(content_text)

    result = decode_json_result(json_result)
    if not result and json_result_corrector:
        json_result = json_result_corrector(json_result)
        result = decode_json_result(json_result)
    return result


def get_response_from_together(
    model: str,
    prompt: str,
    cancel_state: Optional[CancelState] = None,
    instructions: Optional[str] = None,
    response_schema: Optional[dict] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    json_result_corrector: Optional[Callable[[str], str]] = None,
) -> Union[dict, None]:
    logger.debug("Together AI call, model %s", model)

    if is_cancelled(cancel_state):
        return None

    messages = [
        {
            "role": "system",
            "content": (
                instructions
                if instructions
                else (
                    "You are a helpful assistant for processing Japanese text. You are a"
                    " superlative expert in the Japanese language and its writing system. You are"
                    " designed to output JSON."
                )
            ),
        },
        {"role": "user", "content": prompt},
    ]
    config = mw.addonManager.getConfig(__name__)
    if config is None:
        logger.error("No configuration found for the addon.")
        return None
    together_api_key = config.get("together_api_key", "")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {together_api_key}",
    }

    data: dict[str, Any] = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": messages,
        "max_tokens": max_output_tokens or MAX_TOKENS_VALUE,
    }
    if temperature is not None:
        data["temperature"] = temperature
        logger.debug("Using temperature %s", temperature)

    # Make the API call
    response = post_to_api(
        provider=TOGETHER,
        model=model,
        url="https://api.together.xyz/v1/chat/completions",
        headers=headers,
        data=data,
        config=config,
        cancel_state=cancel_state,
    )
    if response is None:
        return None

    if response.status_code != 200:
        logger.error(f"Error: {response.status_code}, {response.text}")
        return None

    try:
        decoded_json = json.loads(response.text)
        content_text = decoded_json["choices"][0]["message"]["content"]
    except json.JSONDecodeError as je:
        logger.error(f"Error decoding JSON: {je}")
        logger.error("response %s", response.text)
        return None
    except KeyError as ke:
        logger.error(f"Error extracting content: {ke}")
        logger.error("response %s", response.text)
        return None

    json_result = extract_json_string(content_text)

    result = decode_json_result(json_result)
    if not result and json_result_corrector:
        json_result = json_result_corrector(json_result)
        result = decode_json_result(json_result)
    return result


def get_response_from_anthropic(
    model: str,
    prompt: str,
    cancel_state: Optional[CancelState] = None,
    instructions: Optional[str] = None,
    response_schema: Optional[dict] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    json_result_corrector: Optional[Callable[[str], str]] = None,
) -> Union[dict, None]:
    """Get a response from Anthropic's Claude API.

    Args:
        prompt: The prompt to send to the API.

    Returns:
        A dict containing the parsed JSON response, or None if there was an error.
    """
    logger.debug("Anthropic call, model %s", model)

    if is_cancelled(cancel_state):
        return None

    messages = [
        {"role": "user", "content": prompt},
    ]
    # Create the request body
    data: dict[str, Any] = {
        "model": model,
        "system": (
            instructions
            if instructions
            else (
                "You are a helpful assistant for processing Japanese text. You are a"
                " superlative expert in the Japanese language and its writing system. You are"
                " designed to output JSON."
            )
        ),
        "max_tokens": max_output_tokens or MAX_TOKENS_VALUE,
        "messages": messages,
    }
    use_temperature = temperature is not None and anthropic_supports_custom_temperature(model)
    if use_temperature:
        data["temperature"] = temperature
        logger.debug("Using temperature %s", temperature)
    elif temperature is not None:
        logger.debug(
            "Skipping custom temperature %s for model %s because this Anthropic model family"
            " only accepts the default temperature.",
            temperature,
            model,
        )

    if response_schema:
        data["output_config"] = {
            "format": {
                "type": "json_schema",
                "schema": response_schema,
            }
        }
        logger.debug(
            "Using response schema %s", json.dumps(response_schema, ensure_ascii=False, indent=2)
        )

    config = mw.addonManager.getConfig(__name__)
    if config is None:
        logger.error("No configuration found for the addon.")
        return None
    anthropic_api_key = config.get("anthropic_api_key", "")

    headers = {
        "x-api-key": anthropic_api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    # Make the API call
    url = "https://api.anthropic.com/v1/messages"
    response = post_to_api(
        provider=ANTHROPIC,
        model=model,
        url=url,
        headers=headers,
        data=data,
        config=config,
        cancel_state=cancel_state,
    )
    if response is None:
        return None

    if (
        response.status_code == 400
        and "temperature" in data
        and anthropic_response_indicates_unsupported_temperature(response.text)
    ):
        logger.warning(
            "Anthropic model %s rejected custom temperature; retrying with default temperature.",
            model,
        )
        retry_data = dict(data)
        retry_data.pop("temperature", None)

        response = post_to_api(
            provider=ANTHROPIC,
            model=model,
            url=url,
            headers=headers,
            data=retry_data,
            config=config,
            cancel_state=cancel_state,
        )
        if response is None:
            return None

    if response.status_code != 200:
        logger.error(f"Error: {response.status_code}, {response.text}")
        return None

    try:
        decoded_json = json.loads(response.text)
        content_blocks = decoded_json.get("content", [])
        text_blocks = [
            block.get("text", "")
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        content_text = "\n".join([text for text in text_blocks if text]).strip()
        if not content_text:
            raise KeyError("content text blocks")
    except json.JSONDecodeError as je:
        logger.error(f"Error decoding JSON: {je}")
        logger.error("response %s", response.text)
        return None
    except KeyError as ke:
        logger.error(f"Error extracting content: {ke}")
        logger.error("response %s", response.text)
        return None

    # Extract the cleaned meaning from the response
    json_result = extract_json_string(content_text)

    result = decode_json_result(json_result)
    if not result and json_result_corrector:
        json_result = json_result_corrector(json_result)
        result = decode_json_result(json_result)
    return result


def extract_json_string(content_text):
    # Add logic to extract the cleaned meaning from the GPT response
    # You may need to parse the JSON or use other string manipulation techniques
    # based on the structure of the response.

    # For simplicity, let's assume that the stuff asked for is surrounded by curly braces in the
    # response.
    # Find the first occurrence of "{" and the last occurrence of "}" in the response.
    start_index = content_text.find("{")
    end_index = content_text.rfind("}")

    if start_index != -1 and end_index != -1:
        return content_text[start_index : end_index + 1]
    else:
        print("Did not return JSON parseable result")
        return content_text


class CancelManager:
    """
    A class to manage cancellation of asynchronous operations.
    It provides a way to set and check cancellation requests.
    """

    def __init__(
        self,
        tasks,
        cancel_state: Optional[CancelState] = None,
        progress_updater: Optional["AsyncTaskProgressUpdater"] = None,
    ):
        self.cancel_requested = False
        self.tasks = tasks  # List of tasks to cancel if needed
        self.cancel_state = cancel_state or CancelState()
        self.progress_updater = progress_updater
        # Lets whoever is waiting on the tasks stop waiting the moment cancel is requested,
        # rather than having to see every task through to the end
        self.cancelled_event = asyncio.Event()
        self.monitor_task = asyncio.create_task(self.monitor_for_cancellation())

    async def wait_cancelled(self) -> None:
        await self.cancelled_event.wait()

    def request_cancel(self):
        """Request cancellation of all tasks managed by this instance."""
        started = time.monotonic()
        pending = sum(1 for t in self.tasks if not t.done())
        logger.info(
            "Cancellation requested: %d tasks (%d still running), %d threads alive",
            len(self.tasks),
            pending,
            threading.active_count(),
        )
        self.cancel_requested = True
        self.cancelled_event.set()
        note_cancel_time()
        # What the worker threads are doing at the moment of the cancel, and then repeatedly
        # until they are gone. Everything that has made cancelling slow so far has been work
        # continuing in threads nothing was waiting for, and this is what shows it.
        dump_thread_stacks("at cancel")
        start_cancel_watchdog()

        # Set the cancelled state first: this is what actually stops the work. Requests that
        # have not been sent yet are skipped, cooldown waits wake up, requests already in
        # flight have their sockets shut down so the threads waiting on them return at once,
        # and any response that still arrives afterwards is discarded instead of being acted
        # on. cancel_run() is the one that matters - the ops overwhelmingly do not pass their
        # cancel_state down, so without it their abandoned threads keep calling APIs and
        # querying the collection.
        cancel_run()
        self.cancel_state.cancel()

        # Drops the pooled connections so nothing new reuses them. The in-flight ones were
        # already aborted by cancel_run() above.
        close_all_sessions()

        # Show the cancelling message and stop every other progress update from here on. A
        # cancelled run unwinds hundreds of tasks at once, and each one asking the main thread
        # to redraw the dialog is what made the window lock up instead of closing.
        if self.progress_updater is not None:
            self.progress_updater.show_cancelling()
        else:
            mw.taskman.run_on_main(
                lambda: mw.progress.update(
                    label="<b>Cancelling operations...</b><br>Finishing up.",
                    value=0,
                    max=0,  # Indeterminate progress
                )
            )

        started = log_phase("cancel: show_cancelling", started)

        # Cancel all tasks without waiting
        for task in self.tasks:
            if not task.done():
                task.cancel()
        self.monitor_task.cancel()
        log_phase("cancel: cancel all tasks", started)

    def is_cancel_requested(self) -> bool:
        """Check if cancellation has been requested."""
        return self.cancel_requested

    async def monitor_for_cancellation(self):
        """Monitor for cancellation requests and cancel all tasks if requested."""
        try:
            while not self.cancel_requested:
                # Check for cancellation request from Anki
                if mw.progress.want_cancel():
                    logger.debug("Cancellation requested, setting cancel_requested to True")
                    self.request_cancel()
                    break

                # Check if all tasks are completed naturally
                if all(task.done() for task in self.tasks):
                    logger.debug("All tasks completed naturally, exiting monitor")
                    break

                # Check frequently but don't hog the CPU
                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            logger.debug("Cancellation monitor task cancelled")
            # Just exit the task when cancelled


async def await_tasks_or_cancel(
    tasks: "list[asyncio.Task]", cancel_manager: CancelManager
) -> None:
    """Wait for a window's tasks, but stop waiting the instant cancellation is requested.

    Waiting for the tasks to unwind is not safe to rely on. A task blocked in a worker thread
    detaches when cancelled, but nothing can interrupt the blocking HTTP request itself, and
    any task that swallows the cancellation or is stuck on something uninterruptible would
    hold the whole run open - which is what made cancelling hang for minutes on one straggler.

    So on cancellation we simply stop waiting. The tasks are cancelled and abandoned; their
    requests finish in their own threads and the results are discarded (post_with_retry throws
    away anything that arrives after a cancel), and the run moves on to saving what it already
    has.
    """
    started = time.monotonic()
    if not tasks:
        # asyncio.wait rejects an empty set, where gather was happy with one
        return
    # A task rather than asyncio.gather: cancelling a gather leaves a future holding a
    # CancelledError that nothing reads, and asyncio complains about it on the console once the
    # loop has closed and the callback that would have read it can no longer run. A cancelled
    # task is simply cancelled, with nothing left to retrieve.
    all_done: "asyncio.Future[Any]" = asyncio.ensure_future(asyncio.wait(tasks))
    cancel_waiter: "asyncio.Future[Any]" = asyncio.ensure_future(cancel_manager.wait_cancelled())
    waiting: "set[asyncio.Future[Any]]" = {all_done, cancel_waiter}
    try:
        await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
    finally:
        cancel_waiter.cancel()
        abandoned = 0
        if not all_done.done():
            all_done.cancel()
            abandoned = sum(1 for t in tasks if not t.done())
        # asyncio.wait does not read its tasks' results, so anything a finished task raised is
        # still sitting in it unretrieved. process_op handles its own errors, so reaching here
        # with one means something got past it and is worth seeing rather than being reported
        # much later against a closed loop.
        for task in tasks:
            if task.done() and not task.cancelled():
                error = task.exception()
                if error is not None:
                    logger.error("Task failed: %s", error)
                    print_error_traceback(error, logger)
        log_phase(
            "await window tasks",
            started,
            tasks=len(tasks),
            cancelled=cancel_manager.is_cancel_requested(),
            abandoned=abandoned,
            threads=threading.active_count(),
        )


class AsyncTaskProgressUpdater:
    """A class to update the progress dialog in async ops."""

    def __init__(
        self, total_notes: Optional[int] = None, total_tasks: int = 0, title: Optional[str] = None
    ):
        self.total_tasks = total_tasks
        self.tasks_done = 0
        self.tasks_in_progress = 0
        self.notes_done = 0
        self.total_notes = total_notes
        # Sum of each task's execution time, for estimating average time per task
        self.cumulative_task_time = 0.0
        self.max_task_time = 0.0
        self.start_time = time.time()
        # Counters are incremented from both the event loop and executor threads
        self._counts_lock = threading.Lock()
        # Set by the bulk ops so the dialog can show what the concurrency gate is doing
        self.gate: Optional[ConcurrencyGate] = None
        # Every finishing task asks for a redraw, so bursts have to be coalesced - see _push
        self._ui_lock = threading.Lock()
        self._update_pending = False
        self._last_update_at = 0.0
        self._suppressed = False
        if title is None:
            title = "Processing asynchronous tasks..."
        self.set_title(title)

        # Periodic updater (started when an event loop is running)
        self.autoupdate_task: Optional[asyncio.Task] = None
        self._autoupdate_started = False
        self._autoupdate_deferred = False
        self.start_autoupdate()

    def start_autoupdate(self):
        """Start periodic progress updates if an event loop is running."""
        if self.autoupdate_task and not self.autoupdate_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop yet; allow caller to retry later from async context
            self._autoupdate_deferred = True
            return
        self.autoupdate_task = loop.create_task(self._periodic_update_progress())
        self._autoupdate_started = True
        self._autoupdate_deferred = False

    async def _periodic_update_progress(self):
        while True:
            self.update_progress()
            await asyncio.sleep(1)

    def stop_autoupdate(self):
        """Stop the periodic progress update task."""
        if self.autoupdate_task:
            self.autoupdate_task.cancel()
            self.autoupdate_task = None

    def set_total_notes(self, total_notes: int):
        self.total_notes = total_notes

    def set_total_tasks(self, total_tasks: int):
        self.total_tasks = total_tasks

    def set_title(self, title: str):
        mw.taskman.run_on_main(lambda: mw.progress.set_title(title))

    def _push(self, label: str, value: int, maximum: int, force: bool = False) -> None:
        """Queue a dialog redraw, collapsing bursts into a single update.

        Progress is refreshed from every task as it finishes, so a run with a high
        concurrency limit produces hundreds of these at once - and a cancelled run produces
        them all in the same instant, as every task unwinds together. Each one has to be run
        on the main thread, so letting them all through buries Anki's event loop: the window
        stops repainting and the click on Cancel isn't even seen for a long time.

        At most one redraw is ever queued, and only one every PROGRESS_UPDATE_INTERVAL, so a
        burst of any size costs the main thread exactly one update.
        """
        with self._ui_lock:
            if self._suppressed and not force:
                return
            if self._update_pending:
                return
            now = time.time()
            if not force and now - self._last_update_at < PROGRESS_UPDATE_INTERVAL:
                return
            self._update_pending = True
            self._last_update_at = now

        def run_update():
            try:
                mw.progress.update(label=label, value=value, max=maximum)
            finally:
                with self._ui_lock:
                    self._update_pending = False

        mw.taskman.run_on_main(run_update)

    def show_cancelling(self) -> None:
        """Switch the dialog to the cancelling message and stop all further updates."""
        with self._ui_lock:
            self._suppressed = False
            self._update_pending = False
        self._push(
            "<b>Cancelling operations...</b><br>Finishing up.",
            value=0,
            maximum=0,  # Indeterminate progress
            force=True,
        )
        with self._ui_lock:
            self._suppressed = True

    def increment_counts(
        self,
        total_tasks=0,
        tasks_done: int = 0,
        tasks_in_progress: int = 0,
        notes_done: int = 0,
        cumulative_task_time: float = 0.0,
    ):
        """Increment the counts of tasks and notes."""
        with self._counts_lock:
            self.total_tasks += total_tasks
            self.tasks_done += tasks_done
            self.tasks_in_progress += tasks_in_progress
            self.notes_done += notes_done
            self.cumulative_task_time += cumulative_task_time
            if cumulative_task_time > self.max_task_time:
                self.max_task_time = cumulative_task_time

    def update_progress(self):
        """Update the Step 1 progress dialog with the current task and note counts."""
        task_progress_msg = f"""<strong>Processing:</strong>
            <br><strong><code>{self.tasks_done}/{self.total_tasks}</code></strong>
            tasks <small style="opacity: 0.85"> | Waiting response: {self.tasks_in_progress}</small>
            """
        if self.gate is not None:
            task_progress_msg += (
                f'<br><small style="opacity: 0.85">Running: {self.gate.status_text()}</small>'
            )
        if self.total_notes is not None:
            tasks_per_note = (
                round(self.tasks_done / self.notes_done, 1) if self.notes_done > 0 else 0
            )
            task_progress_msg += (
                f"<br><strong><code>{self.notes_done}/{self.total_notes}</code></strong> notes"
                f' <small style="opacity: 0.85"> | Avg tasks per note: {tasks_per_note}</small>'
            )

        elapsed_s = time.time() - self.start_time
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_s))
        # estimate time remaining from tasks_done and elapsed_time
        time_msg = f"<br><code>Time: {elapsed_time}</code>"
        if self.tasks_done > 3:
            eta_s = (self.total_tasks - self.tasks_done) * (elapsed_s / self.tasks_done)
            eta_time = time.strftime("%H:%M:%S", time.gmtime(eta_s))
            avg_per_op_s = self.cumulative_task_time / self.tasks_done
            time_msg += f""" | <small> Avg time per task: {avg_per_op_s:.2f}s
            | Max: {self.max_task_time:.2f}s</small>
            <br><code>ETA: {eta_time}</code>"""
        self._push(f"{task_progress_msg}{time_msg}", self.tasks_done, self.total_tasks)

    def update_preparation_progress(
        self,
        notes_prepared: int = 0,
        total_notes: int = 0,
        tasks_planned: int = 0,
    ):
        """Update the dialog while a nested op works out what it has to do.

        Nothing is running yet at this point, but an op that fans out per note has to read
        every note's word list before it knows its task total, and for a large selection that
        pass takes long enough to look like a hang if the dialog says nothing.
        """
        elapsed_s = time.time() - self.start_time
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_s))
        task_progress_msg = f"""<strong>Preparing:</strong>
            <br><strong><code>{notes_prepared}/{total_notes}</code></strong> notes read
            <small style="opacity: 0.85"> | Tasks found: {tasks_planned}</small>
            <br><code>Time: {elapsed_time}</code>"""
        self._push(task_progress_msg, notes_prepared, total_notes)

    def update_note_adding_progress(
        self,
        notes_added: int = 0,
        total_notes: int = 0,
        failed: int = 0,
    ):
        """
        Update the Step 2 progress dialog for note adding operations occuring after async tasks
        are done.
        """
        try:
            elapsed_s = time.time() - self.start_time
            elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_s))
            time_msg = f"<br><code>Time: {elapsed_time}</code>"
            if notes_added > 0:
                eta_s = (notes_added - self.notes_done) * (elapsed_s / notes_added)
                eta_time = time.strftime("%H:%M:%S", time.gmtime(eta_s))
                avg_per_note_s = elapsed_s / notes_added
                time_msg += f""" | <small> Avg time per note: {avg_per_note_s:.2f}s</small>
                <br><code>ETA: {eta_time}</code>"""
            task_progress_msg = f"""<strong>Adding notes:</strong>
                <br><strong><code>{notes_added}/{total_notes}</code></strong> notes"""
            if failed > 0:
                task_progress_msg += f""" | <strong style="color: red;">{failed} failed</strong>"""
        except Exception as e:
            logger.error("Error updating note adding progress: %s", e)
            return
        self._push(f"{task_progress_msg}{time_msg}", notes_added, total_notes)

    def update_new_note_processing_progress(
        self,
        new_notes_processed: int = 0,
        total_notes: int = 0,
    ):
        """Update the Step 3 progress dialog for processing new notes after they have been added."""
        elapsed_s = time.time() - self.start_time
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_s))
        time_msg = f"<br><code>Time: {elapsed_time}</code>"
        if new_notes_processed > 0:
            eta_s = (total_notes - new_notes_processed) * (elapsed_s / new_notes_processed)
            eta_time = time.strftime("%H:%M:%S", time.gmtime(eta_s))
            avg_per_note_s = elapsed_s / new_notes_processed
            time_msg += f""" | <small> Avg time per note: {avg_per_note_s:.2f}s</small>
            <br><code>ETA: {eta_time}</code>"""
        task_progress_msg = f"""<strong>Processing new notes:</strong>
            <br><strong><code>{new_notes_processed}/{total_notes}</code></strong> notes"""
        self._push(f"{task_progress_msg}{time_msg}", new_notes_processed, total_notes)


def make_inner_bulk_op(
    config: dict,
    op: Callable[..., bool],
    gate: ConcurrencyGate,
    progress_updater: AsyncTaskProgressUpdater,
    handle_op_error: Callable[[Exception], None],
    handle_op_result: Callable[[bool], None],
    cancel_state: Optional[CancelState] = None,
    one_task_per_op: bool = False,
) -> Callable[..., Coroutine[Any, Any, bool]]:
    """
    Creates an asynchronous operation processor for bulk operations, limited by the shared
    concurrency gate and reporting progress as it goes.

    Requests are not paced here: each provider throttles itself against the rate-limit
    responses it gets back (see api_client.post_with_retry). What this limits instead is how
    many operations are in flight at once, which is what drives memory use.

    :param config (dict): Addon config
    :param op (Callable[[dict, ...], bool]): The operation function to execute for each item. It
            accepts the config dictionary as the first argument, followed by additional arguments.
    :param gate (ConcurrencyGate): Shared gate limiting concurrent operations. Must be the same
            instance for every task in a bulk run, otherwise nothing is actually limited.
    :param progress_updater (AsyncTaskProgressUpdater): Progress dialog updater.
    :param handle_op_error (Callable[[Exception], None]): Callback to handle exceptions raised during
            operation execution.
    :param handle_op_result (Callable[[bool], None]): Callback to handle the result of each operation.
    :param cancel_state (Optional[CancelState]): Shared cancellation state.
    :param one_task_per_op (bool): Whether each operation corresponds to a single task for progress
            tracking. If False, the caller is responsible for updating done note counts.

    Returns:
        Callable[..., Coroutine[Any, Any, bool]]: An asynchronous function that processes a
            single operation.
    """
    cancel_state = cancel_state or CancelState()

    # Wrapper function to process a single note
    async def process_op(
        notes_to_add_dict: dict[str, list[Note]],
        notes_to_update_dict: dict[NoteId, Note],
        **op_args,
    ) -> bool:
        """Process a single operation, waiting for a slot in the concurrency gate first.
        Args:
            notes_to_add_dict (dict[str, list[Note]]): Dictionary of notes to add.
            notes_to_update_dict (dict[NoteId, Note]): Dictionary of notes to update.
            **op_args: Additional keyword arguments to pass to the operation function.
        Returns:
            bool: The result of the operation, True if successful, False otherwise.
        """
        op_result = False
        try:
            # Wait until memory allows another operation to start
            await gate.acquire()
            try:
                # Check for cancel request before starting the operation
                if mw.progress.want_cancel():
                    logger.debug("Inner bulk operation mw.progress.want_cancel()")
                    return False

                progress_updater.increment_counts(
                    tasks_in_progress=1,
                )
                task_start_time = time.time()
                progress_updater.update_progress()

                try:
                    if mw.progress.want_cancel() or cancel_state.is_cancelled():
                        logger.debug("Inner process op: cancellation requested")
                        return False

                    # If the op itself is async, run it directly; otherwise run the blocking call
                    # in the thread pool so the event loop is not blocked by HTTP requests.
                    # to_thread uses the loop's default executor, which selected_notes_op has
                    # pointed at the run's shared, bounded pool.
                    if asyncio.iscoroutinefunction(op):
                        op_result = await op(
                            config,
                            notes_to_add_dict=notes_to_add_dict,
                            notes_to_update_dict=notes_to_update_dict,
                            **op_args,
                        )
                    else:
                        op_result = await asyncio.to_thread(
                            op,
                            config,
                            notes_to_add_dict=notes_to_add_dict,
                            notes_to_update_dict=notes_to_update_dict,
                            **op_args,
                        )

                    if asyncio.iscoroutine(op_result):
                        op_result = await op_result

                except RunCancelled as e:
                    # The op asked the collection for something after the run was cancelled.
                    # Expected, not an error: the task is being abandoned on purpose, and
                    # whatever it had done so far is simply left unfinished.
                    logger.log(diagnostic_level(), "Op abandoned on cancellation: %s", e)
                    return False
                except Exception as e:
                    logger.error("Inner process op error, passing to handle_op_error: %s", e)
                    handle_op_error(e)
                    return False
                finally:
                    task_time = time.time() - task_start_time
                    # A task that only finishes well after the cancel is one that kept working
                    # regardless of it; how long it took to stop is the thing to measure
                    since_cancel = seconds_since_cancel()
                    if since_cancel is not None:
                        logger.log(
                            diagnostic_level(),
                            "[stage] op returned %.1fs after the cancel (ran %.1fs)",
                            since_cancel,
                            task_time,
                        )
                    progress_updater.increment_counts(
                        tasks_done=1,
                        tasks_in_progress=-1,
                        cumulative_task_time=task_time,
                        notes_done=1 if one_task_per_op else 0,
                    )
                    progress_updater.update_progress()
            finally:
                gate.release()

            # Handle results
            handle_op_result(op_result)
            return op_result

        except asyncio.CancelledError:
            return False

    return process_op


class NotePlan(NamedTuple):
    """One note's work, worked out before any of it has been started.

    `task_count` is how many API tasks the note will produce. Knowing it before the run starts
    is what lets the progress dialog show the real total from the beginning, instead of a total
    that climbs every time a window of notes is reached.

    `spawn` creates those tasks, appending them to the window's task list. It is called only
    when the note's window comes up, so the tasks themselves - which each hold a note and a
    prompt for as long as they live - still exist only a window at a time.
    """

    task_count: int
    spawn: Callable[[list[asyncio.Task]], None]


async def bulk_nested_notes_op(
    message: str,
    config: dict,
    bulk_inner_op: Callable[..., Optional[NotePlan]],
    col: Collection,
    notes: Sequence[Note],
    edited_nids: list[NoteId],
    progress_updater: AsyncTaskProgressUpdater,
    notes_to_add_dict: dict[str, list[Note]],
    notes_to_update_dict: dict[NoteId, Note],
    notes_to_remove: Optional[list[NoteId]] = None,
    model: str = "",
    on_end: Optional[Callable[..., None]] = None,
) -> tuple[int, dict[str, list[Note]], dict[NoteId, Note], list[NoteId]]:
    """
    Perform a bulk operation on a sequence of notes, with multiple nested async operations occurring
    per note instead of just one. Otherwise similar to `bulk_notes_op` except this cannot be
    performed synchronously and thus, requires rate limits to be set in the config.


    :param message: A message to display in the progress dialog.
    :param config: Addon config dict.
    :param bulk_inner_op: The nested operation function to apply to each note. It is called once
           per note up front and must not start any work itself: it returns a NotePlan saying how
           many tasks the note will produce and how to create them, or None if the note has
           nothing to do. This op itself handles calling inner_bulk_op and updating updated_notes
           and edited_nids.
    :param col: The Anki collection object.
    :param notes: A sequence of Note objects to process.
    :param edited_nids: A list to store the IDs of edited notes, to be mutated in place.
    :param model: The AI model to use for the operation.
    :param on_end: An optional callback to run on completion of the bulk op. Should be running other
           side effects that do not edit or add notes as those should be handled through
    """
    pos = col.add_custom_undo_entry(f"{message} for {len(notes)} notes.")
    if notes_to_remove is None:
        notes_to_remove = []
    if not model:
        logger.error("Model arg missing in bulk_nested_notes_op, aborting")
        return pos, notes_to_add_dict, notes_to_update_dict, notes_to_remove

    progress_updater.set_total_notes(len(notes))

    # The message doubles as the op's identity for the learned per-task memory cost. The pool
    # is sized to the gate's ceiling, which is a guess until the op has been measured - so the
    # gate says when it moves it and the pool follows, rather than staying at the guess.
    gate = ConcurrencyGate(config, op_key=message, on_ceiling_changed=set_connection_pool_size)
    progress_updater.gate = gate
    gate.start_adapting()
    set_connection_pool_size(gate.max_limit)
    rate_limit_tracker.reset()

    cancel_state = CancelState()
    cancel_manager: Optional[CancelManager] = None

    # Work out what every note needs doing before starting any of it. This pass is synchronous
    # and creates no tasks - it only reads the notes, which are in memory already - so it costs
    # nothing in concurrency, and it is what gives the dialog the run's real task total from the
    # start. A note here can fan out to anywhere between one and dozens of API calls, so a count
    # of notes on its own says very little about how much work is left.
    plan_started = time.monotonic()
    plans: list[NotePlan] = []
    planned_tasks = 0
    for note_index, note in enumerate(notes):
        if mw.progress.want_cancel():
            logger.debug("Nested bulk op cancelled while planning")
            cancel_state.cancel()
            break
        plan = bulk_inner_op(
            config,
            note,
            edited_nids=edited_nids,
            notes_to_add_dict=notes_to_add_dict,
            notes_to_update_dict=notes_to_update_dict,
            progress_updater=progress_updater,
            cancel_state=cancel_state,
            gate=gate,
        )
        if plan is not None:
            plans.append(plan)
            planned_tasks += plan.task_count
        progress_updater.set_total_tasks(planned_tasks)
        progress_updater.update_preparation_progress(
            notes_prepared=note_index + 1,
            total_notes=len(notes),
            tasks_planned=planned_tasks,
        )
    log_phase("nested op: plan notes", plan_started, notes=len(notes), tasks=planned_tasks)

    # Only now that the total is known: until this point the periodic updater would be drawing
    # a task line whose total is still growing, over the preparation line
    progress_updater.start_autoupdate()

    try:
        # Every task holds onto its note, prompt and config for as long as it lives. Creating
        # them all up front is what runs the machine out of memory, so notes are processed in
        # windows instead: only the current window's tasks exist at once. The window is sized by
        # the tasks the notes in it will produce rather than by note count, since that is what
        # decides the memory - a few times the gate limit, so tasks queue behind the gate rather
        # than the run stalling between windows. Shared per-run state (word locks, generated
        # meanings) lives in the caller's closure and carries across windows.
        task_budget = max(1, gate.limit * TASK_QUEUE_DEPTH)
        index = 0
        while index < len(plans):
            if mw.progress.want_cancel() or cancel_state.is_cancelled():
                break
            window: list[NotePlan] = []
            window_task_count = 0
            # Always take at least one note, however many tasks it turns out to want
            while index < len(plans) and (not window or window_task_count < task_budget):
                window.append(plans[index])
                # A plan with no API tasks of its own still creates the bookkeeping tasks that
                # write its note back, so it cannot count as free: a run of them would never
                # move the budget and every note would land in one window.
                window_task_count += max(1, plans[index].task_count)
                index += 1

            # Nothing is in flight here, so this is a clean baseline to measure the window's
            # concurrent memory use against
            gate.begin_window()

            tasks: list[asyncio.Task] = []
            window_api_tasks = 0
            for note_plan in window:
                if mw.progress.want_cancel():
                    break
                note_plan.spawn(tasks)
                window_api_tasks += note_plan.task_count
            if not tasks:
                continue
            # The API tasks are alive from here, whether or not they hold a gate slot yet, and
            # each holds its note and prompt. That count is what the window's memory growth has
            # to be divided by to get what one task costs. Not len(tasks): spawn() also creates
            # a per-word-list and a per-note bookkeeping task, which hold no prompt and would
            # only dilute the average - and task_count is the unit the budget above is spent
            # in, so both halves of the memory arithmetic stay in the same one.
            gate.note_window_tasks(window_api_tasks)
            progress_updater.update_progress()

            cancel_manager = CancelManager(
                tasks, cancel_state=cancel_state, progress_updater=progress_updater
            )
            try:
                await await_tasks_or_cancel(tasks, cancel_manager)
            except asyncio.CancelledError:
                logger.debug("Cancelling bulk operation")
            finally:
                if not cancel_manager.monitor_task.done():
                    cancel_manager.monitor_task.cancel()

            if cancel_manager.is_cancel_requested():
                logger.debug("Bulk operation was cancelled, returning results so far")
                marker = time.monotonic()
                gate.abort()
                log_phase("nested op: gate.abort", marker)
                break

            # Fold this window into what we know an op of this kind costs, which may move the
            # ceiling; the gate may also have resized under memory pressure while it ran
            gate.end_window()
            task_budget = max(1, gate.limit * TASK_QUEUE_DEPTH)
    finally:
        marker = time.monotonic()
        gate.finish()
        marker = log_phase("nested op: gate.finish", marker)
        progress_updater.gate = None

    if on_end:
        on_end()
        marker = log_phase("nested op: on_end", marker)
    progress_updater.stop_autoupdate()
    log_phase("nested op: stop_autoupdate", marker, threads=threading.active_count())
    return pos, notes_to_add_dict, notes_to_update_dict, notes_to_remove


def sync_bulk_notes_op(
    pos: int,
    col: Collection,
    config: dict,
    op: Callable[..., bool],
    notes: Sequence[Note],
    edited_nids: list[NoteId],
    message: str,
    notes_to_add_dict: Optional[dict[str, list[Note]]] = None,
    notes_to_update_dict: Optional[dict[NoteId, Note]] = None,
    notes_to_remove: Optional[list[NoteId]] = None,
    on_end: Optional[Callable[..., None]] = None,
):
    """
    Perform a simple sync bulk operation on a sequence of notes. Will run the operation
    function on each note, updating the progress dialog and collecting edited note IDs.

    Used as a fallback for when the async version is not needed or rate limits are not set.

    :param pos: The position in the undo stack to add the operation.
    :param col: The Anki collection object.
    :param config: Addon config dict.
    :param op: The operation function to apply to each note.
    :param col: The Anki collection object.
    :param notes: A sequence of Note objects to process.
    :param edited_nids: A list to store the IDs of edited notes, to be mutated in place.
    :param message: A message to display in the progress dialog.
    :param on_end: An optional callback to run on completion of the bulk op. Should be running other
            side effects that do not edit or add notes as those should be handled through
            notes_to_add_dict and notes_to_update_dict.
    """
    total_notes = len(notes)
    if notes_to_remove is None:
        notes_to_remove = []
    note_cnt = 0
    start_time = time.time()
    for note in notes:
        try:
            op(
                config=config,
                note=note,
                notes_to_add_dict=notes_to_add_dict,
                notes_to_update_dict=notes_to_update_dict,
            )
        except Exception as e:
            logger.error("Sync bulk notes op: Error processing note %s: %s", note.id, e)
        note_cnt += 1

        elapsed_s = time.time() - start_time
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_s))
        time_msg = f"<br><code>Time: {elapsed_time}</code>"
        if note_cnt > 3:
            eta_s = (total_notes - note_cnt) * (elapsed_s / note_cnt)
            eta_time = time.strftime("%H:%M:%S", time.gmtime(eta_s))
            time_msg += f"""<br><code>ETA: {eta_time}</code>"""
        mw.taskman.run_on_main(
            lambda: mw.progress.update(
                label=f"<b>{message}</b><br>{note_cnt}/{total_notes} notes processed{time_msg}",
                value=note_cnt,
                max=total_notes,
            )
        )
        if mw.progress.want_cancel():
            break

    if on_end:
        on_end()

    mw.taskman.run_on_main(lambda: mw.progress.finish())

    return pos, notes_to_add_dict, notes_to_update_dict, notes_to_remove


BulkOpResult = tuple[int, dict[str, list[Note]], dict[NoteId, Note], list[NoteId]]


async def bulk_notes_op(
    message,
    config,
    op,
    col: Collection,
    notes: Sequence[Note],
    edited_nids: list[NoteId],
    progress_updater: AsyncTaskProgressUpdater,
    notes_to_add_dict: Optional[dict[str, list[Note]]] = None,
    notes_to_update_dict: Optional[dict[NoteId, Note]] = None,
    notes_to_remove: Optional[list[NoteId]] = None,
    on_end: Optional[Callable[..., None]] = None,
    is_sync_op: bool = False,
) -> BulkOpResult:
    """
    Perform a simple async or sync bulk operation on a sequence of notes. Will run the operation
    function on each note, updating the progress dialog and collecting edited note IDs.
    Each note will create one async task.

    The bulk op runs asynchronously unless is_sync_op is set. How many notes are processed at
    once is decided by the shared ConcurrencyGate, based on available memory, not by a
    configured request rate — the providers throttle themselves against their own rate-limit
    responses.

    Args:
        message: A message to display in the progress dialog.
        config: Addon config dict.
        op: The operation function to apply to each note.
        col: The Anki collection object.
        notes: A sequence of Note objects to process.
        edited_nids: A list to store the IDs of edited notes, to be mutated in place.
        is_sync_op: Run the notes sequentially instead of concurrently. For local ops that
            make no API calls.
        on_end: An optional callback to run on completion of the bulk op. Should be running other
            side effects that do not edit or add notes as those should be handled through
            notes_to_add_dict and notes_to_update_dict.
    """
    if notes_to_remove is None:
        notes_to_remove = []
    if notes_to_add_dict is None:
        notes_to_add_dict = {}
    if notes_to_update_dict is None:
        notes_to_update_dict = {}
    pos = col.add_custom_undo_entry(f"{message} for {len(notes)} notes.")
    if is_sync_op:
        return sync_bulk_notes_op(
            pos=pos,
            col=col,
            config=config,
            op=op,
            notes=notes,
            edited_nids=edited_nids,
            message=message,
            notes_to_add_dict=notes_to_add_dict,
            notes_to_update_dict=notes_to_update_dict,
            notes_to_remove=notes_to_remove,
            on_end=on_end,
        )

    updated_notes: list[Note] = []

    progress_updater.set_total_notes(len(notes))
    progress_updater.set_total_tasks(len(notes))
    # Can start auto updater now that we're in an async context with a running loop
    progress_updater.start_autoupdate()

    # The message doubles as the op's identity for the learned per-task memory cost. The pool
    # is sized to the gate's ceiling, which is a guess until the op has been measured - so the
    # gate says when it moves it and the pool follows, rather than staying at the guess.
    gate = ConcurrencyGate(config, op_key=message, on_ceiling_changed=set_connection_pool_size)
    progress_updater.gate = gate
    gate.start_adapting()
    set_connection_pool_size(gate.max_limit)
    rate_limit_tracker.reset()

    def handle_op_success(
        note: Note,
        was_success: bool,
    ):
        """Handle successful operation result."""
        if was_success and edited_nids is not None:
            updated_notes.append(note)
            edited_nids.append(note.id)
        logger.debug(f"Bulk notes op success for note {note.id}, was_success: {was_success}")

    cancel_state = CancelState()
    cancel_manager: Optional[CancelManager] = None
    cancelled = False

    try:
        # Notes are processed in windows so only the current window's tasks exist at once;
        # a task holds onto its note and prompt for as long as it lives, so creating one per
        # note up front is what runs a smaller machine out of memory. The window is a few
        # times the gate limit so tasks queue behind it rather than the run stalling between
        # windows.
        window_size = max(1, gate.limit * TASK_QUEUE_DEPTH)
        index = 0
        while index < len(notes):
            if mw.progress.want_cancel() or cancel_state.is_cancelled():
                logger.debug("Bulk notes op cancelled before starting tasks")
                cancelled = True
                break
            window = notes[index : index + window_size]
            index += window_size

            # Nothing is in flight here, so this is a clean baseline to measure the window's
            # concurrent memory use against
            gate.begin_window()

            tasks: list[asyncio.Task] = []
            for note in window:

                def handle_error(current_note, e):
                    logger.error(f"Error during operation with note {current_note.id}: {e}")
                    print_error_traceback(e, logger)

                handle_op_error = partial(
                    lambda current_note, e: handle_error(current_note, e),
                    note,
                )

                handle_op_result = partial(
                    lambda current_note, was_success: handle_op_success(current_note, was_success),
                    note,
                )
                process_note = make_inner_bulk_op(
                    config=config,
                    op=op,
                    gate=gate,
                    progress_updater=progress_updater,
                    handle_op_error=handle_op_error,
                    handle_op_result=handle_op_result,
                    cancel_state=cancel_state,
                    one_task_per_op=True,
                )
                if mw.progress.want_cancel():
                    logger.debug("Bulk notes op cancelled before starting tasks")
                    break
                tasks.append(
                    asyncio.create_task(
                        process_note(
                            notes_to_add_dict=notes_to_add_dict,
                            notes_to_update_dict=notes_to_update_dict,
                            # note is passed to the op function, along with config in
                            # make_inner_bulk_op
                            note=note,
                        )
                    )
                )
            if not tasks:
                continue
            # All of them are alive from here, whether or not they hold a gate slot yet, and
            # each holds its note and prompt. That count is what the window's memory growth
            # has to be divided by to get what one task costs.
            gate.note_window_tasks(len(tasks))
            progress_updater.update_progress()

            cancel_manager = CancelManager(
                tasks, cancel_state, progress_updater=progress_updater
            )
            try:
                logger.debug("Bulk notes op awaiting %d tasks", len(tasks))
                await await_tasks_or_cancel(tasks, cancel_manager)
            except asyncio.CancelledError:
                logger.debug("Bulk notes op asyncio.CancelledError caught")
                cancel_manager.request_cancel()
            finally:
                if not cancel_manager.monitor_task.done():
                    cancel_manager.monitor_task.cancel()

            if cancel_manager.is_cancel_requested():
                logger.debug("Bulk notes op cancellation requested, returning early")
                marker = time.monotonic()
                gate.abort()
                log_phase("bulk op: gate.abort", marker)
                cancelled = True
                break

            # Fold this window into what we know an op of this kind costs, which may move the
            # ceiling; the gate may also have resized under memory pressure while it ran
            gate.end_window()
            window_size = max(1, gate.limit * TASK_QUEUE_DEPTH)
    finally:
        marker = time.monotonic()
        gate.finish()
        marker = log_phase("bulk op: gate.finish", marker)
        progress_updater.gate = None

    if not cancelled:
        logger.debug("Bulk notes op completed successfully, updating notes")

    if on_end:
        on_end()
        marker = log_phase("bulk op: on_end", marker)

    progress_updater.stop_autoupdate()
    log_phase(
        "bulk op: stop_autoupdate",
        marker,
        cancelled=cancelled,
        to_update=len(notes_to_update_dict),
        to_add=sum(len(v) for v in notes_to_add_dict.values()),
        threads=threading.active_count(),
    )
    return pos, notes_to_add_dict, notes_to_update_dict, notes_to_remove


def on_bulk_success(
    out,
    done_text: str,
    edited_nids: Sequence[NoteId],
    edited_other_nids: Sequence[NoteId],
    nids: Sequence[NoteId],
    parent: Browser,
    # notes_to_add_dict: Optional[dict[str, list[Note]]] = None,
    extra_callback=None,
):
    success_started = time.monotonic()
    logger.debug("[phase] on_bulk_success reached, closing progress")
    mw.taskman.run_on_main(lambda: mw.progress.finish())
    # if DEBUG:
    # print("on_bulk_success", out, notes_to_add_dict)
    if extra_callback:
        extra_callback()
        log_phase("success: extra_callback", success_started)
    # if notes_to_add_dict:
    #     new_notes: list[Note] = []
    #     for note_list in notes_to_add_dict.values():
    #         new_notes.extend(note_list)
    #     if new_notes:
    #         new_notes_tsv_str = make_tsv_from_notes(
    #             notes=new_notes,
    #             config=mw.addonManager.getConfig(__name__) or {},
    #         )
    #         if new_notes_tsv_str:
    #             # Write the TSV to the media folder
    #             import_tsv_file(
    #                 "new_notes.tsv",
    #                 new_notes_tsv_str,
    #             )
    # Show a tooltip after the import call as otherwise the import dialog would close the tooltip
    # immediately after it had appeared
    message = f"{done_text} in {len(edited_nids)}/{len(nids)} selected notes."
    if edited_other_nids:
        message += f"<br>Edited {len(edited_other_nids)} other notes not among the selection."
    tooltip(
        message,
        parent=parent,
        period=5000,
    )


NewNotesOp = Callable[[list[Note], dict, AsyncTaskProgressUpdater], dict[NoteId, Note]]
FilterNewNotesOp = Callable[
    [list[Note], dict, AsyncTaskProgressUpdater],
    tuple[list[Note], dict[NoteId, Note]],
]


def selected_notes_op(
    done_text: str,
    bulk_op: Callable[..., Coroutine[Any, Any, BulkOpResult]],
    nids: Sequence[NoteId],
    parent: Browser,
    progress_updater: AsyncTaskProgressUpdater,
    new_notes_op: Optional[NewNotesOp] = None,
    filter_new_notes_op: Optional[FilterNewNotesOp] = None,
    on_success: Optional[Callable] = None,
):
    edited_nids: list[NoteId] = []
    edited_other_nids: list[NoteId] = []
    notes_to_add_dict: dict[str, list[Note]] = {}
    notes_to_update_dict: dict[NoteId, Note] = {}
    notes_to_remove: set[NoteId] = set()
    config = mw.addonManager.getConfig(__name__) or {}
    nids_set = set(nids)

    # Create a wrapper function that handles the async operation
    def run_bulk_op(col: Collection) -> OpChanges:
        # Every operation enters here, which makes this the only place that can promise a run
        # starts uncancelled. bulk_notes_op and bulk_nested_notes_op used to do the clearing,
        # but an op is free to read the collection before it gets that far - the single-word
        # match ops search out the notes to work on first - and those reads go through
        # collection_access, which refuses while the previous, cancelled run's flag is still
        # set. That turned "cancel a run, then start another" into a RunCancelled traceback out
        # of the new operation before it had done anything.
        run = begin_run()
        # And the same for the cancel marker the diagnostics time everything against: a run
        # that starts after a cancelled one must not report its tasks as having returned
        # minutes after a cancel that belongs to the previous run.
        clear_cancel_time()

        async def async_wrapper():
            nonlocal edited_nids, edited_other_nids
            result = await bulk_op(
                col,
                notes=[mw.col.get_note(nid) for nid in nids],
                edited_nids=edited_nids,
                progress_updater=progress_updater,
                notes_to_add_dict=notes_to_add_dict,
                notes_to_update_dict=notes_to_update_dict,
            )
            cleanup_started = time.monotonic()
            logger.debug("[phase] bulk op returned, starting cleanup")
            # From here on this thread is saving what the run managed to do, which is the whole
            # point of cancelling gracefully - so it keeps its access to the collection even
            # though the run is cancelled. Some ops have real work left here, such as resolving
            # the ids of the notes they added. Cleared in run_bulk_op's finally.
            begin_cleanup_phase()
            pos, res_notes_to_add_dict, res_notes_to_update_dict, res_notes_to_remove = result

            sanitized_notes_to_remove: list[NoteId] = []
            if isinstance(res_notes_to_remove, str):
                logger.error("Invalid notes_to_remove payload type str: %s", res_notes_to_remove)
            else:
                try:
                    for nid in res_notes_to_remove:
                        if isinstance(nid, int):
                            sanitized_notes_to_remove.append(nid)
                        elif isinstance(nid, str) and nid.isdigit():
                            sanitized_notes_to_remove.append(NoteId(int(nid)))
                        else:
                            logger.error(
                                "Skipping invalid notes_to_remove nid type=%s value=%s",
                                type(nid),
                                nid,
                            )
                except TypeError:
                    logger.error(
                        "Invalid notes_to_remove payload non-iterable: %s",
                        res_notes_to_remove,
                    )

            # A cancelled run leaves its requests running in worker threads, and one of them
            # can still be writing into these dicts while we work through them here. Take a
            # snapshot so the cleanup sees a consistent set and can't trip over a dict that
            # changed size mid-iteration.
            res_notes_to_add_dict = dict(res_notes_to_add_dict)
            res_notes_to_update_dict = dict(res_notes_to_update_dict)

            logger.debug(f"res_notes_to_update_dict keys: {res_notes_to_update_dict.keys()}")
            notes_to_add_dict.update(res_notes_to_add_dict)
            notes_to_update_dict.update(res_notes_to_update_dict)
            notes_to_remove.update(sanitized_notes_to_remove)
            logger.debug(f"notes_to_update_dict keys: {notes_to_update_dict.keys()}")
            for nid in res_notes_to_update_dict.keys():
                if nid not in nids_set:
                    edited_other_nids.append(nid)
            for nid in sanitized_notes_to_remove:
                if nid not in nids_set:
                    edited_other_nids.append(nid)
            edited_nids = list(filter(lambda x: x in nids, notes_to_update_dict.keys()))
            edited_nids.extend(
                [nid for nid in sanitized_notes_to_remove if nid in nids and nid not in edited_nids]
            )

            if notes_to_remove:
                for removed_nid in notes_to_remove:
                    if removed_nid in notes_to_update_dict:
                        del notes_to_update_dict[removed_nid]

            # Remove note.id=0 notes from updated_notes
            all_updated_notes_dict: dict[NoteId, Note] = {}
            for note in list(notes_to_update_dict.values()):
                if note.id == 0:
                    logger.error(f"Found note.id=0, fields: {note.fields}")
                elif note.id not in all_updated_notes_dict:
                    all_updated_notes_dict[note.id] = note
                else:
                    # If the note is already in notes_to_update_dict, this might be a problem in the
                    # logic of the bulk op
                    logger.warning(
                        f"Note {note.id} occurring multiple times in notes_to_update_dict during"
                        " bulk op final update"
                    )
            all_updated_notes = [n for n in all_updated_notes_dict.values() if n.id != 0]
            cleanup_started = log_phase(
                "cleanup: collect notes", cleanup_started, notes=len(all_updated_notes)
            )
            # This write has been the visible symptom of every cancellation hang so far, taking
            # minutes even with nothing to write. Record what the rest of the process is doing
            # on either side of it: an empty write cannot be slow by itself, so whatever is
            # holding it up is in these stacks.
            if run_cancelled():
                dump_thread_stacks("about to update_notes")
            try:
                mw.col.update_notes(all_updated_notes)
            except Exception as e:
                logger.error(f"Error updating notes: {e}")
                logger.error(f"Notes causing error: {[n.fields for n in all_updated_notes]}")
                print_error_traceback(e, logger)
            cleanup_started = log_phase("cleanup: update_notes", cleanup_started)
            if run_cancelled():
                dump_thread_stacks("finished update_notes")
            if notes_to_remove:
                try:
                    mw.col.remove_notes(sorted(notes_to_remove))
                except Exception as e:
                    logger.error(f"Error removing notes: {e}")
                    logger.error(f"Note IDs causing error: {sorted(notes_to_remove)}")
                    print_error_traceback(e, logger)
                cleanup_started = log_phase("cleanup: remove_notes", cleanup_started)
            op_changes = mw.col.merge_undo_entries(pos)
            notes_to_add = []
            if notes_to_add_dict:
                for note_list in list(notes_to_add_dict.values()):
                    notes_to_add.extend(list(note_list))
            cleanup_started = log_phase(
                "cleanup: merge_undo_entries", cleanup_started, to_add=len(notes_to_add)
            )

            if notes_to_add and filter_new_notes_op:
                notes_to_add, filtered_notes_to_update_dict = filter_new_notes_op(
                    notes_to_add,
                    config,
                    progress_updater,
                )
                valid_filtered_notes = [
                    note
                    for note in filtered_notes_to_update_dict.values()
                    if note.id != 0 and note.id not in notes_to_remove
                ]
                if valid_filtered_notes:
                    try:
                        mw.col.update_notes(valid_filtered_notes)
                    except Exception as e:
                        logger.error(f"Error updating notes after filter_new_notes_op: {e}")
                        print_error_traceback(e, logger)
                    op_changes = mw.col.merge_undo_entries(pos)
                    for note in valid_filtered_notes:
                        if note.id in nids_set:
                            if note.id not in edited_nids:
                                edited_nids.append(note.id)
                        elif note.id not in edited_other_nids:
                            edited_other_nids.append(note.id)
                cleanup_started = log_phase(
                    "cleanup: filter_new_notes_op", cleanup_started, kept=len(notes_to_add)
                )

            if notes_to_add:
                logger.debug(
                    f"Adding {len(notes_to_add)} new notes to note_will_be_added hooks will be run"
                )
                total_notes = len(notes_to_add)
                failed_cnt = 0
                added_cnt = 0
                for index, note in enumerate(notes_to_add):
                    note_type = note.note_type()
                    if note_type is None:
                        logger.debug(
                            f"Error: Note type for note {note.id} is None, skipping note adding"
                        )
                        continue
                    insert_deck = get_field_config(config, "insert_deck", note_type)
                    insert_deck_id = None
                    if insert_deck:
                        insert_deck_id = mw.col.decks.id_for_name(insert_deck)
                    else:
                        insert_deck_id = mw.col.decks.id_for_name("Default")
                        logger.debug("No insert deck set, setting deck_id to Default")
                    if insert_deck_id is None:
                        logger.debug("Default deck not found, skipping note adding")
                        continue
                    if mw.progress.want_cancel():
                        logger.debug("Bulk notes op cancelled during note adding")
                        break
                    try:
                        logger.debug(f"Adding note {index} to deck {insert_deck_id}")
                        mw.col.add_note(note, insert_deck_id)
                        added_cnt += 1
                        op_changes = mw.col.merge_undo_entries(pos)
                    except Exception as e:
                        logger.error(f"Error adding note {index}: {e}")
                        print_error_traceback(e, logger)
                        failed_cnt += 1

                    progress_updater.update_note_adding_progress(
                        notes_added=added_cnt,
                        total_notes=total_notes,
                        failed=failed_cnt,
                    )
                cleanup_started = log_phase(
                    "cleanup: add_note loop", cleanup_started, added=added_cnt, failed=failed_cnt
                )
                if new_notes_op:
                    # Run the new notes operation if provided
                    # col.add_note mutates the note given, adding the id to it
                    additional_updates_notes_dict = new_notes_op(
                        notes_to_add, config, progress_updater
                    )
                    cleanup_started = log_phase("cleanup: new_notes_op", cleanup_started)

                    additional_updated_notes = list(additional_updates_notes_dict.values())
                    if additional_updated_notes:
                        # Skip notes where the id is still zero, something went wrong during adding
                        valid_notes = []
                        invalid_notes = []
                        for note in additional_updated_notes:
                            if note.id == 0:
                                invalid_notes.append(note)
                            else:
                                valid_notes.append(note)
                        if invalid_notes:
                            logger.debug(f"Invalid notes found after adding: {len(invalid_notes)}")
                            new_notes_tsv_str = make_tsv_from_notes(
                                notes=invalid_notes,
                                config=mw.addonManager.getConfig(__name__) or {},
                            )
                            if new_notes_tsv_str:
                                # Write the TSV to the media folder
                                import_tsv_file(
                                    "new_notes.tsv",
                                    new_notes_tsv_str,
                                    do_import=False,
                                )
                        try:
                            mw.col.update_notes(valid_notes)
                        except Exception as e:
                            logger.error(f"Error updating valid notes after new_notes_op: {e}")
                            print_error_traceback(e, logger)
                        op_changes = mw.col.merge_undo_entries(pos)
                        edited_nids.extend(
                            [note.id for note in valid_notes if note.id not in edited_nids]
                        )
            log_phase("cleanup: finished", cleanup_started, threads=threading.active_count())
            return op_changes

        # Create and run the event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # One bounded thread pool for the whole operation, shared by make_inner_bulk_op and
        # every asyncio.to_thread call beneath it. Previously each task built its own pool and
        # never shut it down, so threads accumulated for the life of the process. Sized to the
        # highest limit the gate could reach, since it may raise its ceiling mid-run once it
        # has measured what the op costs; unused threads are never spawned.
        executor = ThreadPoolExecutor(
            max_workers=max_possible_concurrency(config) + 4,
            thread_name_prefix="simple_anki_ai_prompts",
            # Every worker takes part in this run, so cancelling it stops their requests and
            # collection reads too - including in the threads the run is abandoning, which
            # keep the enrolment for as long as they are alive.
            initializer=partial(join_run, run),
        )
        loop.set_default_executor(executor)
        try:
            return loop.run_until_complete(async_wrapper())
        except RunCancelled as e:
            # A cancel that landed on a collection read this thread makes outside the cleanup
            # phase, so the op unwound before it could save anything. There is nothing left to
            # write, but being cancelled is a normal outcome rather than an error: end the
            # operation quietly instead of showing the user a traceback.
            logger.info("Bulk op abandoned after cancellation: %s", e)
            return OpChanges()
        finally:
            teardown_started = time.monotonic()
            logger.debug(
                "[phase] teardown starting, %d threads alive", threading.active_count()
            )
            # This thread goes back to Anki's pool and will run other operations, so its
            # exemption must not outlive this one
            end_cleanup_phase()
            # shutdown(wait=False) tells the pool's threads to exit once their current call
            # returns, without blocking on them. Never join them here: a blocking HTTP request
            # cannot be interrupted from outside, so joining would make cancelling take as
            # long as the slowest request still in flight. Their results are discarded.
            executor.shutdown(wait=False)
            teardown_started = log_phase("teardown: executor.shutdown", teardown_started)
            # Tasks abandoned by a cancellation are still pending; drop them so closing the
            # loop doesn't complain about them
            leftover = [t for t in asyncio.all_tasks(loop) if not t.done()]
            for task in leftover:
                task.cancel()
            teardown_started = log_phase(
                "teardown: cancel leftover tasks", teardown_started, leftover=len(leftover)
            )
            loop.close()
            teardown_started = log_phase("teardown: loop.close", teardown_started)
            close_all_sessions()
            log_phase(
                "teardown: close_all_sessions",
                teardown_started,
                threads=threading.active_count(),
            )
            # Last, so everything above still logs as part of the run it belongs to. This
            # thread is Anki's and goes back to a pool that runs other work, including our own
            # single-note ops, so its membership of this run must not outlive it: leaving it
            # enrolled in a cancelled run is what used to make every later op the editor hooks
            # run - a story or a translation on field unfocus - quietly do nothing for the
            # rest of the session. The run's own worker threads stay enrolled; they are the
            # ones that must keep seeing the cancellation.
            end_run()

    return (
        CollectionOp(
            parent=parent,
            op=run_bulk_op,
        )
        .success(
            lambda out: on_bulk_success(
                out,
                done_text,
                edited_nids,
                edited_other_nids,
                nids,
                parent,
                # notes_to_add_dict,
                on_success,
            )
        )
        .run_in_background()
    )
