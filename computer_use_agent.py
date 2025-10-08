from __future__ import annotations

import logging
import os
import time
from typing import Dict, Iterable, List, Mapping, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.types import Content, Part
from playwright.sync_api import Page, sync_playwright

# Screen dimensions expected by the Gemini Computer Use tool.
SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 900
DEFAULT_MODEL = "gemini-2.5-computer-use-preview-10-2025"
DEFAULT_START_URL = "https://finance.yahoo.com/"
MAX_TURNS = 5
MAX_TYPABLE_CHARS = 1024
PAGE_SETTLE_DELAY_SECONDS = 1.0
LOAD_STATE_TIMEOUT_MS = 5_000
NAVIGATION_TIMEOUT_MS = 30_000
SAFE_ACTIONS = frozenset({"open_web_browser", "click_at", "type_text_at"})

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure a basic logger once per process."""
    if LOGGER.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def create_client() -> genai.Client:
    """Instantiate the Gemini client after loading environment variables."""
    load_dotenv(override=False)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Provide it via the environment or .env file."
        )
    return genai.Client(api_key=api_key)


def denormalize_coordinate(value: float, dimension: int) -> int:
    """Convert the normalized coordinate (0-1000) into a screen pixel location."""
    clamped = max(0.0, min(1000.0, float(value)))
    return int(clamped / 1000.0 * dimension)


def sanitize_text(text: object) -> str:
    """Limit generated text length and strip non-printable characters."""
    raw = str(text) if text is not None else ""
    printable_chars = (char for char in raw if char.isprintable() or char in ("\n", "\r", "\t"))
    return "".join(printable_chars)[:MAX_TYPABLE_CHARS]


def collect_function_calls(candidate: types.Candidate) -> List[types.FunctionCall]:
    """Extract function calls from the model candidate, if any."""
    calls: List[types.FunctionCall] = []
    content = getattr(candidate, "content", None)
    if not content:
        return calls
    for part in getattr(content, "parts", []) or []:
        function_call = getattr(part, "function_call", None)
        if function_call:
            calls.append(function_call)
    return calls


def execute_function_calls(
    candidate: types.Candidate,
    page: Page,
    screen_width: int,
    screen_height: int,
) -> List[Tuple[str, Dict[str, str]]]:
    """
    Execute tool calls suggested by the model while validating inputs.

    Returns a list containing the function name along with execution metadata.
    """
    results: List[Tuple[str, Dict[str, str]]] = []
    for function_call in collect_function_calls(candidate):
        fname = getattr(function_call, "name", "")
        args: Mapping[str, object] = getattr(function_call, "args", {}) or {}
        LOGGER.info("Executing tool action: %s", fname)

        if fname not in SAFE_ACTIONS:
            LOGGER.warning("Skipping unsupported function: %s", fname)
            results.append((fname, {"error": "unsupported_function"}))
            continue

        action_result: Dict[str, str] = {"status": "ok"}
        try:
            if fname == "open_web_browser":
                # Browser is already running; nothing to do.
                LOGGER.debug("Browser already active; no operation required.")
            elif fname == "click_at":
                x = denormalize_coordinate(args.get("x", 0), screen_width)
                y = denormalize_coordinate(args.get("y", 0), screen_height)
                page.mouse.click(x, y)
            elif fname == "type_text_at":
                x = denormalize_coordinate(args.get("x", 0), screen_width)
                y = denormalize_coordinate(args.get("y", 0), screen_height)
                text = sanitize_text(args.get("text", ""))
                press_enter = bool(args.get("press_enter", False))

                page.mouse.click(x, y)
                page.keyboard.press("Meta+A")
                page.keyboard.press("Backspace")
                page.keyboard.type(text)
                if press_enter:
                    page.keyboard.press("Enter")

            page.wait_for_load_state(timeout=LOAD_STATE_TIMEOUT_MS)
            time.sleep(PAGE_SETTLE_DELAY_SECONDS)
        except Exception as exc:  # Broad catch to keep the loop resilient.
            LOGGER.exception("Error executing %s: %s", fname, exc)
            action_result = {"error": str(exc)}

        results.append((fname, action_result))

    return results


def get_function_responses(
    page: Page,
    results: Iterable[Tuple[str, Dict[str, str]]],
) -> List[types.FunctionResponse]:
    """
    Capture the current page state for the tool responses.

    The screenshot gives the model visual context for the next turn.
    """
    try:
        screenshot_bytes = page.screenshot(type="png")
    except Exception as exc:
        LOGGER.warning("Unable to capture screenshot: %s", exc)
        screenshot_bytes = b""
    current_url = getattr(page, "url", "")

    function_responses: List[types.FunctionResponse] = []
    for name, result in results:
        response_payload = {"url": current_url}
        response_payload.update(result)
        function_responses.append(
            types.FunctionResponse(
                name=name,
                response=response_payload,
                parts=[
                    types.FunctionResponsePart(
                        inline_data=types.FunctionResponseBlob(
                            mime_type="image/png",
                            data=screenshot_bytes,
                        )
                    )
                ],
            )
        )
    return function_responses


def extract_text_response(parts: Iterable[Part]) -> str:
    """Collect plain-text responses from the model output."""
    snippets = [part.text.strip() for part in parts if getattr(part, "text", "").strip()]
    return " ".join(snippets)


def run_agent(user_prompt: str, *, headless: bool = False) -> None:
    """Run the Gemini Computer Use agent end-to-end."""
    configure_logging()
    client = create_client()

    config = types.GenerateContentConfig(
        tools=[
            types.Tool(
                computer_use=types.ComputerUse(
                    environment=types.Environment.ENVIRONMENT_BROWSER
                )
            )
        ]
    )

    LOGGER.info("Starting Playwright and launching browser (headless=%s)...", headless)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        try:
            with browser.new_context(
                viewport={"width": SCREEN_WIDTH, "height": SCREEN_HEIGHT}
            ) as context:
                page = context.new_page()
                LOGGER.info("Navigating to %s", DEFAULT_START_URL)
                page.goto(
                    DEFAULT_START_URL,
                    wait_until="domcontentloaded",
                    timeout=NAVIGATION_TIMEOUT_MS,
                )

                initial_screenshot = page.screenshot(type="png")
                contents: List[Content] = [
                    Content(
                        role="user",
                        parts=[
                            Part(text=user_prompt),
                            Part.from_bytes(data=initial_screenshot, mime_type="image/png"),
                        ],
                    )
                ]

                for turn in range(1, MAX_TURNS + 1):
                    LOGGER.info("--- Turn %s ---", turn)
                    try:
                        response = client.models.generate_content(
                            model=DEFAULT_MODEL,
                            contents=contents,
                            config=config,
                        )
                    except Exception as exc:
                        LOGGER.exception("Model request failed: %s", exc)
                        break

                    if not response.candidates:
                        LOGGER.warning("Model returned no candidates; stopping.")
                        break

                    candidate = response.candidates[0]
                    contents.append(candidate.content)

                    has_function_calls = any(
                        getattr(part, "function_call", None)
                        for part in getattr(candidate.content, "parts", []) or []
                    )
                    if not has_function_calls:
                        text_response = extract_text_response(
                            getattr(candidate.content, "parts", []) or []
                        )
                        LOGGER.info("Agent finished: %s", text_response)
                        break

                    results = execute_function_calls(
                        candidate,
                        page,
                        SCREEN_WIDTH,
                        SCREEN_HEIGHT,
                    )
                    function_responses = get_function_responses(page, results)
                    contents.append(
                        Content(
                            role="user",
                            parts=[Part(function_response=fr) for fr in function_responses],
                        )
                    )
                else:
                    LOGGER.info("Reached maximum number of turns (%s).", MAX_TURNS)
        finally:
            LOGGER.info("Closing browser.")
            browser.close()


def main() -> None:
    """Entry point for manual execution."""
    user_prompt = "Get the latest stock price of Nvidia and write the result."
    run_agent(user_prompt, headless=False)


if __name__ == "__main__":
    main()
