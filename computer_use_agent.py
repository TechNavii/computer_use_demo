import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.types import Content, Part
from playwright.sync_api import sync_playwright

# Load API credentials from .env so local runs stay configurable.
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set in the environment or .env file.")

# Initialize the Gemini client
client = genai.Client(api_key=api_key)

# Screen dimensions
SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 900

def denormalize_x(x: int, screen_width: int) -> int:
    """Convert normalized x coordinate (0-1000) to actual pixel coordinate."""
    return int(x / 1000 * screen_width)

def denormalize_y(y: int, screen_height: int) -> int:
    """Convert normalized y coordinate (0-1000) to actual pixel coordinate."""
    return int(y / 1000 * screen_height)

def execute_function_calls(candidate, page, screen_width, screen_height):
    """Execute the actions suggested by the model."""
    results = []
    function_calls = []
    
    for part in candidate.content.parts:
        if part.function_call:
            function_calls.append(part.function_call)

    for function_call in function_calls:
        action_result = {}
        fname = function_call.name
        args = function_call.args
        print(f"  -> Executing: {fname}")

        try:
            if fname == "open_web_browser":
                pass  # Already open
            elif fname == "click_at":
                actual_x = denormalize_x(args["x"], screen_width)
                actual_y = denormalize_y(args["y"], screen_height)
                page.mouse.click(actual_x, actual_y)
            elif fname == "type_text_at":
                actual_x = denormalize_x(args["x"], screen_width)
                actual_y = denormalize_y(args["y"], screen_height)
                text = args["text"]
                press_enter = args.get("press_enter", False)

                page.mouse.click(actual_x, actual_y)
                page.keyboard.press("Meta+A")
                page.keyboard.press("Backspace")
                page.keyboard.type(text)
                if press_enter:
                    page.keyboard.press("Enter")
            
            page.wait_for_load_state(timeout=5000)
            time.sleep(1)

        except Exception as e:
            print(f"Error executing {fname}: {e}")
            action_result = {"error": str(e)}

        results.append((fname, action_result))

    return results

def get_function_responses(page, results):
    """Capture screenshot and URL after actions."""
    screenshot_bytes = page.screenshot(type="png")
    current_url = page.url
    function_responses = []
    
    for name, result in results:
        response_data = {"url": current_url}
        response_data.update(result)
        function_responses.append(
            types.FunctionResponse(
                name=name,
                response=response_data,
                parts=[types.FunctionResponsePart(
                    inline_data=types.FunctionResponseBlob(
                        mime_type="image/png",
                        data=screenshot_bytes))
                ]
            )
        )
    return function_responses

# Main program
print("Initialising browser...")
playwright = sync_playwright().start()
browser = playwright.chromium.launch(headless=False)
context = browser.new_context(viewport={"width": SCREEN_WIDTH, "height": SCREEN_HEIGHT})
page = context.new_page()

try:
    # Go to initial page
    page.goto("https://finance.yahoo.com/")
    
    # Configure the model with Computer Use tool
    config = types.GenerateContentConfig(
        tools=[types.Tool(computer_use=types.ComputerUse(
            environment=types.Environment.ENVIRONMENT_BROWSER
        ))],
    )

    # Take initial screenshot
    initial_screenshot = page.screenshot(type="png")
    USER_PROMPT = """
    Get the latest stock price of Nvidia and write the result
    """
    print(f"Goal: {USER_PROMPT}")

    contents = [
        Content(role="user", parts=[
            Part(text=USER_PROMPT),
            Part.from_bytes(data=initial_screenshot, mime_type='image/png')
        ])
    ]

    # Agent loop - maximum 5 turns
    for i in range(5):
        print(f"\n--- Turn {i+1} ---")
        print("Thinking...")
        
        response = client.models.generate_content(
            model='gemini-2.5-computer-use-preview-10-2025',
            contents=contents,
            config=config,
        )

        candidate = response.candidates[0]
        contents.append(candidate.content)

        # Check if there are function calls to execute
        has_function_calls = any(part.function_call for part in candidate.content.parts)
        if not has_function_calls:
            text_response = " ".join([part.text for part in candidate.content.parts if part.text])
            print("Agent finished:", text_response)
            break

        print("Executing actions...")
        results = execute_function_calls(candidate, page, SCREEN_WIDTH, SCREEN_HEIGHT)

        print("Capturing state...")
        function_responses = get_function_responses(page, results)

        contents.append(
            Content(role="user", parts=[Part(function_response=fr) for fr in function_responses])
        )

finally:
    print("\nClosing browser...")
    browser.close()
    playwright.stop()
    print("Done!")
