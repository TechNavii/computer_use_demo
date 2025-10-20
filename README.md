# Computer Use Agent Demo

This project automates a Playwright browser while using Gemini's Computer Use API to reason about actions and extract information from the web page.

## Setup

1. Create a virtual environment (optional but recommended) and install dependencies:
   ```bash
   pip install -r requirements.txt
   # or
   uv pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and provide your Gemini API key:
   ```bash
   cp .env.example .env
   echo GEMINI_API_KEY=your_actual_key >> .env
   ```
3. Playwright also needs browser binaries. Install them once:
   ```bash
   playwright install chromium
   ```

## Running

Execute the agent script:
```bash
python computer_use_agent.py
```

By default the script launches Chromium in headed mode to show the agent acting on the page. Set `headless=True` in the script if you prefer headless automation.

## Testing

This project includes comprehensive unit and integration tests.

### Running Tests

1. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

2. Run all tests:
   ```bash
   pytest
   ```

3. Run tests with coverage report:
   ```bash
   pytest --cov=computer_use_agent --cov-report=term-missing
   ```

4. Run specific test categories:
   ```bash
   # Run only unit tests
   pytest -m unit

   # Run only integration tests
   pytest -m integration

   # Run tests in a specific file
   pytest tests/test_utils.py
   ```

### Test Structure

- `tests/test_utils.py` - Tests for utility functions (coordinate normalization, text sanitization)
- `tests/test_function_calls.py` - Tests for function call extraction and response processing
- `tests/test_execution.py` - Tests for browser action execution
- `tests/test_config.py` - Tests for configuration and client creation
- `tests/test_integration.py` - End-to-end integration tests
- `tests/conftest.py` - Shared test fixtures

### Test Coverage

The test suite achieves 96%+ code coverage and includes:
- 64+ unit and integration tests
- Mocked browser and API interactions
- Edge case validation
- Error handling verification

