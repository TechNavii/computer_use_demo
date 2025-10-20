"""Pytest configuration and shared fixtures."""
import pytest
from unittest.mock import Mock


@pytest.fixture
def mock_page():
    """Create a mock Playwright page object."""
    page = Mock()
    page.url = "https://example.com"
    page.screenshot.return_value = b"fake_screenshot"
    page.mouse = Mock()
    page.keyboard = Mock()
    page.wait_for_load_state = Mock()
    return page


@pytest.fixture
def mock_browser():
    """Create a mock Playwright browser object."""
    browser = Mock()
    browser.close = Mock()
    return browser


@pytest.fixture
def mock_genai_client():
    """Create a mock Gemini API client."""
    client = Mock()
    return client


@pytest.fixture
def sample_function_call():
    """Create a sample function call for testing."""
    from unittest.mock import Mock
    from google.genai.types import FunctionCall

    function_call = Mock(spec=FunctionCall)
    function_call.name = "click_at"
    function_call.args = {"x": 500, "y": 500}
    return function_call


@pytest.fixture
def sample_candidate_with_text():
    """Create a sample candidate with text response."""
    from unittest.mock import Mock
    from google.genai.types import Candidate, Part

    part = Mock(spec=Part)
    part.text = "Sample response"
    part.function_call = None

    candidate = Mock(spec=Candidate)
    candidate.content = Mock()
    candidate.content.parts = [part]

    return candidate


@pytest.fixture
def sample_candidate_with_function_call(sample_function_call):
    """Create a sample candidate with function call."""
    from unittest.mock import Mock
    from google.genai.types import Candidate

    part = Mock()
    part.function_call = sample_function_call

    candidate = Mock(spec=Candidate)
    candidate.content = Mock()
    candidate.content.parts = [part]

    return candidate
