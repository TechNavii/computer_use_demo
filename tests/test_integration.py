"""Integration tests for the computer use agent."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from computer_use_agent import run_agent, SCREEN_WIDTH, SCREEN_HEIGHT


@pytest.mark.integration
class TestRunAgent:
    """Integration tests for run_agent function."""

    @patch("computer_use_agent.sync_playwright")
    @patch("computer_use_agent.create_client")
    @patch("computer_use_agent.configure_logging")
    def test_run_agent_basic_flow(
        self, mock_configure_logging, mock_create_client, mock_playwright
    ):
        """Test basic agent flow with mocked dependencies."""
        # Setup mocks
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        # Mock browser and page
        mock_browser = Mock()
        mock_context = Mock()
        mock_page = Mock()
        mock_page.url = "https://example.com"
        mock_page.screenshot.return_value = b"fake_screenshot"

        # Setup context manager chain
        mock_context.new_page.return_value = mock_page
        mock_context.__enter__ = Mock(return_value=mock_context)
        mock_context.__exit__ = Mock(return_value=False)

        mock_browser.new_context.return_value = mock_context
        mock_browser.close = Mock()

        playwright_instance = Mock()
        playwright_instance.chromium.launch.return_value = mock_browser
        playwright_instance.__enter__ = Mock(return_value=playwright_instance)
        playwright_instance.__exit__ = Mock(return_value=False)

        mock_playwright.return_value = playwright_instance

        # Mock model response (text response, no function calls)
        mock_candidate = Mock()
        mock_candidate.content = Mock()
        mock_candidate.content.parts = [Mock(text="Task completed", function_call=None)]

        mock_response = Mock()
        mock_response.candidates = [mock_candidate]
        mock_client.models.generate_content.return_value = mock_response

        # Run agent
        run_agent("Test prompt", headless=True)

        # Verify logging was configured
        mock_configure_logging.assert_called_once()

        # Verify client was created
        mock_create_client.assert_called_once()

        # Verify browser was launched
        playwright_instance.chromium.launch.assert_called_once_with(headless=True)

        # Verify browser context was created with correct viewport
        mock_browser.new_context.assert_called_once_with(
            viewport={"width": SCREEN_WIDTH, "height": SCREEN_HEIGHT}
        )

        # Verify page navigation
        mock_page.goto.assert_called_once()

        # Verify browser was closed
        mock_browser.close.assert_called_once()

    @patch("computer_use_agent.sync_playwright")
    @patch("computer_use_agent.create_client")
    @patch("computer_use_agent.configure_logging")
    def test_run_agent_with_function_calls(
        self, mock_configure_logging, mock_create_client, mock_playwright
    ):
        """Test agent flow with function calls."""
        # Setup mocks
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        # Mock browser and page
        mock_browser = Mock()
        mock_context = Mock()
        mock_page = Mock()
        mock_page.url = "https://example.com"
        mock_page.screenshot.return_value = b"fake_screenshot"
        mock_page.mouse = Mock()
        mock_page.wait_for_load_state = Mock()

        # Setup context manager chain
        mock_context.new_page.return_value = mock_page
        mock_context.__enter__ = Mock(return_value=mock_context)
        mock_context.__exit__ = Mock(return_value=False)

        mock_browser.new_context.return_value = mock_context

        playwright_instance = Mock()
        playwright_instance.chromium.launch.return_value = mock_browser
        playwright_instance.__enter__ = Mock(return_value=playwright_instance)
        playwright_instance.__exit__ = Mock(return_value=False)

        mock_playwright.return_value = playwright_instance

        # Mock model responses
        # First response: function call
        function_call = Mock()
        function_call.name = "click_at"
        function_call.args = {"x": 500, "y": 500}

        first_part = Mock()
        first_part.function_call = function_call

        first_candidate = Mock()
        first_candidate.content = Mock()
        first_candidate.content.parts = [first_part]

        first_response = Mock()
        first_response.candidates = [first_candidate]

        # Second response: text completion
        second_part = Mock()
        second_part.text = "Done"
        second_part.function_call = None

        second_candidate = Mock()
        second_candidate.content = Mock()
        second_candidate.content.parts = [second_part]

        second_response = Mock()
        second_response.candidates = [second_candidate]

        # Setup generate_content to return different responses
        mock_client.models.generate_content.side_effect = [
            first_response,
            second_response,
        ]

        # Run agent
        with patch("computer_use_agent.time.sleep"):
            run_agent("Test prompt with actions", headless=True)

        # Verify function was executed
        mock_page.mouse.click.assert_called_once()

        # Verify model was called twice
        assert mock_client.models.generate_content.call_count == 2

    @patch("computer_use_agent.sync_playwright")
    @patch("computer_use_agent.create_client")
    @patch("computer_use_agent.configure_logging")
    def test_run_agent_max_turns(
        self, mock_configure_logging, mock_create_client, mock_playwright
    ):
        """Test that agent stops after maximum turns."""
        # Setup mocks
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        # Mock browser and page
        mock_browser = Mock()
        mock_context = Mock()
        mock_page = Mock()
        mock_page.url = "https://example.com"
        mock_page.screenshot.return_value = b"fake_screenshot"
        mock_page.mouse = Mock()
        mock_page.wait_for_load_state = Mock()

        # Setup context manager chain
        mock_context.new_page.return_value = mock_page
        mock_context.__enter__ = Mock(return_value=mock_context)
        mock_context.__exit__ = Mock(return_value=False)

        mock_browser.new_context.return_value = mock_context

        playwright_instance = Mock()
        playwright_instance.chromium.launch.return_value = mock_browser
        playwright_instance.__enter__ = Mock(return_value=playwright_instance)
        playwright_instance.__exit__ = Mock(return_value=False)

        mock_playwright.return_value = playwright_instance

        # Mock model to always return function calls (never complete)
        function_call = Mock()
        function_call.name = "click_at"
        function_call.args = {"x": 500, "y": 500}

        part = Mock()
        part.function_call = function_call

        candidate = Mock()
        candidate.content = Mock()
        candidate.content.parts = [part]

        response = Mock()
        response.candidates = [candidate]

        mock_client.models.generate_content.return_value = response

        # Run agent
        with patch("computer_use_agent.time.sleep"):
            run_agent("Test prompt", headless=True)

        # Verify model was called MAX_TURNS times (5)
        from computer_use_agent import MAX_TURNS

        assert mock_client.models.generate_content.call_count == MAX_TURNS

    @patch("computer_use_agent.sync_playwright")
    @patch("computer_use_agent.create_client")
    @patch("computer_use_agent.configure_logging")
    def test_run_agent_handles_api_error(
        self, mock_configure_logging, mock_create_client, mock_playwright
    ):
        """Test that agent handles API errors gracefully."""
        # Setup mocks
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        # Mock browser and page
        mock_browser = Mock()
        mock_context = Mock()
        mock_page = Mock()
        mock_page.screenshot.return_value = b"fake_screenshot"

        mock_context.new_page.return_value = mock_page
        mock_context.__enter__ = Mock(return_value=mock_context)
        mock_context.__exit__ = Mock(return_value=False)

        mock_browser.new_context.return_value = mock_context

        playwright_instance = Mock()
        playwright_instance.chromium.launch.return_value = mock_browser
        playwright_instance.__enter__ = Mock(return_value=playwright_instance)
        playwright_instance.__exit__ = Mock(return_value=False)

        mock_playwright.return_value = playwright_instance

        # Mock API to raise error
        mock_client.models.generate_content.side_effect = Exception("API Error")

        # Should not raise exception
        run_agent("Test prompt", headless=True)

        # Verify browser was still closed
        mock_browser.close.assert_called_once()

    @patch("computer_use_agent.sync_playwright")
    @patch("computer_use_agent.create_client")
    @patch("computer_use_agent.configure_logging")
    def test_run_agent_headless_parameter(
        self, mock_configure_logging, mock_create_client, mock_playwright
    ):
        """Test that headless parameter is passed correctly."""
        # Setup minimal mocks
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        mock_browser = Mock()
        mock_context = Mock()
        mock_page = Mock()
        mock_page.screenshot.return_value = b"fake_screenshot"

        mock_context.new_page.return_value = mock_page
        mock_context.__enter__ = Mock(return_value=mock_context)
        mock_context.__exit__ = Mock(return_value=False)

        mock_browser.new_context.return_value = mock_context

        playwright_instance = Mock()
        playwright_instance.chromium.launch.return_value = mock_browser
        playwright_instance.__enter__ = Mock(return_value=playwright_instance)
        playwright_instance.__exit__ = Mock(return_value=False)

        mock_playwright.return_value = playwright_instance

        # Mock simple text response
        mock_candidate = Mock()
        mock_candidate.content = Mock()
        mock_candidate.content.parts = [Mock(text="Done", function_call=None)]

        mock_response = Mock()
        mock_response.candidates = [mock_candidate]
        mock_client.models.generate_content.return_value = mock_response

        # Test with headless=True
        run_agent("Test", headless=True)
        playwright_instance.chromium.launch.assert_called_with(headless=True)

        # Test with headless=False
        run_agent("Test", headless=False)
        playwright_instance.chromium.launch.assert_called_with(headless=False)
