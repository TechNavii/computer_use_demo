"""Unit tests for function execution logic."""
import pytest
from unittest.mock import Mock, MagicMock, patch
from computer_use_agent import execute_function_calls, SCREEN_WIDTH, SCREEN_HEIGHT
from google.genai.types import Candidate, FunctionCall


class TestExecuteFunctionCalls:
    """Tests for execute_function_calls function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.page = Mock()
        self.page.mouse = Mock()
        self.page.keyboard = Mock()
        self.page.wait_for_load_state = Mock()

    def create_candidate_with_call(self, function_name, args):
        """Helper to create a candidate with a function call."""
        function_call = Mock(spec=FunctionCall)
        function_call.name = function_name
        function_call.args = args

        part = Mock()
        part.function_call = function_call

        candidate = Mock(spec=Candidate)
        candidate.content = Mock()
        candidate.content.parts = [part]

        return candidate

    def test_open_web_browser(self):
        """Test open_web_browser action (no-op)."""
        candidate = self.create_candidate_with_call("open_web_browser", {})

        results = execute_function_calls(
            candidate, self.page, SCREEN_WIDTH, SCREEN_HEIGHT
        )

        assert len(results) == 1
        assert results[0][0] == "open_web_browser"
        assert results[0][1] == {"status": "ok"}
        # Should not interact with page
        self.page.mouse.click.assert_not_called()

    def test_click_at_basic(self):
        """Test basic click_at action."""
        candidate = self.create_candidate_with_call("click_at", {"x": 500, "y": 500})

        results = execute_function_calls(
            candidate, self.page, SCREEN_WIDTH, SCREEN_HEIGHT
        )

        assert len(results) == 1
        assert results[0][0] == "click_at"
        assert results[0][1] == {"status": "ok"}
        # 500/1000 * 1440 = 720, 500/1000 * 900 = 450
        self.page.mouse.click.assert_called_once_with(720, 450)
        self.page.wait_for_load_state.assert_called_once()

    def test_click_at_zero_coordinates(self):
        """Test click_at at origin (0,0)."""
        candidate = self.create_candidate_with_call("click_at", {"x": 0, "y": 0})

        results = execute_function_calls(
            candidate, self.page, SCREEN_WIDTH, SCREEN_HEIGHT
        )

        assert results[0][1] == {"status": "ok"}
        self.page.mouse.click.assert_called_once_with(0, 0)

    def test_click_at_max_coordinates(self):
        """Test click_at at maximum coordinates."""
        candidate = self.create_candidate_with_call("click_at", {"x": 1000, "y": 1000})

        results = execute_function_calls(
            candidate, self.page, SCREEN_WIDTH, SCREEN_HEIGHT
        )

        assert results[0][1] == {"status": "ok"}
        self.page.mouse.click.assert_called_once_with(SCREEN_WIDTH, SCREEN_HEIGHT)

    def test_click_at_missing_coordinates(self):
        """Test click_at with missing coordinates defaults to 0."""
        candidate = self.create_candidate_with_call("click_at", {})

        results = execute_function_calls(
            candidate, self.page, SCREEN_WIDTH, SCREEN_HEIGHT
        )

        assert results[0][1] == {"status": "ok"}
        self.page.mouse.click.assert_called_once_with(0, 0)

    def test_type_text_at_basic(self):
        """Test basic type_text_at action."""
        candidate = self.create_candidate_with_call(
            "type_text_at", {"x": 500, "y": 500, "text": "Hello World"}
        )

        results = execute_function_calls(
            candidate, self.page, SCREEN_WIDTH, SCREEN_HEIGHT
        )

        assert len(results) == 1
        assert results[0][0] == "type_text_at"
        assert results[0][1] == {"status": "ok"}

        # Should click, select all, delete, type
        self.page.mouse.click.assert_called_once_with(720, 450)
        assert self.page.keyboard.press.call_count == 2  # Meta+A and Backspace
        self.page.keyboard.type.assert_called_once_with("Hello World")

    def test_type_text_at_with_enter(self):
        """Test type_text_at with press_enter=True."""
        candidate = self.create_candidate_with_call(
            "type_text_at",
            {"x": 500, "y": 500, "text": "Search query", "press_enter": True},
        )

        results = execute_function_calls(
            candidate, self.page, SCREEN_WIDTH, SCREEN_HEIGHT
        )

        assert results[0][1] == {"status": "ok"}
        self.page.keyboard.type.assert_called_once_with("Search query")
        # Should press Meta+A, Backspace, and Enter
        assert self.page.keyboard.press.call_count == 3

    def test_type_text_at_without_enter(self):
        """Test type_text_at with press_enter=False."""
        candidate = self.create_candidate_with_call(
            "type_text_at",
            {"x": 500, "y": 500, "text": "Test", "press_enter": False},
        )

        results = execute_function_calls(
            candidate, self.page, SCREEN_WIDTH, SCREEN_HEIGHT
        )

        assert results[0][1] == {"status": "ok"}
        # Should press Meta+A and Backspace only
        assert self.page.keyboard.press.call_count == 2

    def test_type_text_at_sanitizes_text(self):
        """Test that type_text_at sanitizes input text."""
        candidate = self.create_candidate_with_call(
            "type_text_at",
            {"x": 500, "y": 500, "text": "Hello\x00World\x01"},
        )

        results = execute_function_calls(
            candidate, self.page, SCREEN_WIDTH, SCREEN_HEIGHT
        )

        assert results[0][1] == {"status": "ok"}
        # Non-printable characters should be removed
        self.page.keyboard.type.assert_called_once_with("HelloWorld")

    def test_unsupported_function(self):
        """Test handling of unsupported function."""
        candidate = self.create_candidate_with_call("unsupported_action", {})

        results = execute_function_calls(
            candidate, self.page, SCREEN_WIDTH, SCREEN_HEIGHT
        )

        assert len(results) == 1
        assert results[0][0] == "unsupported_action"
        assert results[0][1] == {"error": "unsupported_function"}
        # Should not interact with page
        self.page.mouse.click.assert_not_called()

    def test_multiple_function_calls(self):
        """Test execution of multiple function calls."""
        function_call1 = Mock(spec=FunctionCall)
        function_call1.name = "click_at"
        function_call1.args = {"x": 100, "y": 100}

        function_call2 = Mock(spec=FunctionCall)
        function_call2.name = "type_text_at"
        function_call2.args = {"x": 200, "y": 200, "text": "test"}

        part1 = Mock()
        part1.function_call = function_call1
        part2 = Mock()
        part2.function_call = function_call2

        candidate = Mock(spec=Candidate)
        candidate.content = Mock()
        candidate.content.parts = [part1, part2]

        results = execute_function_calls(
            candidate, self.page, SCREEN_WIDTH, SCREEN_HEIGHT
        )

        assert len(results) == 2
        assert results[0][0] == "click_at"
        assert results[1][0] == "type_text_at"

    def test_exception_handling(self):
        """Test that exceptions are caught and returned as errors."""
        self.page.mouse.click.side_effect = Exception("Click failed")
        candidate = self.create_candidate_with_call("click_at", {"x": 500, "y": 500})

        results = execute_function_calls(
            candidate, self.page, SCREEN_WIDTH, SCREEN_HEIGHT
        )

        assert len(results) == 1
        assert results[0][0] == "click_at"
        assert "error" in results[0][1]
        assert "Click failed" in results[0][1]["error"]

    def test_wait_for_load_state_called(self):
        """Test that wait_for_load_state is called after each action."""
        candidate = self.create_candidate_with_call("click_at", {"x": 500, "y": 500})

        execute_function_calls(candidate, self.page, SCREEN_WIDTH, SCREEN_HEIGHT)

        self.page.wait_for_load_state.assert_called_once()

    @patch("computer_use_agent.time.sleep")
    def test_page_settle_delay(self, mock_sleep):
        """Test that page settle delay is applied."""
        candidate = self.create_candidate_with_call("click_at", {"x": 500, "y": 500})

        execute_function_calls(candidate, self.page, SCREEN_WIDTH, SCREEN_HEIGHT)

        mock_sleep.assert_called_once_with(1.0)  # PAGE_SETTLE_DELAY_SECONDS
