"""Unit tests for function call extraction and processing."""
import pytest
from unittest.mock import Mock, MagicMock
from computer_use_agent import collect_function_calls, get_function_responses
from google.genai.types import Candidate, Content, Part, FunctionCall, FunctionResponse


class TestCollectFunctionCalls:
    """Tests for collect_function_calls function."""

    def test_no_content(self):
        """Test candidate with no content."""
        candidate = Mock(spec=Candidate)
        candidate.content = None
        result = collect_function_calls(candidate)
        assert result == []

    def test_no_parts(self):
        """Test candidate with content but no parts."""
        candidate = Mock(spec=Candidate)
        candidate.content = Mock()
        candidate.content.parts = None
        result = collect_function_calls(candidate)
        assert result == []

    def test_empty_parts(self):
        """Test candidate with empty parts list."""
        candidate = Mock(spec=Candidate)
        candidate.content = Mock()
        candidate.content.parts = []
        result = collect_function_calls(candidate)
        assert result == []

    def test_single_function_call(self):
        """Test extraction of single function call."""
        function_call = Mock(spec=FunctionCall)
        function_call.name = "click_at"
        function_call.args = {"x": 100, "y": 200}

        part = Mock()
        part.function_call = function_call

        candidate = Mock(spec=Candidate)
        candidate.content = Mock()
        candidate.content.parts = [part]

        result = collect_function_calls(candidate)
        assert len(result) == 1
        assert result[0] == function_call

    def test_multiple_function_calls(self):
        """Test extraction of multiple function calls."""
        function_call1 = Mock(spec=FunctionCall)
        function_call1.name = "click_at"
        function_call1.args = {"x": 100, "y": 200}

        function_call2 = Mock(spec=FunctionCall)
        function_call2.name = "type_text_at"
        function_call2.args = {"x": 100, "y": 200, "text": "hello"}

        part1 = Mock()
        part1.function_call = function_call1

        part2 = Mock()
        part2.function_call = function_call2

        candidate = Mock(spec=Candidate)
        candidate.content = Mock()
        candidate.content.parts = [part1, part2]

        result = collect_function_calls(candidate)
        assert len(result) == 2
        assert result[0] == function_call1
        assert result[1] == function_call2

    def test_mixed_parts(self):
        """Test extraction with mixed parts (text and function calls)."""
        function_call = Mock(spec=FunctionCall)
        function_call.name = "click_at"

        part_with_call = Mock()
        part_with_call.function_call = function_call

        part_without_call = Mock()
        part_without_call.function_call = None

        candidate = Mock(spec=Candidate)
        candidate.content = Mock()
        candidate.content.parts = [part_without_call, part_with_call]

        result = collect_function_calls(candidate)
        assert len(result) == 1
        assert result[0] == function_call


class TestGetFunctionResponses:
    """Tests for get_function_responses function."""

    def test_empty_results(self):
        """Test with empty results list."""
        page = Mock()
        page.screenshot.return_value = b"fake_screenshot"
        page.url = "https://example.com"

        result = get_function_responses(page, [])
        assert result == []

    def test_single_result(self):
        """Test with single function result."""
        page = Mock()
        page.screenshot.return_value = b"fake_screenshot"
        page.url = "https://example.com"

        results = [("click_at", {"status": "ok"})]
        responses = get_function_responses(page, results)

        assert len(responses) == 1
        assert isinstance(responses[0], FunctionResponse)
        assert responses[0].name == "click_at"
        assert responses[0].response == {"url": "https://example.com", "status": "ok"}

    def test_multiple_results(self):
        """Test with multiple function results."""
        page = Mock()
        page.screenshot.return_value = b"fake_screenshot"
        page.url = "https://example.com"

        results = [
            ("click_at", {"status": "ok"}),
            ("type_text_at", {"status": "ok"}),
        ]
        responses = get_function_responses(page, results)

        assert len(responses) == 2
        assert responses[0].name == "click_at"
        assert responses[1].name == "type_text_at"

    def test_error_result(self):
        """Test with error result."""
        page = Mock()
        page.screenshot.return_value = b"fake_screenshot"
        page.url = "https://example.com"

        results = [("click_at", {"error": "Element not found"})]
        responses = get_function_responses(page, results)

        assert len(responses) == 1
        assert responses[0].response == {
            "url": "https://example.com",
            "error": "Element not found",
        }

    def test_screenshot_failure(self):
        """Test handling of screenshot failure."""
        page = Mock()
        page.screenshot.side_effect = Exception("Screenshot failed")
        page.url = "https://example.com"

        results = [("click_at", {"status": "ok"})]
        responses = get_function_responses(page, results)

        # Should still return response but with empty screenshot
        assert len(responses) == 1
        assert responses[0].name == "click_at"

    def test_url_included_in_response(self):
        """Test that current URL is included in response."""
        page = Mock()
        page.screenshot.return_value = b"fake_screenshot"
        page.url = "https://example.com/page"

        results = [("click_at", {"status": "ok"})]
        responses = get_function_responses(page, results)

        assert responses[0].response["url"] == "https://example.com/page"

    def test_screenshot_included_in_parts(self):
        """Test that screenshot is included in response parts."""
        page = Mock()
        screenshot_data = b"fake_screenshot_data"
        page.screenshot.return_value = screenshot_data
        page.url = "https://example.com"

        results = [("click_at", {"status": "ok"})]
        responses = get_function_responses(page, results)

        assert len(responses[0].parts) == 1
        assert responses[0].parts[0].inline_data.mime_type == "image/png"
        assert responses[0].parts[0].inline_data.data == screenshot_data
