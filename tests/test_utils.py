"""Unit tests for utility functions."""
import pytest
from computer_use_agent import (
    denormalize_coordinate,
    sanitize_text,
    extract_text_response,
)
from google.genai.types import Part


class TestDenormalizeCoordinate:
    """Tests for denormalize_coordinate function."""

    def test_zero_coordinate(self):
        """Test that 0 normalizes to 0."""
        result = denormalize_coordinate(0, 1440)
        assert result == 0

    def test_max_coordinate(self):
        """Test that 1000 normalizes to screen dimension."""
        result = denormalize_coordinate(1000, 1440)
        assert result == 1440

    def test_middle_coordinate(self):
        """Test that 500 normalizes to half of screen dimension."""
        result = denormalize_coordinate(500, 1440)
        assert result == 720

    def test_clamping_above_max(self):
        """Test that values above 1000 are clamped to max."""
        result = denormalize_coordinate(1500, 1440)
        assert result == 1440

    def test_clamping_below_min(self):
        """Test that negative values are clamped to 0."""
        result = denormalize_coordinate(-100, 1440)
        assert result == 0

    def test_float_values(self):
        """Test that float values are handled correctly."""
        result = denormalize_coordinate(250.5, 1440)
        assert result == 360  # 250.5/1000 * 1440 = 360.72

    def test_different_dimensions(self):
        """Test with different screen dimensions."""
        assert denormalize_coordinate(500, 900) == 450
        assert denormalize_coordinate(500, 1920) == 960
        assert denormalize_coordinate(500, 1080) == 540


class TestSanitizeText:
    """Tests for sanitize_text function."""

    def test_simple_text(self):
        """Test that simple text passes through unchanged."""
        result = sanitize_text("Hello World")
        assert result == "Hello World"

    def test_none_input(self):
        """Test that None input returns empty string."""
        result = sanitize_text(None)
        assert result == ""

    def test_preserves_whitespace(self):
        """Test that tabs, newlines, and spaces are preserved."""
        result = sanitize_text("Hello\n\tWorld\r\n")
        assert result == "Hello\n\tWorld\r\n"

    def test_strips_non_printable(self):
        """Test that non-printable characters are removed."""
        result = sanitize_text("Hello\x00World\x01")
        assert result == "HelloWorld"

    def test_max_length_enforcement(self):
        """Test that text is truncated to MAX_TYPABLE_CHARS."""
        long_text = "A" * 2000
        result = sanitize_text(long_text)
        assert len(result) == 1024  # MAX_TYPABLE_CHARS

    def test_unicode_text(self):
        """Test that unicode characters are preserved."""
        result = sanitize_text("Hello 世界 🌍")
        assert result == "Hello 世界 🌍"

    def test_numeric_input(self):
        """Test that numeric input is converted to string."""
        result = sanitize_text(12345)
        assert result == "12345"

    def test_mixed_content(self):
        """Test mixed printable and non-printable content."""
        result = sanitize_text("Valid\x00Invalid\x01Text")
        assert result == "ValidInvalidText"


class TestExtractTextResponse:
    """Tests for extract_text_response function."""

    def test_single_text_part(self):
        """Test extraction from single text part."""
        parts = [Part(text="Hello World")]
        result = extract_text_response(parts)
        assert result == "Hello World"

    def test_multiple_text_parts(self):
        """Test extraction from multiple text parts."""
        parts = [
            Part(text="Hello"),
            Part(text="World"),
        ]
        result = extract_text_response(parts)
        assert result == "Hello World"

    def test_empty_parts(self):
        """Test with empty parts list."""
        result = extract_text_response([])
        assert result == ""

    def test_parts_with_whitespace(self):
        """Test that whitespace is stripped from parts."""
        parts = [
            Part(text="  Hello  "),
            Part(text="  World  "),
        ]
        result = extract_text_response(parts)
        assert result == "Hello World"

    def test_parts_with_empty_text(self):
        """Test that empty text parts are ignored."""
        parts = [
            Part(text="Hello"),
            Part(text=""),
            Part(text="World"),
        ]
        result = extract_text_response(parts)
        assert result == "Hello World"

    def test_parts_without_text_attribute(self):
        """Test handling of parts without text attribute."""
        # Create a mock part without text attribute
        class MockPart:
            pass

        parts = [MockPart()]
        result = extract_text_response(parts)
        assert result == ""
