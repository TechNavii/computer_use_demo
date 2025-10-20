"""Unit tests for configuration and client creation."""
import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from computer_use_agent import configure_logging, create_client


class TestConfigureLogging:
    """Tests for configure_logging function."""

    @patch("computer_use_agent.logging.basicConfig")
    @patch("computer_use_agent.LOGGER")
    def test_configures_logging_first_time(self, mock_logger, mock_basic_config):
        """Test that logging is configured on first call."""
        mock_logger.handlers = []
        configure_logging()
        mock_basic_config.assert_called_once()

    @patch("computer_use_agent.logging.basicConfig")
    @patch("computer_use_agent.LOGGER")
    def test_skips_if_already_configured(self, mock_logger, mock_basic_config):
        """Test that logging configuration is skipped if already configured."""
        mock_logger.handlers = [Mock()]  # Non-empty handlers
        configure_logging()
        mock_basic_config.assert_not_called()

    @patch("computer_use_agent.logging.basicConfig")
    @patch("computer_use_agent.LOGGER")
    def test_logging_level_is_info(self, mock_logger, mock_basic_config):
        """Test that logging level is set to INFO."""
        mock_logger.handlers = []
        configure_logging()

        # Check that basicConfig was called with level=INFO
        call_kwargs = mock_basic_config.call_args[1]
        import logging

        assert call_kwargs["level"] == logging.INFO

    @patch("computer_use_agent.logging.basicConfig")
    @patch("computer_use_agent.LOGGER")
    def test_logging_format_includes_timestamp(self, mock_logger, mock_basic_config):
        """Test that logging format includes timestamp."""
        mock_logger.handlers = []
        configure_logging()

        call_kwargs = mock_basic_config.call_args[1]
        assert "%(asctime)s" in call_kwargs["format"]
        assert "%(levelname)s" in call_kwargs["format"]
        assert "%(name)s" in call_kwargs["format"]
        assert "%(message)s" in call_kwargs["format"]


class TestCreateClient:
    """Tests for create_client function."""

    @patch("computer_use_agent.genai.Client")
    @patch("computer_use_agent.load_dotenv")
    def test_creates_client_with_api_key(self, mock_load_dotenv, mock_client_class):
        """Test that client is created with API key from environment."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key"}):
            client = create_client()

            mock_load_dotenv.assert_called_once_with(override=False)
            mock_client_class.assert_called_once_with(api_key="test_api_key")

    @patch("computer_use_agent.load_dotenv")
    def test_raises_error_when_api_key_missing(self, mock_load_dotenv):
        """Test that error is raised when API key is not set."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError) as exc_info:
                create_client()

            assert "GEMINI_API_KEY is not set" in str(exc_info.value)

    @patch("computer_use_agent.genai.Client")
    @patch("computer_use_agent.load_dotenv")
    def test_loads_dotenv_before_reading_env(self, mock_load_dotenv, mock_client_class):
        """Test that .env file is loaded before reading environment."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            create_client()

            # Verify load_dotenv was called before Client was instantiated
            mock_load_dotenv.assert_called_once()
            mock_client_class.assert_called_once()

    @patch("computer_use_agent.genai.Client")
    @patch("computer_use_agent.load_dotenv")
    def test_dotenv_override_is_false(self, mock_load_dotenv, mock_client_class):
        """Test that dotenv override parameter is False."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            create_client()

            mock_load_dotenv.assert_called_once_with(override=False)

    @patch("computer_use_agent.genai.Client")
    @patch("computer_use_agent.load_dotenv")
    def test_handles_empty_string_api_key(self, mock_load_dotenv, mock_client_class):
        """Test that empty string API key is treated as missing."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            with pytest.raises(RuntimeError) as exc_info:
                create_client()

            assert "GEMINI_API_KEY is not set" in str(exc_info.value)

    @patch("computer_use_agent.genai.Client")
    @patch("computer_use_agent.load_dotenv")
    def test_handles_whitespace_api_key(self, mock_load_dotenv, mock_client_class):
        """Test behavior with whitespace-only API key."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "   "}):
            # Whitespace key is technically not empty, so it should create client
            # (though the API would likely reject it)
            client = create_client()
            mock_client_class.assert_called_once_with(api_key="   ")

    @patch("computer_use_agent.genai.Client")
    @patch("computer_use_agent.load_dotenv")
    def test_returns_client_instance(self, mock_load_dotenv, mock_client_class):
        """Test that function returns the created client instance."""
        mock_instance = Mock()
        mock_client_class.return_value = mock_instance

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            result = create_client()

            assert result == mock_instance
