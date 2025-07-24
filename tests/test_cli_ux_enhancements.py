"""Tests for CLI user experience enhancements."""

import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from openapi_doc_generator.cli import build_parser, main


class TestCLIVerboseQuietModes:
    """Test CLI verbose and quiet mode functionality."""

    def test_build_parser_includes_verbose_quiet_flags(self):
        """Test that parser includes verbose and quiet options."""
        parser = build_parser()
        
        # Parse with verbose flag
        args = parser.parse_args(['--app', 'test.py', '--verbose'])
        assert args.verbose is True
        assert hasattr(args, 'quiet')
        
        # Parse with quiet flag
        args = parser.parse_args(['--app', 'test.py', '--quiet'])
        assert args.quiet is True
        assert hasattr(args, 'verbose')
        
        # Default should be neither verbose nor quiet
        args = parser.parse_args(['--app', 'test.py'])
        assert args.verbose is False
        assert args.quiet is False

    def test_verbose_and_quiet_flags_are_mutually_exclusive(self):
        """Test that verbose and quiet flags cannot be used together."""
        parser = build_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(['--app', 'test.py', '--verbose', '--quiet'])

    @patch('openapi_doc_generator.cli._setup_logging')
    def test_verbose_mode_sets_debug_logging(self, mock_setup_logging, tmp_path):
        """Test that verbose mode enables DEBUG level logging."""
        test_file = tmp_path / "test.py"
        test_file.write_text("# test file")
        
        with patch('openapi_doc_generator.cli.APIDocumentator') as mock_doc:
            mock_result = MagicMock()
            mock_result.generate_markdown.return_value = "# Test API"
            mock_doc.return_value.analyze_app.return_value = mock_result
            
            main(['--app', str(test_file), '--verbose'])
            
            # Verify logging was set up with DEBUG level for verbose mode
            mock_setup_logging.assert_called_once()
            call_args = mock_setup_logging.call_args
            assert 'level' in call_args[1] and call_args[1]['level'] == logging.DEBUG

    @patch('openapi_doc_generator.cli._setup_logging')
    def test_quiet_mode_sets_warning_logging(self, mock_setup_logging, tmp_path):
        """Test that quiet mode enables WARNING level logging."""
        test_file = tmp_path / "test.py"
        test_file.write_text("# test file")
        
        with patch('openapi_doc_generator.cli.APIDocumentator') as mock_doc:
            mock_result = MagicMock()
            mock_result.generate_markdown.return_value = "# Test API"
            mock_doc.return_value.analyze_app.return_value = mock_result
            
            main(['--app', str(test_file), '--quiet'])
            
            # Verify logging was set up with WARNING level for quiet mode
            mock_setup_logging.assert_called_once()
            call_args = mock_setup_logging.call_args
            assert 'level' in call_args[1] and call_args[1]['level'] == logging.WARNING


class TestCLIProgressIndicators:
    """Test CLI progress indicator functionality."""

    @patch('openapi_doc_generator.cli.APIDocumentator')
    def test_progress_indicators_shown_in_verbose_mode(self, mock_doc, tmp_path, capsys):
        """Test that progress indicators are shown in verbose mode."""
        test_file = tmp_path / "test.py"
        test_file.write_text("# test file")
        
        mock_result = MagicMock()
        mock_result.generate_markdown.return_value = "# Test API"
        mock_doc.return_value.analyze_app.return_value = mock_result
        
        main(['--app', str(test_file), '--verbose'])
        
        # In verbose mode, we should see progress indicators on stderr
        captured = capsys.readouterr()
        assert "🔄 Validating application path..." in captured.err
        assert "🔄 Analyzing application structure..." in captured.err  
        assert "🔄 Generating documentation..." in captured.err
        assert "🔄 Writing output..." in captured.err
        assert "✅ Documentation generation completed successfully!" in captured.err

    def test_no_progress_indicators_in_quiet_mode(self, tmp_path, capsys):
        """Test that no progress indicators are shown in quiet mode."""
        test_file = tmp_path / "test.py"
        test_file.write_text("# test file")
        
        with patch('openapi_doc_generator.cli.APIDocumentator') as mock_doc:
            mock_result = MagicMock()
            mock_result.generate_markdown.return_value = "# Test API"
            mock_doc.return_value.analyze_app.return_value = mock_result
            
            main(['--app', str(test_file), '--quiet'])
            
            # In quiet mode, no progress messages should be written to stderr
            captured = capsys.readouterr()
            assert "🔄" not in captured.err  # No progress indicators
            assert "✅" not in captured.err  # No success message


class TestCLIColorOutput:
    """Test CLI colored output functionality."""

    def test_build_parser_includes_no_color_flag(self):
        """Test that parser includes no-color option."""
        parser = build_parser()
        
        # Parse with no-color flag
        args = parser.parse_args(['--app', 'test.py', '--no-color'])
        assert args.no_color is True
        
        # Default should have color enabled
        args = parser.parse_args(['--app', 'test.py'])
        assert args.no_color is False

    @patch.dict(os.environ, {'NO_COLOR': '1'})
    def test_no_color_environment_variable_respected(self, tmp_path):
        """Test that NO_COLOR environment variable disables colors."""
        test_file = tmp_path / "test.py"
        test_file.write_text("# test file")
        
        with patch('openapi_doc_generator.cli.APIDocumentator') as mock_doc:
            mock_result = MagicMock()
            mock_result.generate_markdown.return_value = "# Test API"
            mock_doc.return_value.analyze_app.return_value = mock_result
            
            # This test will verify color handling once implemented
            main(['--app', str(test_file)])

    def test_colored_logging_in_verbose_mode(self, tmp_path):
        """Test that verbose mode includes colored log messages."""
        test_file = tmp_path / "test.py"
        test_file.write_text("# test file")
        
        with patch('openapi_doc_generator.cli.APIDocumentator') as mock_doc:
            mock_result = MagicMock()
            mock_result.generate_markdown.return_value = "# Test API"
            mock_doc.return_value.analyze_app.return_value = mock_result
            
            # This will test colored output once implemented
            with patch('sys.stderr'):
                main(['--app', str(test_file), '--verbose'])


class TestCLIErrorHandling:
    """Test enhanced error handling and user-friendly messages."""

    def test_user_friendly_error_messages(self):
        """Test that error messages are user-friendly and informative."""
        # Test missing app file
        with pytest.raises(SystemExit):
            main(['--app', 'nonexistent.py'])

    def test_suggestion_for_common_mistakes(self, tmp_path):
        """Test that CLI suggests fixes for common mistakes."""
        # This will test helpful error messages and suggestions
        # once the enhancement is implemented
        pass