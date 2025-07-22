"""Test CLI security features including path traversal protection."""

import pytest
from openapi_doc_generator.cli import main


class TestCLIPathTraversalSecurity:
    """Test path traversal attack prevention in CLI arguments."""

    def test_app_path_traversal_basic_dotdot(self, tmp_path):
        """Test that basic ../ traversal attempts are blocked in app path."""
        app = tmp_path / "app.py"
        app.write_text("from fastapi import FastAPI\napp = FastAPI()")

        # Try to access a file outside current directory using ../
        with pytest.raises(SystemExit) as exc_info:
            main(["--app", "../../../etc/passwd"])

        assert exc_info.value.code == 2

    def test_app_path_traversal_absolute_with_dotdot(self, tmp_path):
        """Test that absolute paths with ../ traversal are blocked."""
        app = tmp_path / "app.py"
        app.write_text("from fastapi import FastAPI\napp = FastAPI()")

        # Try absolute path with traversal
        with pytest.raises(SystemExit) as exc_info:
            main(["--app", "/tmp/../etc/passwd"])

        assert exc_info.value.code == 2

    def test_app_path_traversal_nested_dotdot(self, tmp_path):
        """Test that nested ../ patterns are blocked."""
        app = tmp_path / "app.py"
        app.write_text("from fastapi import FastAPI\napp = FastAPI()")

        # Try nested traversal patterns
        with pytest.raises(SystemExit) as exc_info:
            main(["--app", "dir1/../dir2/../../etc/passwd"])

        assert exc_info.value.code == 2

    def test_output_path_traversal_protection(self, tmp_path):
        """Test that output path traversal attempts are blocked."""
        app = tmp_path / "app.py"
        app.write_text("from fastapi import FastAPI\napp = FastAPI()")

        # Try to write to a file outside using traversal
        with pytest.raises(SystemExit) as exc_info:
            main(["--app", str(app), "-o", "../../../tmp/malicious_output.json"])

        assert exc_info.value.code == 2

    def test_tests_path_traversal_protection(self, tmp_path):
        """Test that tests path traversal attempts are blocked."""
        app = tmp_path / "app.py"
        app.write_text("from fastapi import FastAPI\napp = FastAPI()")

        # Try to write tests to unauthorized location
        with pytest.raises(SystemExit) as exc_info:
            main(["--app", str(app), "--tests", "../../../tmp/malicious_tests.py"])

        assert exc_info.value.code == 2

    def test_old_spec_path_traversal_protection(self, tmp_path):
        """Test that old-spec path traversal attempts are blocked."""
        app = tmp_path / "app.py"
        app.write_text("from fastapi import FastAPI\napp = FastAPI()")

        # Try to read old spec from unauthorized location
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--app",
                    str(app),
                    "--format",
                    "guide",
                    "--old-spec",
                    "../../../etc/passwd",
                ]
            )

        assert exc_info.value.code == 2

    def test_legitimate_relative_paths_allowed(self, tmp_path):
        """Test that legitimate relative paths without traversal work."""
        app = tmp_path / "app.py"
        app.write_text("from fastapi import FastAPI\napp = FastAPI()")

        subdir = tmp_path / "subdir"
        subdir.mkdir()
        output_file = subdir / "output.json"

        # This should work - relative path within current directory tree
        try:
            main(["--app", str(app), "-o", str(output_file)])
            # Should not raise SystemExit
        except SystemExit as e:
            # Only fail if it's due to path validation (code 2)
            if e.code == 2:
                pytest.fail("Legitimate relative path was incorrectly blocked")

    def test_legitimate_absolute_paths_allowed(self, tmp_path):
        """Test that legitimate absolute paths without traversal work."""
        app = tmp_path / "app.py"
        app.write_text("from fastapi import FastAPI\napp = FastAPI()")

        output_file = tmp_path / "output.json"

        # Absolute path without traversal should work
        try:
            main(["--app", str(app), "-o", str(output_file)])
            # Should not raise SystemExit due to path validation
        except SystemExit as e:
            # Only fail if it's due to path validation (code 2)
            if e.code == 2:
                pytest.fail("Legitimate absolute path was incorrectly blocked")


class TestCLISecurityErrorMessages:
    """Test that security error messages are informative but not revealing."""

    def test_path_traversal_error_message_content(self, tmp_path, capsys):
        """Test that path traversal errors have appropriate error messages."""
        with pytest.raises(SystemExit):
            main(["--app", "../../../etc/passwd"])

        captured = capsys.readouterr()
        error_output = captured.err.lower()

        # Should contain CLI error code and indicate path issue
        assert "cli001" in error_output
        assert "invalid app path" in error_output

        # Should not reveal sensitive system details beyond user input
        assert "sensitive" not in error_output

    def test_output_path_traversal_error_code(self, tmp_path, capsys):
        """Test that output path traversal uses correct error code."""
        app = tmp_path / "app.py"
        app.write_text("from fastapi import FastAPI\napp = FastAPI()")

        with pytest.raises(SystemExit):
            main(["--app", str(app), "-o", "../../../tmp/evil"])

        captured = capsys.readouterr()
        # Should use CLI004 error code for output path issues
        assert "cli004" in captured.err.lower()

    def test_tests_path_traversal_error_code(self, tmp_path, capsys):
        """Test that tests path traversal uses correct error code."""
        app = tmp_path / "app.py"
        app.write_text("from fastapi import FastAPI\napp = FastAPI()")

        with pytest.raises(SystemExit):
            main(["--app", str(app), "--tests", "../../../tmp/evil"])

        captured = capsys.readouterr()
        # Should use CLI005 error code for tests path issues
        assert "cli005" in captured.err.lower()


class TestCLISecurityBoundaryConditions:
    """Test edge cases and boundary conditions in security validation."""

    def test_empty_path_handling(self):
        """Test behavior with empty path arguments."""
        with pytest.raises(SystemExit):
            main(["--app", ""])
        # Should fail gracefully, not crash

    def test_just_dotdot_path(self):
        """Test path that is just '..'"""
        with pytest.raises(SystemExit) as exc_info:
            main(["--app", ".."])

        assert exc_info.value.code == 2

    def test_path_with_null_bytes(self):
        """Test paths containing null bytes (potential injection)."""
        with pytest.raises(SystemExit):
            main(["--app", "app\x00.py"])
        # Should be blocked or handled gracefully

    def test_unicode_traversal_attempts(self):
        """Test Unicode-based traversal attempts."""
        with pytest.raises(SystemExit):
            main(["--app", "\u002e\u002e/\u002e\u002e/etc/passwd"])
        # Unicode dots should still be caught

    def test_mixed_slash_types(self):
        """Test mixed forward and back slashes."""
        with pytest.raises(SystemExit):
            main(["--app", "..\\..\\windows\\system32\\config"])
        # Should handle Windows-style paths
