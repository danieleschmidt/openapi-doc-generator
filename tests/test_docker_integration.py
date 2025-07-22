"""Tests for Docker image functionality and integration."""

import subprocess
import json
import pytest
from pathlib import Path


class TestDockerImage:
    """Test suite for Docker image functionality."""

    def test_dockerfile_exists(self):
        """Test that Dockerfile exists in project root."""
        dockerfile_path = Path(__file__).parent.parent / "Dockerfile"
        assert dockerfile_path.exists(), "Dockerfile should exist in project root"

    def test_dockerignore_exists(self):
        """Test that .dockerignore exists in project root."""
        dockerignore_path = Path(__file__).parent.parent / ".dockerignore"
        assert dockerignore_path.exists(), ".dockerignore should exist in project root"

    def test_docker_image_builds_successfully(self):
        """Test that Docker image can be built without errors (via fixture)."""
        # Image is built by the setup_and_cleanup_test_image fixture
        # This test verifies the image exists and can be inspected
        result = subprocess.run(
            ["docker", "inspect", "openapi-doc-generator:test"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, (
            f"Docker image not found or inspect failed: {result.stderr}"
        )

    def test_docker_image_has_correct_entrypoint(self):
        """Test that Docker image has correct entrypoint and can run CLI."""
        # Run docker image with --help to verify entrypoint
        result = subprocess.run(
            ["docker", "run", "--rm", "openapi-doc-generator:test", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Docker run failed: {result.stderr}"
        assert "Generate documentation in various formats" in result.stdout

    def test_docker_image_can_process_example(self):
        """Test that Docker image can process the example application."""
        project_root = Path(__file__).parent.parent
        examples_dir = project_root / "examples"

        if not examples_dir.exists():
            pytest.skip("Examples directory not found")

        # Mount examples directory and run documentation generation
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{examples_dir}:/workspace",
                "openapi-doc-generator:test",
                "--app",
                "/workspace/app.py",
                "--output",
                "/tmp/output.md",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, (
            f"Docker run with example failed: {result.stderr}"
        )

    def test_docker_image_supports_json_logging(self):
        """Test that Docker image supports JSON logging format."""
        project_root = Path(__file__).parent.parent
        examples_dir = project_root / "examples"

        if not examples_dir.exists():
            pytest.skip("Examples directory not found")

        # Run with JSON logging
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{examples_dir}:/workspace",
                "openapi-doc-generator:test",
                "--app",
                "/workspace/app.py",
                "--log-format",
                "json",
                "--output",
                "/tmp/output.md",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, (
            f"Docker run with JSON logging failed: {result.stderr}"
        )

        # Verify JSON log format in stderr
        log_lines = [line for line in result.stderr.split("\n") if line.strip()]
        for line in log_lines:
            if line.strip():
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    pytest.fail(f"Invalid JSON log line: {line}")

    def test_docker_image_size_reasonable(self):
        """Test that Docker image size is reasonable (< 500MB)."""
        result = subprocess.run(
            [
                "docker",
                "images",
                "openapi-doc-generator:test",
                "--format",
                "table {{.Size}}",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Docker images command failed: {result.stderr}"

        # Parse size (assuming format like "123MB" or "1.23GB")
        size_line = result.stdout.strip().split("\n")[-1]
        if "GB" in size_line:
            size_gb = float(size_line.replace("GB", "").strip())
            assert size_gb < 0.5, f"Docker image too large: {size_line}"
        elif "MB" in size_line:
            size_mb = float(size_line.replace("MB", "").strip())
            assert size_mb < 500, f"Docker image too large: {size_line}"

    def test_docker_image_runs_as_non_root(self):
        """Test that Docker image runs as non-root user for security."""
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint=",
                "openapi-doc-generator:test",
                "whoami",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Docker whoami failed: {result.stderr}"
        assert result.stdout.strip() != "root", (
            "Docker container should not run as root"
        )

    def test_docker_image_has_health_check(self):
        """Test that Docker image includes a health check."""
        result = subprocess.run(
            ["docker", "inspect", "openapi-doc-generator:test"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Docker inspect failed: {result.stderr}"

        inspect_data = json.loads(result.stdout)[0]
        config = inspect_data.get("Config", {})

        assert "Healthcheck" in config, (
            "Docker image should have health check configured"
        )
        assert config["Healthcheck"]["Test"], "Health check should have test command"

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_cleanup_test_image(self):
        """Setup and clean up test Docker image for tests."""
        project_root = Path(__file__).parent.parent

        # Build test image before tests
        result = subprocess.run(
            ["docker", "build", "-t", "openapi-doc-generator:test", "."],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            pytest.fail(f"Failed to build Docker image for tests: {result.stderr}")

        yield

        # Remove test image after tests
        subprocess.run(
            ["docker", "rmi", "openapi-doc-generator:test"], capture_output=True
        )


class TestDockerCompose:
    """Test suite for Docker Compose configuration."""

    def test_docker_compose_file_exists(self):
        """Test that docker-compose.yml exists for development."""
        compose_path = Path(__file__).parent.parent / "docker-compose.yml"
        assert compose_path.exists(), "docker-compose.yml should exist for development"

    def test_docker_compose_service_configuration(self):
        """Test that Docker Compose has correct service configuration."""
        compose_path = Path(__file__).parent.parent / "docker-compose.yml"

        if not compose_path.exists():
            pytest.skip("docker-compose.yml not found")

        import yaml

        with open(compose_path) as f:
            compose_config = yaml.safe_load(f)

        assert "services" in compose_config
        assert "openapi-doc-generator" in compose_config["services"]

        service = compose_config["services"]["openapi-doc-generator"]
        assert "build" in service or "image" in service
        assert "volumes" in service  # Should mount source code for development
