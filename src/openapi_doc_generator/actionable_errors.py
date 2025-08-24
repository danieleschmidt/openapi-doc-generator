"""Actionable error messages and guidance for user issues."""

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class ErrorCategory(Enum):
    """Categories of errors with different guidance approaches."""
    FRAMEWORK_DETECTION = "framework_detection"
    FILE_ACCESS = "file_access"
    DEPENDENCY_MISSING = "dependency_missing"
    CONFIGURATION = "configuration"
    INPUT_VALIDATION = "input_validation"
    PLUGIN_LOADING = "plugin_loading"
    PARSING_ERROR = "parsing_error"


@dataclass
class QuickFix:
    """A quick fix suggestion."""
    description: str
    command: Optional[str] = None
    example_code: Optional[str] = None


@dataclass
class DetailedStep:
    """A detailed troubleshooting step."""
    step_number: int
    title: str
    description: str
    command: Optional[str] = None
    example_code: Optional[str] = None
    verification: Optional[str] = None


@dataclass
class ActionableError:
    """Complete error with actionable guidance."""
    category: ErrorCategory
    original_error: str
    user_message: str
    quick_fixes: List[QuickFix]
    detailed_steps: List[DetailedStep]
    related_documentation: Dict[str, str]
    prevention_tips: List[str]


class ActionableErrorHandler:
    """Converts technical errors into actionable user guidance."""
    
    def __init__(self):
        self.error_patterns = self._initialize_error_patterns()
        self.guidance_templates = self._initialize_guidance_templates()
    
    def _initialize_error_patterns(self) -> Dict[str, tuple]:
        """Initialize error pattern matching."""
        return {
            r"No module named '(\w+)'": (ErrorCategory.DEPENDENCY_MISSING, "missing_dependency"),
            r"Framework detection failed": (ErrorCategory.FRAMEWORK_DETECTION, "framework_not_detected"),
            r"\[Errno 2\] No such file or directory: '(.+)'": (ErrorCategory.FILE_ACCESS, "file_not_found"),
            r"Permission denied.*'(.+)'": (ErrorCategory.FILE_ACCESS, "permission_denied"),
            r"Invalid app path": (ErrorCategory.INPUT_VALIDATION, "invalid_app_path"),
            r"Plugin loading failed.*'(\w+)'": (ErrorCategory.PLUGIN_LOADING, "plugin_load_failed"),
            r"SyntaxError.*line (\d+)": (ErrorCategory.PARSING_ERROR, "syntax_error"),
        }
    
    def _initialize_guidance_templates(self) -> Dict[str, ActionableError]:
        """Initialize guidance templates for common errors."""
        return {
            "missing_dependency": ActionableError(
                category=ErrorCategory.DEPENDENCY_MISSING,
                original_error="",
                user_message="Required dependency is not installed.",
                quick_fixes=[
                    QuickFix(
                        "Install missing dependency",
                        command="pip install {dependency}",
                        example_code="pip install jinja2"
                    ),
                    QuickFix(
                        "Install all dependencies",
                        command="pip install -e .[dev]",
                    )
                ],
                detailed_steps=[
                    DetailedStep(1, "Check Python environment", 
                                "Verify you're in the correct Python environment",
                                command="which python && python --version"),
                    DetailedStep(2, "Install missing dependency",
                                "Install the specific dependency mentioned in the error",
                                verification="Try importing the module: python -c 'import {dependency}'"),
                    DetailedStep(3, "Verify installation",
                                "Confirm the package is properly installed",
                                command="pip show {dependency}")
                ],
                related_documentation={
                    "Installation Guide": "https://github.com/user/repo#installation",
                    "Dependencies": "https://github.com/user/repo#dependencies"
                },
                prevention_tips=[
                    "Use virtual environments to isolate dependencies",
                    "Install the package in development mode with pip install -e .[dev]",
                    "Check requirements before running the tool"
                ]
            ),
            
            "framework_not_detected": ActionableError(
                category=ErrorCategory.FRAMEWORK_DETECTION,
                original_error="",
                user_message="Unable to detect your web framework automatically.",
                quick_fixes=[
                    QuickFix(
                        "Check supported frameworks",
                        example_code="# Supported: FastAPI, Flask, Django, Express.js, Tornado"
                    ),
                    QuickFix(
                        "Verify framework imports",
                        example_code="from flask import Flask  # Flask example"
                    ),
                    QuickFix(
                        "Use standard decorators",
                        example_code="@app.route('/users')  # Flask/FastAPI style"
                    )
                ],
                detailed_steps=[
                    DetailedStep(1, "Check file content", 
                                "Open your main application file and verify it contains framework imports"),
                    DetailedStep(2, "Verify imports",
                                "Ensure your file has proper framework imports",
                                example_code="from flask import Flask\nfrom fastapi import FastAPI"),
                    DetailedStep(3, "Check route decorators",
                                "Verify your routes use standard framework decorators",
                                example_code="@app.route('/api/users')  # Flask\n@app.get('/api/users')  # FastAPI"),
                    DetailedStep(4, "Verify app instance",
                                "Check that your application instance is properly named",
                                example_code="app = Flask(__name__)  # Standard naming")
                ],
                related_documentation={
                    "Framework Support": "https://github.com/user/repo#framework-support",
                    "Flask Example": "https://github.com/user/repo/examples/flask_app.py",
                    "FastAPI Example": "https://github.com/user/repo/examples/fastapi_app.py"
                },
                prevention_tips=[
                    "Use conventional naming (app = Flask(__name__))",
                    "Keep framework imports at the top of your file",
                    "Use standard route decorators (@app.route, @app.get, etc.)",
                    "Avoid complex factory patterns that obscure detection"
                ]
            ),
            
            "file_not_found": ActionableError(
                category=ErrorCategory.FILE_ACCESS,
                original_error="",
                user_message="The specified file could not be found.",
                quick_fixes=[
                    QuickFix("Check file path", "Verify the file path is correct and the file exists"),
                    QuickFix("Use absolute path", "Try using an absolute path instead of relative"),
                    QuickFix("Check working directory", command="pwd  # Check current directory")
                ],
                detailed_steps=[
                    DetailedStep(1, "Verify file exists",
                                "Check that the file exists at the specified location",
                                command="ls -la {file_path}"),
                    DetailedStep(2, "Check permissions",
                                "Ensure you have read permissions for the file",
                                command="ls -la $(dirname {file_path})"),
                    DetailedStep(3, "Try absolute path",
                                "Use an absolute path to avoid relative path issues",
                                command="realpath {file_path}")
                ],
                related_documentation={
                    "Usage Examples": "https://github.com/user/repo#usage",
                    "CLI Reference": "https://github.com/user/repo#cli-reference"
                },
                prevention_tips=[
                    "Use absolute paths when possible",
                    "Verify file existence before running the tool",
                    "Check file permissions in restrictive environments"
                ]
            )
        }
    
    def handle_error(self, error: Exception, context: Dict = None) -> ActionableError:
        """Convert an exception into actionable guidance."""
        error_message = str(error)
        
        # Match error patterns
        for pattern, (category, template_key) in self.error_patterns.items():
            match = re.search(pattern, error_message)
            if match:
                template = self.guidance_templates.get(template_key)
                if template:
                    return self._customize_template(template, error_message, match, context)
        
        # Fallback for unknown errors
        return ActionableError(
            category=ErrorCategory.CONFIGURATION,
            original_error=error_message,
            user_message=f"An unexpected error occurred: {error_message}",
            quick_fixes=[
                QuickFix("Check logs", "Review the error logs for more details"),
                QuickFix("Try with verbose mode", command="--verbose")
            ],
            detailed_steps=[
                DetailedStep(1, "Enable verbose logging",
                           "Run the command with --verbose for more information"),
                DetailedStep(2, "Check system requirements",
                           "Ensure all system requirements are met"),
                DetailedStep(3, "Contact support",
                           "If the issue persists, please report it as a bug")
            ],
            related_documentation={
                "Troubleshooting": "https://github.com/user/repo#troubleshooting",
                "Issue Tracker": "https://github.com/user/repo/issues"
            },
            prevention_tips=[
                "Keep the tool updated to the latest version",
                "Use supported Python versions",
                "Test with simple examples first"
            ]
        )
    
    def _customize_template(self, template: ActionableError, error_message: str, 
                          match: re.Match, context: Dict = None) -> ActionableError:
        """Customize template with specific error details."""
        context = context or {}
        
        # Extract specific information from the match
        extracted_info = match.groups() if match else ()
        
        # Customize based on error category
        if template.category == ErrorCategory.DEPENDENCY_MISSING:
            dependency = extracted_info[0] if extracted_info else "unknown"
            template = self._customize_dependency_error(template, dependency)
        elif template.category == ErrorCategory.FILE_ACCESS:
            file_path = extracted_info[0] if extracted_info else "unknown"
            template = self._customize_file_error(template, file_path)
        
        # Set the original error
        template.original_error = error_message
        return template
    
    def _customize_dependency_error(self, template: ActionableError, dependency: str) -> ActionableError:
        """Customize dependency error with specific package name."""
        for fix in template.quick_fixes:
            if fix.command:
                fix.command = fix.command.format(dependency=dependency)
            if fix.example_code:
                fix.example_code = fix.example_code.format(dependency=dependency)
        
        for step in template.detailed_steps:
            if step.verification:
                step.verification = step.verification.format(dependency=dependency)
        
        return template
    
    def _customize_file_error(self, template: ActionableError, file_path: str) -> ActionableError:
        """Customize file error with specific file path."""
        for step in template.detailed_steps:
            if step.command:
                step.command = step.command.format(file_path=file_path)
        
        return template
    
    def format_error_message(self, actionable_error: ActionableError) -> str:
        """Format actionable error into user-friendly message."""
        output = []
        
        # Main error message
        output.append(f"❌ {actionable_error.user_message}")
        output.append("")
        
        # Quick fixes
        if actionable_error.quick_fixes:
            output.append("🚀 Quick Fixes:")
            for i, fix in enumerate(actionable_error.quick_fixes, 1):
                output.append(f"  {i}. {fix.description}")
                if fix.command:
                    output.append(f"     Command: {fix.command}")
                if fix.example_code:
                    output.append(f"     Example: {fix.example_code}")
            output.append("")
        
        # Detailed steps
        if actionable_error.detailed_steps:
            output.append("🔧 Detailed Steps:")
            for step in actionable_error.detailed_steps:
                output.append(f"  {step.step_number}. {step.title}")
                output.append(f"     {step.description}")
                if step.command:
                    output.append(f"     Command: {step.command}")
                if step.example_code:
                    output.append(f"     Example: {step.example_code}")
                if step.verification:
                    output.append(f"     Verify: {step.verification}")
            output.append("")
        
        # Documentation links
        if actionable_error.related_documentation:
            output.append("📚 Related Documentation:")
            for title, url in actionable_error.related_documentation.items():
                output.append(f"  • {title}: {url}")
            output.append("")
        
        # Prevention tips
        if actionable_error.prevention_tips:
            output.append("💡 Prevention Tips:")
            for tip in actionable_error.prevention_tips:
                output.append(f"  • {tip}")
        
        return "\n".join(output)


# Global error handler instance
error_handler = ActionableErrorHandler()


def handle_user_facing_error(error: Exception, context: Dict = None) -> str:
    """Convert exception to user-friendly actionable message."""
    actionable_error = error_handler.handle_error(error, context)
    return error_handler.format_error_message(actionable_error)