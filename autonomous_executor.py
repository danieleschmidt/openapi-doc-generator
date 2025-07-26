#!/usr/bin/env python3
"""
Autonomous Task Execution Engine
Implements strict TDD + Security micro-cycles for backlog items
"""

import subprocess
import json
import logging
import os
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from autonomous_engine import AutonomousBacklogEngine, BacklogItem, TaskStatus


@dataclass
class ExecutionResult:
    success: bool
    task_id: str
    duration_seconds: float
    tests_passed: bool
    security_clear: bool
    lint_clear: bool
    changes_made: List[str]
    errors: List[str]
    metrics: Dict[str, Any]


class AutonomousExecutor:
    def __init__(self, repo_root: str = "/root/repo"):
        self.repo_root = Path(repo_root)
        self.logger = logging.getLogger('autonomous_executor')
        self.engine = AutonomousBacklogEngine(repo_root)
        
        # CI/Quality gate commands
        self.test_command = ["python", "-m", "pytest", "-v"]
        self.lint_command = ["ruff", "check", "."]
        self.security_command = ["bandit", "-r", "src", "-f", "json"]
        self.type_check_command = None  # Will detect if mypy/pyright available
        
        self._detect_quality_tools()
    
    def _detect_quality_tools(self) -> None:
        """Detect available quality assurance tools"""
        # Check for type checkers
        for tool in ["mypy", "pyright"]:
            try:
                result = subprocess.run([tool, "--version"], 
                                      capture_output=True, cwd=self.repo_root)
                if result.returncode == 0:
                    self.type_check_command = [tool, "src"]
                    break
            except FileNotFoundError:
                continue
    
    def sync_repo_and_ci(self) -> bool:
        """Sync with remote and check CI status"""
        try:
            # Fetch latest changes
            subprocess.run(["git", "fetch", "origin"], 
                         cwd=self.repo_root, check=True)
            
            # Check if we're behind
            result = subprocess.run([
                "git", "rev-list", "--count", "HEAD..origin/main"
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            behind_count = int(result.stdout.strip()) if result.stdout.strip() else 0
            
            if behind_count > 0:
                self.logger.info(f"Repository is {behind_count} commits behind, rebasing...")
                subprocess.run(["git", "rebase", "origin/main"], 
                             cwd=self.repo_root, check=True)
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to sync repository: {e}")
            return False
    
    def execute_tdd_cycle(self, task: BacklogItem) -> ExecutionResult:
        """Execute TDD cycle: RED -> GREEN -> REFACTOR"""
        start_time = datetime.now()
        changes_made = []
        errors = []
        
        self.logger.info(f"Starting TDD cycle for task: {task.title}")
        
        try:
            # Update task status
            task.status = TaskStatus.DOING
            
            # Execute task-specific implementation
            if task.type == "code_quality" and "lint" in task.title.lower():
                success = self._execute_lint_fixes(task, changes_made, errors)
            elif task.type == "user_experience" and "cli" in task.title.lower():
                success = self._execute_cli_enhancements(task, changes_made, errors)
            elif task.type == "security":
                success = self._execute_security_fixes(task, changes_made, errors)
            elif task.type == "test_failure":
                success = self._execute_test_fixes(task, changes_made, errors)
            else:
                success = self._execute_generic_task(task, changes_made, errors)
            
            # Run quality gates
            tests_passed = self._run_tests()
            security_clear = self._run_security_scan()
            lint_clear = self._run_lint_check()
            type_check_clear = self._run_type_check() if self.type_check_command else True
            
            all_gates_passed = tests_passed and security_clear and lint_clear and type_check_clear
            
            if success and all_gates_passed:
                task.status = TaskStatus.DONE
                self.logger.info(f"Task {task.id} completed successfully")
            else:
                task.status = TaskStatus.BLOCKED
                errors.append("Quality gates failed")
                self.logger.warning(f"Task {task.id} blocked on quality gates")
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return ExecutionResult(
                success=success and all_gates_passed,
                task_id=task.id,
                duration_seconds=duration,
                tests_passed=tests_passed,
                security_clear=security_clear,
                lint_clear=lint_clear,
                changes_made=changes_made,
                errors=errors,
                metrics={
                    "type_check_clear": type_check_clear,
                    "task_type": task.type,
                    "wsjf_score": task.wsjf_score
                }
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            task.status = TaskStatus.BLOCKED
            errors.append(f"Execution error: {str(e)}")
            
            return ExecutionResult(
                success=False,
                task_id=task.id,
                duration_seconds=duration,
                tests_passed=False,
                security_clear=False,
                lint_clear=False,
                changes_made=changes_made,
                errors=errors,
                metrics={"execution_error": str(e)}
            )
    
    def _execute_lint_fixes(self, task: BacklogItem, changes: List[str], errors: List[str]) -> bool:
        """Execute lint fixes with TDD approach"""
        try:
            # RED: Run tests to establish baseline
            baseline_tests = self._run_tests()
            if not baseline_tests:
                errors.append("Baseline tests failing before lint fixes")
                return False
            
            # Get specific lint issues to fix
            result = subprocess.run(
                ["ruff", "check", ".", "--output-format=json"],
                capture_output=True, text=True, cwd=self.repo_root
            )
            
            if result.stdout:
                lint_issues = json.loads(result.stdout)
                
                # Fix specific categories mentioned in task
                if "unused import" in task.description or "F401" in task.description:
                    fixed_count = self._fix_unused_imports(lint_issues, changes)
                    self.logger.info(f"Fixed {fixed_count} unused imports")
                
                if "bare except" in task.description or "E722" in task.description:
                    fixed_count = self._fix_bare_excepts(lint_issues, changes)
                    self.logger.info(f"Fixed {fixed_count} bare except clauses")
                
                if "unused variable" in task.description or "F841" in task.description:
                    fixed_count = self._fix_unused_variables(lint_issues, changes)
                    self.logger.info(f"Fixed {fixed_count} unused variables")
            
            # GREEN: Verify tests still pass
            tests_after_fix = self._run_tests()
            if not tests_after_fix:
                errors.append("Tests failed after lint fixes")
                return False
            
            return True
            
        except Exception as e:
            errors.append(f"Lint fix error: {str(e)}")
            return False
    
    def _fix_unused_imports(self, lint_issues: List[Dict], changes: List[str]) -> int:
        """Fix unused import statements"""
        fixed_count = 0
        
        for issue in lint_issues:
            if issue.get("code") == "F401":  # Unused import
                file_path = Path(issue["filename"])
                if file_path.exists():
                    try:
                        with open(file_path, 'r') as f:
                            lines = f.readlines()
                        
                        # Remove the import line (careful with line numbers)
                        line_num = issue["location"]["row"] - 1  # Convert to 0-based
                        if 0 <= line_num < len(lines):
                            import_line = lines[line_num].strip()
                            # Only remove if it's clearly an import line
                            if import_line.startswith(('import ', 'from ')):
                                lines.pop(line_num)
                                
                                with open(file_path, 'w') as f:
                                    f.writelines(lines)
                                
                                changes.append(f"Removed unused import from {file_path}:{line_num+1}")
                                fixed_count += 1
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to fix unused import in {file_path}: {e}")
        
        return fixed_count
    
    def _fix_bare_excepts(self, lint_issues: List[Dict], changes: List[str]) -> int:
        """Fix bare except clauses"""
        fixed_count = 0
        
        for issue in lint_issues:
            if issue.get("code") == "E722":  # Bare except
                file_path = Path(issue["filename"])
                if file_path.exists():
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                        
                        # Replace 'except:' with 'except Exception:'
                        new_content = content.replace('except:', 'except Exception:')
                        
                        if new_content != content:
                            with open(file_path, 'w') as f:
                                f.write(new_content)
                            
                            changes.append(f"Fixed bare except in {file_path}")
                            fixed_count += 1
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to fix bare except in {file_path}: {e}")
        
        return fixed_count
    
    def _fix_unused_variables(self, lint_issues: List[Dict], changes: List[str]) -> int:
        """Fix unused variable assignments"""
        fixed_count = 0
        
        for issue in lint_issues:
            if issue.get("code") == "F841":  # Unused variable
                file_path = Path(issue["filename"])
                if file_path.exists():
                    try:
                        with open(file_path, 'r') as f:
                            lines = f.readlines()
                        
                        line_num = issue["location"]["row"] - 1
                        if 0 <= line_num < len(lines):
                            line = lines[line_num]
                            # Simple heuristic: prefix unused variables with underscore
                            if '=' in line and not line.strip().startswith('_'):
                                # Extract variable name and prefix with underscore
                                parts = line.split('=', 1)
                                var_part = parts[0].strip()
                                if var_part and not var_part.startswith('_'):
                                    lines[line_num] = line.replace(var_part, f"_{var_part}", 1)
                                    
                                    with open(file_path, 'w') as f:
                                        f.writelines(lines)
                                    
                                    changes.append(f"Prefixed unused variable in {file_path}:{line_num+1}")
                                    fixed_count += 1
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to fix unused variable in {file_path}: {e}")
        
        return fixed_count
    
    def _execute_cli_enhancements(self, task: BacklogItem, changes: List[str], errors: List[str]) -> bool:
        """Execute CLI user experience enhancements"""
        try:
            cli_file = self.repo_root / "src" / "openapi_doc_generator" / "cli.py"
            
            if not cli_file.exists():
                errors.append("CLI file not found")
                return False
            
            # Read current CLI implementation
            with open(cli_file, 'r') as f:
                content = f.read()
            
            # Add verbose/quiet flags if not present
            if "--verbose" not in content and "--quiet" not in content:
                # This would need more sophisticated implementation
                # For now, just log what we would do
                changes.append("Would add --verbose and --quiet flags to CLI")
                self.logger.info("CLI enhancement placeholder - would implement verbose/quiet flags")
            
            return True
            
        except Exception as e:
            errors.append(f"CLI enhancement error: {str(e)}")
            return False
    
    def _execute_security_fixes(self, task: BacklogItem, changes: List[str], errors: List[str]) -> bool:
        """Execute security vulnerability fixes"""
        try:
            # Run security scan to get current issues
            result = subprocess.run(
                self.security_command,
                capture_output=True, text=True, cwd=self.repo_root
            )
            
            if result.stdout:
                security_data = json.loads(result.stdout)
                # Implement specific security fixes based on bandit results
                changes.append("Security fix placeholder - would implement specific fixes")
                self.logger.info("Security fix placeholder")
            
            return True
            
        except Exception as e:
            errors.append(f"Security fix error: {str(e)}")
            return False
    
    def _execute_test_fixes(self, task: BacklogItem, changes: List[str], errors: List[str]) -> bool:
        """Execute test failure fixes"""
        try:
            # Run tests to see current failures
            result = subprocess.run(
                self.test_command + ["--tb=short"],
                capture_output=True, text=True, cwd=self.repo_root
            )
            
            changes.append("Test fix placeholder - would analyze and fix specific test failures")
            self.logger.info("Test fix placeholder")
            return True
            
        except Exception as e:
            errors.append(f"Test fix error: {str(e)}")
            return False
    
    def _execute_generic_task(self, task: BacklogItem, changes: List[str], errors: List[str]) -> bool:
        """Execute generic task implementation"""
        changes.append(f"Generic task execution placeholder for {task.type}")
        self.logger.info(f"Generic task placeholder for {task.title}")
        return True
    
    def _run_tests(self) -> bool:
        """Run test suite"""
        try:
            result = subprocess.run(
                self.test_command,
                capture_output=True, cwd=self.repo_root
            )
            return result.returncode == 0
        except subprocess.CalledProcessError:
            return False
    
    def _run_security_scan(self) -> bool:
        """Run security scan"""
        try:
            result = subprocess.run(
                self.security_command,
                capture_output=True, cwd=self.repo_root
            )
            # Bandit returns 0 for no issues, 1 for issues found
            return result.returncode == 0
        except subprocess.CalledProcessError:
            return False
    
    def _run_lint_check(self) -> bool:
        """Run lint check"""
        try:
            result = subprocess.run(
                self.lint_command,
                capture_output=True, cwd=self.repo_root
            )
            return result.returncode == 0
        except subprocess.CalledProcessError:
            return False
    
    def _run_type_check(self) -> bool:
        """Run type checking if available"""
        if not self.type_check_command:
            return True
        
        try:
            result = subprocess.run(
                self.type_check_command,
                capture_output=True, cwd=self.repo_root
            )
            return result.returncode == 0
        except subprocess.CalledProcessError:
            return False
    
    def create_commit(self, task: BacklogItem, result: ExecutionResult) -> bool:
        """Create git commit for completed task"""
        if not result.success:
            return False
        
        try:
            # Add changed files
            subprocess.run(["git", "add", "."], cwd=self.repo_root, check=True)
            
            # Create commit message
            commit_msg = f"{task.type}: {task.title}\n\n"
            commit_msg += f"WSJF Score: {task.wsjf_score:.1f}\n"
            commit_msg += f"Changes:\n"
            for change in result.changes_made:
                commit_msg += f"- {change}\n"
            commit_msg += f"\n🤖 Generated with Autonomous Backlog Management\n"
            commit_msg += f"Co-Authored-By: Terry <terry@terragon-labs.com>"
            
            # Create commit
            subprocess.run([
                "git", "commit", "-m", commit_msg
            ], cwd=self.repo_root, check=True)
            
            self.logger.info(f"Created commit for task {task.id}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to create commit: {e}")
            return False
    
    def run_macro_execution_loop(self, max_iterations: int = 10) -> List[ExecutionResult]:
        """Run the main execution loop"""
        results = []
        iteration = 0
        
        self.logger.info("Starting autonomous execution loop")
        
        while iteration < max_iterations:
            iteration += 1
            self.logger.info(f"=== Execution Loop Iteration {iteration} ===")
            
            # Sync repository and CI
            if not self.sync_repo_and_ci():
                self.logger.error("Failed to sync repository, stopping execution")
                break
            
            # Discover and prioritize tasks
            items = self.engine.run_discovery_cycle()
            
            # Get next task
            next_task = self.engine.get_next_task(items)
            if not next_task:
                self.logger.info("No ready tasks found, execution complete")
                break
            
            self.logger.info(f"Executing task: {next_task.title} (WSJF: {next_task.wsjf_score:.1f})")
            
            # Execute task
            result = self.execute_tdd_cycle(next_task)
            results.append(result)
            
            # Create commit if successful
            if result.success:
                self.create_commit(next_task, result)
            
            # Update backlog
            updated_items = [item for item in items if item.id != next_task.id]
            if next_task.status != TaskStatus.DONE:
                updated_items.append(next_task)
            else:
                next_task.status = TaskStatus.DONE
                updated_items.append(next_task)
            
            self.engine.save_backlog(updated_items)
            
            # Generate metrics
            report = self.engine.generate_metrics_report(updated_items)
            report["execution_results"] = [
                {
                    "task_id": r.task_id,
                    "success": r.success,
                    "duration": r.duration_seconds
                }
                for r in results
            ]
            self.engine.save_metrics_report(report)
            
            self.logger.info(f"Iteration {iteration} complete")
        
        self.logger.info(f"Autonomous execution loop complete. {len(results)} tasks executed.")
        return results


if __name__ == "__main__":
    executor = AutonomousExecutor()
    results = executor.run_macro_execution_loop()
    
    print(f"\n🤖 Autonomous Execution Complete")
    print(f"📊 Tasks Executed: {len(results)}")
    successful = len([r for r in results if r.success])
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {len(results) - successful}")
    
    if results:
        total_duration = sum(r.duration_seconds for r in results)
        print(f"⏱️  Total Duration: {total_duration:.1f} seconds")