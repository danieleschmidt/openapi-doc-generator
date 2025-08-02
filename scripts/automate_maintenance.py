#!/usr/bin/env python3
"""
Automated maintenance script for OpenAPI Doc Generator.
Performs routine maintenance tasks to keep the project healthy.
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import shutil


class MaintenanceAutomator:
    """Automates common maintenance tasks."""
    
    def __init__(self, config_path: str = ".github/maintenance-config.json"):
        self.config_path = Path(config_path)
        self.results = []
        self.timestamp = datetime.utcnow().isoformat() + "Z"
        
    def run_all_maintenance(self) -> List[Dict[str, Any]]:
        """Run all maintenance tasks."""
        print("🔧 Starting automated maintenance...")
        
        tasks = [
            ("Clean build artifacts", self._clean_build_artifacts),
            ("Update dependencies", self._update_dependencies),
            ("Run security scans", self._run_security_scans),
            ("Check code quality", self._check_code_quality),
            ("Update documentation", self._update_documentation),
            ("Optimize Docker images", self._optimize_docker_images),
            ("Archive old logs", self._archive_old_logs),
            ("Check disk space", self._check_disk_space),
            ("Validate configurations", self._validate_configurations),
            ("Update metrics", self._update_metrics),
        ]
        
        for task_name, task_func in tasks:
            print(f"\n📋 {task_name}...")
            try:
                result = task_func()
                self.results.append({
                    "task": task_name,
                    "status": "success",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "details": result
                })
                print(f"✅ {task_name} completed successfully")
            except Exception as e:
                self.results.append({
                    "task": task_name,
                    "status": "failed",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "error": str(e)
                })
                print(f"❌ {task_name} failed: {e}")
        
        return self.results
    
    def _clean_build_artifacts(self) -> Dict[str, Any]:
        """Clean build artifacts and cache files."""
        artifacts_cleaned = 0
        space_freed = 0
        
        # Directories to clean
        clean_dirs = [
            "build/", "dist/", "*.egg-info/", "__pycache__/",
            ".pytest_cache/", ".mypy_cache/", ".ruff_cache/",
            "htmlcov/", ".coverage", ".tox/", ".nox/"
        ]
        
        for pattern in clean_dirs:
            for path in Path(".").glob(f"**/{pattern}"):
                if path.exists():
                    try:
                        if path.is_file():
                            size = path.stat().st_size
                            path.unlink()
                            space_freed += size
                        else:
                            size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                            shutil.rmtree(path)
                            space_freed += size
                        artifacts_cleaned += 1
                    except Exception:
                        pass
        
        # Clean pip cache
        try:
            subprocess.run(["pip", "cache", "purge"], capture_output=True)
            artifacts_cleaned += 1
        except:
            pass
            
        return {
            "artifacts_cleaned": artifacts_cleaned,
            "space_freed_mb": round(space_freed / (1024 * 1024), 2)
        }
    
    def _update_dependencies(self) -> Dict[str, Any]:
        """Update project dependencies."""
        updates_available = []
        updates_applied = []
        
        # Check for outdated packages
        try:
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                outdated = json.loads(result.stdout)
                updates_available = [pkg["name"] for pkg in outdated]
        except:
            pass
        
        # Update development dependencies (safely)
        try:
            result = subprocess.run(
                ["pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                updates_applied.extend(["pip", "setuptools", "wheel"])
        except:
            pass
        
        # Update pre-commit hooks
        try:
            result = subprocess.run(
                ["pre-commit", "autoupdate"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                updates_applied.append("pre-commit-hooks")
        except:
            pass
        
        return {
            "updates_available": len(updates_available),
            "updates_applied": len(updates_applied),
            "packages_updated": updates_applied
        }
    
    def _run_security_scans(self) -> Dict[str, Any]:
        """Run security scans and checks."""
        scan_results = {}
        
        # Bandit security scan
        try:
            result = subprocess.run(
                ["bandit", "-r", "src/", "-f", "json", "-o", "security_scan_maintenance.json"],
                capture_output=True, text=True
            )
            if Path("security_scan_maintenance.json").exists():
                scan_data = json.loads(Path("security_scan_maintenance.json").read_text())
                scan_results["bandit_issues"] = len(scan_data.get("results", []))
            else:
                scan_results["bandit_issues"] = 0
        except:
            scan_results["bandit_issues"] = "error"
        
        # Safety dependency check
        try:
            result = subprocess.run(
                ["safety", "check", "--json", "--output", "safety_scan_maintenance.json"],
                capture_output=True, text=True
            )
            if Path("safety_scan_maintenance.json").exists():
                scan_data = json.loads(Path("safety_scan_maintenance.json").read_text())
                scan_results["vulnerable_dependencies"] = len(scan_data)
            else:
                scan_results["vulnerable_dependencies"] = 0
        except:
            scan_results["vulnerable_dependencies"] = "error"
        
        # Check for secrets
        try:
            result = subprocess.run(
                ["detect-secrets", "scan", "--baseline", ".secrets.baseline"],
                capture_output=True, text=True
            )
            scan_results["secrets_scan"] = "passed" if result.returncode == 0 else "failed"
        except:
            scan_results["secrets_scan"] = "not_configured"
        
        return scan_results
    
    def _check_code_quality(self) -> Dict[str, Any]:
        """Check and fix code quality issues."""
        quality_results = {}
        
        # Run ruff linter
        try:
            result = subprocess.run(
                ["ruff", "check", "src/", "--output-format", "json"],
                capture_output=True, text=True
            )
            ruff_data = json.loads(result.stdout) if result.stdout else []
            quality_results["ruff_violations"] = len(ruff_data)
            
            # Try to fix auto-fixable issues
            subprocess.run(
                ["ruff", "check", "src/", "--fix"],
                capture_output=True
            )
        except:
            quality_results["ruff_violations"] = "error"
        
        # Run formatter
        try:
            result = subprocess.run(
                ["ruff", "format", "src/"],
                capture_output=True, text=True
            )
            quality_results["formatting"] = "applied" if result.returncode == 0 else "failed"
        except:
            quality_results["formatting"] = "error"
        
        # Type checking
        try:
            result = subprocess.run(
                ["mypy", "src/"],
                capture_output=True, text=True
            )
            type_errors = result.stderr.count("error:")
            quality_results["type_errors"] = type_errors
        except:
            quality_results["type_errors"] = "error"
        
        return quality_results
    
    def _update_documentation(self) -> Dict[str, Any]:
        """Update generated documentation."""
        doc_results = {}
        
        # Generate API documentation
        try:
            result = subprocess.run([
                "openapi-doc-generator",
                "--app", "examples/app.py",
                "--format", "markdown",
                "--output", "docs/generated/API.md"
            ], capture_output=True, text=True)
            doc_results["api_docs"] = "updated" if result.returncode == 0 else "failed"
        except:
            doc_results["api_docs"] = "error"
        
        # Update OpenAPI spec
        try:
            result = subprocess.run([
                "openapi-doc-generator",
                "--app", "examples/app.py",
                "--format", "openapi",
                "--output", "docs/generated/openapi.json"
            ], capture_output=True, text=True)
            doc_results["openapi_spec"] = "updated" if result.returncode == 0 else "failed"
        except:
            doc_results["openapi_spec"] = "error"
        
        # Check documentation links
        doc_files = list(Path(".").rglob("*.md"))
        broken_links = 0
        for doc_file in doc_files:
            try:
                content = doc_file.read_text()
                # Simple check for local file references
                import re
                links = re.findall(r'\[.*?\]\(([^)]+)\)', content)
                for link in links:
                    if link.startswith('./') or link.startswith('../'):
                        if not Path(doc_file.parent / link).exists():
                            broken_links += 1
            except:
                pass
        
        doc_results["broken_links"] = broken_links
        
        return doc_results
    
    def _optimize_docker_images(self) -> Dict[str, Any]:
        """Optimize Docker images and cleanup."""
        docker_results = {}
        
        # Clean up unused Docker images
        try:
            result = subprocess.run(
                ["docker", "image", "prune", "-f"],
                capture_output=True, text=True
            )
            docker_results["images_pruned"] = "yes" if result.returncode == 0 else "failed"
        except:
            docker_results["images_pruned"] = "error"
        
        # Clean up unused containers
        try:
            result = subprocess.run(
                ["docker", "container", "prune", "-f"],
                capture_output=True, text=True
            )
            docker_results["containers_pruned"] = "yes" if result.returncode == 0 else "failed"
        except:
            docker_results["containers_pruned"] = "error"
        
        # Check current image size
        try:
            result = subprocess.run(
                ["docker", "images", "openapi-doc-generator", "--format", "{{.Size}}"],
                capture_output=True, text=True
            )
            docker_results["current_image_size"] = result.stdout.strip() if result.stdout else "unknown"
        except:
            docker_results["current_image_size"] = "error"
        
        return docker_results
    
    def _archive_old_logs(self) -> Dict[str, Any]:
        """Archive or clean up old log files."""
        log_results = {}
        
        # Find log files older than 30 days
        cutoff_date = datetime.now() - timedelta(days=30)
        old_logs = []
        
        log_patterns = ["*.log", "logs/*.log", "*.log.*"]
        for pattern in log_patterns:
            for log_file in Path(".").glob(pattern):
                if log_file.is_file():
                    file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        old_logs.append(log_file)
        
        # Archive or remove old logs
        archived_count = 0
        for log_file in old_logs:
            try:
                # Move to archive directory or remove
                archive_dir = Path("logs/archive")
                archive_dir.mkdir(parents=True, exist_ok=True)
                
                archive_path = archive_dir / f"{log_file.name}.{cutoff_date.strftime('%Y%m%d')}"
                log_file.rename(archive_path)
                archived_count += 1
            except:
                pass
        
        log_results["old_logs_found"] = len(old_logs)
        log_results["logs_archived"] = archived_count
        
        return log_results
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        try:
            import shutil
            total, used, free = shutil.disk_usage(".")
            
            return {
                "total_gb": round(total / (1024**3), 2),
                "used_gb": round(used / (1024**3), 2),
                "free_gb": round(free / (1024**3), 2),
                "usage_percent": round((used / total) * 100, 1)
            }
        except:
            return {"error": "Could not check disk space"}
    
    def _validate_configurations(self) -> Dict[str, Any]:
        """Validate project configuration files."""
        config_results = {}
        
        # Validate pyproject.toml
        try:
            import tomllib
            pyproject_content = Path("pyproject.toml").read_text()
            tomllib.loads(pyproject_content)
            config_results["pyproject_toml"] = "valid"
        except Exception as e:
            config_results["pyproject_toml"] = f"invalid: {e}"
        
        # Validate docker-compose.yml
        try:
            import yaml
            if Path("docker-compose.yml").exists():
                compose_content = Path("docker-compose.yml").read_text()
                yaml.safe_load(compose_content)
                config_results["docker_compose"] = "valid"
            else:
                config_results["docker_compose"] = "not_found"
        except Exception as e:
            config_results["docker_compose"] = f"invalid: {e}"
        
        # Validate GitHub workflows
        workflow_errors = 0
        workflows_dir = Path(".github/workflows")
        if workflows_dir.exists():
            for workflow_file in workflows_dir.glob("*.yml"):
                try:
                    import yaml
                    workflow_content = workflow_file.read_text()
                    yaml.safe_load(workflow_content)
                except:
                    workflow_errors += 1
        
        config_results["github_workflows"] = f"{workflow_errors} errors" if workflow_errors > 0 else "valid"
        
        return config_results
    
    def _update_metrics(self) -> Dict[str, Any]:
        """Update project metrics."""
        try:
            # Run metrics collection
            result = subprocess.run(
                ["python", "scripts/collect_metrics.py", "--output", "metrics_maintenance.json"],
                capture_output=True, text=True
            )
            
            if result.returncode == 0 and Path("metrics_maintenance.json").exists():
                return {"metrics_updated": True, "file": "metrics_maintenance.json"}
            else:
                return {"metrics_updated": False, "error": result.stderr}
        except Exception as e:
            return {"metrics_updated": False, "error": str(e)}
    
    def generate_maintenance_report(self) -> str:
        """Generate a maintenance report."""
        successful_tasks = [r for r in self.results if r["status"] == "success"]
        failed_tasks = [r for r in self.results if r["status"] == "failed"]
        
        report = f"""# 🔧 Automated Maintenance Report

**Generated:** {self.timestamp}
**Total Tasks:** {len(self.results)}
**Successful:** {len(successful_tasks)}
**Failed:** {len(failed_tasks)}

## ✅ Successful Tasks

"""
        
        for task in successful_tasks:
            report += f"- **{task['task']}** ({task['timestamp']})\n"
            if task.get('details'):
                details = task['details']
                if isinstance(details, dict):
                    for key, value in details.items():
                        report += f"  - {key}: {value}\n"
                report += "\n"
        
        if failed_tasks:
            report += "## ❌ Failed Tasks\n\n"
            for task in failed_tasks:
                report += f"- **{task['task']}** ({task['timestamp']})\n"
                report += f"  - Error: {task.get('error', 'Unknown error')}\n\n"
        
        report += """## 📋 Recommended Actions

1. Review failed tasks and address any issues
2. Update dependencies that couldn't be automatically updated
3. Check security scan results and fix any vulnerabilities
4. Review disk space usage if approaching limits
5. Schedule next maintenance run

---
*This report was generated automatically by the maintenance system.*
"""
        
        return report
    
    def save_results(self, output_file: str = None) -> str:
        """Save maintenance results to file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"maintenance_results_{timestamp}.json"
        
        output_path = Path(output_file)
        results_data = {
            "maintenance_run": {
                "timestamp": self.timestamp,
                "total_tasks": len(self.results),
                "successful_tasks": len([r for r in self.results if r["status"] == "success"]),
                "failed_tasks": len([r for r in self.results if r["status"] == "failed"])
            },
            "tasks": self.results
        }
        
        output_path.write_text(json.dumps(results_data, indent=2))
        print(f"💾 Maintenance results saved to {output_path}")
        return str(output_path)


def main():
    """Main function for maintenance automation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run automated maintenance tasks")
    parser.add_argument("--output", "-o", help="Output file for results JSON")
    parser.add_argument("--report", "-r", action="store_true", help="Generate maintenance report")
    parser.add_argument("--task", "-t", help="Run specific task only")
    
    args = parser.parse_args()
    
    try:
        automator = MaintenanceAutomator()
        
        if args.task:
            # Run specific task
            task_methods = {
                "clean": automator._clean_build_artifacts,
                "dependencies": automator._update_dependencies,
                "security": automator._run_security_scans,
                "quality": automator._check_code_quality,
                "docs": automator._update_documentation,
                "docker": automator._optimize_docker_images,
                "logs": automator._archive_old_logs,
                "disk": automator._check_disk_space,
                "config": automator._validate_configurations,
                "metrics": automator._update_metrics,
            }
            
            if args.task in task_methods:
                result = task_methods[args.task]()
                print(f"✅ Task '{args.task}' completed")
                print(json.dumps(result, indent=2))
            else:
                print(f"❌ Unknown task: {args.task}")
                print(f"Available tasks: {', '.join(task_methods.keys())}")
                return 1
        else:
            # Run all maintenance tasks
            results = automator.run_all_maintenance()
            
            # Save results
            output_file = automator.save_results(args.output)
            
            # Generate report if requested
            if args.report:
                report = automator.generate_maintenance_report()
                report_file = output_file.replace(".json", "_report.md")
                Path(report_file).write_text(report)
                print(f"📄 Maintenance report saved to {report_file}")
            
            successful = len([r for r in results if r["status"] == "success"])
            total = len(results)
            print(f"\n🎯 Maintenance completed: {successful}/{total} tasks successful")
        
        return 0
        
    except Exception as e:
        print(f"❌ Maintenance failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())