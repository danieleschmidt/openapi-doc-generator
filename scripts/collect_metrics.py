#!/usr/bin/env python3
"""
Comprehensive metrics collection script for OpenAPI Doc Generator.
Gathers development, operational, and business metrics for reporting.
"""

import json
import subprocess
import sys
import time
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
import requests


class MetricsCollector:
    """Collects various metrics about the project."""
    
    def __init__(self, config_path: str = ".github/project-metrics.json"):
        self.config_path = Path(config_path)
        self.metrics = {}
        self.timestamp = datetime.utcnow().isoformat() + "Z"
        
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all available metrics."""
        print("🔍 Collecting project metrics...")
        
        self.metrics = {
            "collection_timestamp": self.timestamp,
            "repository": self._collect_repository_metrics(),
            "code_quality": self._collect_code_quality_metrics(),
            "security": self._collect_security_metrics(),
            "performance": self._collect_performance_metrics(),
            "testing": self._collect_testing_metrics(),
            "deployment": self._collect_deployment_metrics(),
            "automation": self._collect_automation_metrics(),
            "dependencies": self._collect_dependency_metrics(),
            "git": self._collect_git_metrics(),
            "ci_cd": self._collect_ci_cd_metrics(),
            "documentation": self._collect_documentation_metrics(),
            "business": self._collect_business_metrics(),
        }
        
        # Calculate overall health score
        self.metrics["health_score"] = self._calculate_health_score()
        
        return self.metrics
    
    def _collect_repository_metrics(self) -> Dict[str, Any]:
        """Collect repository-level metrics."""
        try:
            # Get repository size and file counts
            result = subprocess.run(
                ["find", ".", "-type", "f", "-not", "-path", "./.git/*"],
                capture_output=True, text=True
            )
            total_files = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            
            # Get lines of code
            result = subprocess.run(
                ["find", ".", "-name", "*.py", "-not", "-path", "./.git/*", "-exec", "wc", "-l", "{}", "+"],
                capture_output=True, text=True
            )
            total_lines = sum(int(line.split()[0]) for line in result.stdout.strip().split('\n') 
                            if line.strip() and line.split()[0].isdigit()) if result.stdout.strip() else 0
            
            return {
                "total_files": total_files,
                "total_lines_of_code": total_lines,
                "primary_language": "Python",
                "languages": ["Python", "YAML", "Dockerfile", "Shell"],
                "size_mb": self._get_directory_size(),
                "structure": self._analyze_directory_structure()
            }
        except Exception as e:
            print(f"⚠️  Warning: Could not collect repository metrics: {e}")
            return {}
    
    def _collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        try:
            metrics = {}
            
            # Run ruff for code quality
            try:
                result = subprocess.run(
                    ["ruff", "check", "src/", "--output-format", "json"],
                    capture_output=True, text=True
                )
                ruff_data = json.loads(result.stdout) if result.stdout else []
                metrics["ruff_violations"] = len(ruff_data)
                metrics["ruff_severity"] = self._analyze_ruff_severity(ruff_data)
            except:
                metrics["ruff_violations"] = 0
                
            # Run complexity analysis
            try:
                result = subprocess.run(
                    ["radon", "cc", "src/", "-j"],
                    capture_output=True, text=True
                )
                complexity_data = json.loads(result.stdout) if result.stdout else {}
                metrics["cyclomatic_complexity"] = self._analyze_complexity(complexity_data)
            except:
                metrics["cyclomatic_complexity"] = {"average": 0, "max": 0}
                
            # Type annotation coverage
            try:
                result = subprocess.run(
                    ["mypy", "src/", "--strict", "--no-error-summary"],
                    capture_output=True, text=True
                )
                metrics["type_annotation_coverage"] = self._analyze_mypy_output(result.stderr)
            except:
                metrics["type_annotation_coverage"] = 0
                
            return metrics
        except Exception as e:
            print(f"⚠️  Warning: Could not collect code quality metrics: {e}")
            return {}
    
    def _collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security-related metrics."""
        try:
            metrics = {}
            
            # Bandit security scan
            try:
                result = subprocess.run(
                    ["bandit", "-r", "src/", "-f", "json"],
                    capture_output=True, text=True
                )
                bandit_data = json.loads(result.stdout) if result.stdout else {}
                metrics["security_issues"] = len(bandit_data.get("results", []))
                metrics["security_severity"] = self._analyze_bandit_severity(bandit_data)
            except:
                metrics["security_issues"] = 0
                
            # Safety dependency check
            try:
                result = subprocess.run(
                    ["safety", "check", "--json"],
                    capture_output=True, text=True
                )
                safety_data = json.loads(result.stdout) if result.stdout else []
                metrics["vulnerable_dependencies"] = len(safety_data)
            except:
                metrics["vulnerable_dependencies"] = 0
                
            # Check for secrets
            if Path(".secrets.baseline").exists():
                metrics["secrets_baseline_exists"] = True
                metrics["secrets_scan_enabled"] = True
            else:
                metrics["secrets_baseline_exists"] = False
                metrics["secrets_scan_enabled"] = False
                
            return metrics
        except Exception as e:
            print(f"⚠️  Warning: Could not collect security metrics: {e}")
            return {}
    
    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance-related metrics."""
        try:
            metrics = {}
            
            # Build time measurement
            start_time = time.time()
            try:
                subprocess.run(
                    ["python", "-m", "build", "--wheel"],
                    capture_output=True, timeout=300
                )
                build_time = time.time() - start_time
                metrics["build_time_seconds"] = round(build_time, 2)
            except:
                metrics["build_time_seconds"] = 0
                
            # Test execution time
            start_time = time.time()
            try:
                subprocess.run(
                    ["pytest", "tests/", "-q", "--tb=no"],
                    capture_output=True, timeout=300
                )
                test_time = time.time() - start_time
                metrics["test_execution_time_seconds"] = round(test_time, 2)
            except:
                metrics["test_execution_time_seconds"] = 0
                
            # Memory usage during operation
            process = psutil.Process()
            metrics["memory_usage_mb"] = round(process.memory_info().rss / 1024 / 1024, 2)
            metrics["cpu_percent"] = psutil.cpu_percent(interval=1)
            
            return metrics
        except Exception as e:
            print(f"⚠️  Warning: Could not collect performance metrics: {e}")
            return {}
    
    def _collect_testing_metrics(self) -> Dict[str, Any]:
        """Collect testing-related metrics."""
        try:
            metrics = {}
            
            # Test coverage
            try:
                result = subprocess.run(
                    ["coverage", "run", "-m", "pytest", "tests/"],
                    capture_output=True, text=True
                )
                result = subprocess.run(
                    ["coverage", "report", "--format=json"],
                    capture_output=True, text=True
                )
                coverage_data = json.loads(result.stdout) if result.stdout else {}
                metrics["test_coverage_percent"] = coverage_data.get("totals", {}).get("percent_covered", 0)
                metrics["lines_covered"] = coverage_data.get("totals", {}).get("covered_lines", 0)
                metrics["lines_total"] = coverage_data.get("totals", {}).get("num_statements", 0)
            except:
                metrics["test_coverage_percent"] = 0
                
            # Count test files and test functions
            test_files = list(Path("tests").rglob("test_*.py"))
            metrics["test_files_count"] = len(test_files)
            
            # Count total test functions
            total_tests = 0
            for test_file in test_files:
                try:
                    content = test_file.read_text()
                    total_tests += content.count("def test_")
                except:
                    pass
            metrics["total_test_functions"] = total_tests
            
            return metrics
        except Exception as e:
            print(f"⚠️  Warning: Could not collect testing metrics: {e}")
            return {}
    
    def _collect_deployment_metrics(self) -> Dict[str, Any]:
        """Collect deployment-related metrics."""
        try:
            metrics = {}
            
            # Docker image size
            try:
                result = subprocess.run(
                    ["docker", "images", "openapi-doc-generator", "--format", "{{.Size}}"],
                    capture_output=True, text=True
                )
                image_size = result.stdout.strip() if result.stdout else "Unknown"
                metrics["docker_image_size"] = image_size
            except:
                metrics["docker_image_size"] = "Unknown"
                
            # Container build time
            start_time = time.time()
            try:
                subprocess.run(
                    ["docker", "build", "-t", "openapi-doc-generator:metrics", "."],
                    capture_output=True, timeout=600
                )
                build_time = time.time() - start_time
                metrics["container_build_time_seconds"] = round(build_time, 2)
            except:
                metrics["container_build_time_seconds"] = 0
                
            # Check for deployment files
            deployment_files = [
                "Dockerfile", "docker-compose.yml", "Makefile",
                ".github/workflows", "k8s/", "helm/"
            ]
            metrics["deployment_artifacts"] = {
                file: Path(file).exists() for file in deployment_files
            }
            
            return metrics
        except Exception as e:
            print(f"⚠️  Warning: Could not collect deployment metrics: {e}")
            return {}
    
    def _collect_automation_metrics(self) -> Dict[str, Any]:
        """Collect automation-related metrics."""
        try:
            metrics = {}
            
            # Pre-commit hooks
            metrics["pre_commit_enabled"] = Path(".pre-commit-config.yaml").exists()
            
            # GitHub Actions workflows
            workflows_dir = Path(".github/workflows")
            if workflows_dir.exists():
                workflow_files = list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))
                metrics["github_workflows_count"] = len(workflow_files)
                metrics["workflow_files"] = [f.name for f in workflow_files]
            else:
                metrics["github_workflows_count"] = 0
                metrics["workflow_files"] = []
                
            # Automation scripts
            scripts_dir = Path("scripts")
            if scripts_dir.exists():
                script_files = list(scripts_dir.glob("*.py")) + list(scripts_dir.glob("*.sh"))
                metrics["automation_scripts_count"] = len(script_files)
            else:
                metrics["automation_scripts_count"] = 0
                
            # Makefile targets
            if Path("Makefile").exists():
                makefile_content = Path("Makefile").read_text()
                targets = [line.split(':')[0] for line in makefile_content.split('\n') 
                          if ':' in line and not line.startswith('\t') and not line.startswith('#')]
                metrics["makefile_targets"] = len(targets)
            else:
                metrics["makefile_targets"] = 0
                
            return metrics
        except Exception as e:
            print(f"⚠️  Warning: Could not collect automation metrics: {e}")
            return {}
    
    def _collect_dependency_metrics(self) -> Dict[str, Any]:
        """Collect dependency-related metrics."""
        try:
            metrics = {}
            
            # Parse pyproject.toml for dependencies
            if Path("pyproject.toml").exists():
                import tomllib
                pyproject_content = Path("pyproject.toml").read_text()
                pyproject_data = tomllib.loads(pyproject_content)
                
                dependencies = pyproject_data.get("project", {}).get("dependencies", [])
                dev_dependencies = pyproject_data.get("project", {}).get("optional-dependencies", {}).get("dev", [])
                
                metrics["production_dependencies"] = len(dependencies)
                metrics["dev_dependencies"] = len(dev_dependencies)
                metrics["total_dependencies"] = len(dependencies) + len(dev_dependencies)
            else:
                metrics["production_dependencies"] = 0
                metrics["dev_dependencies"] = 0
                metrics["total_dependencies"] = 0
                
            # Check for dependency lock files
            lock_files = ["requirements.txt", "requirements-dev.txt", "poetry.lock", "Pipfile.lock"]
            metrics["lock_files"] = {file: Path(file).exists() for file in lock_files}
            
            return metrics
        except Exception as e:
            print(f"⚠️  Warning: Could not collect dependency metrics: {e}")
            return {}
    
    def _collect_git_metrics(self) -> Dict[str, Any]:
        """Collect Git repository metrics."""
        try:
            metrics = {}
            
            # Commit count
            result = subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                capture_output=True, text=True
            )
            metrics["total_commits"] = int(result.stdout.strip()) if result.stdout.strip() else 0
            
            # Contributors
            result = subprocess.run(
                ["git", "shortlog", "-sn", "--all"],
                capture_output=True, text=True
            )
            contributors = result.stdout.strip().split('\n') if result.stdout.strip() else []
            metrics["contributor_count"] = len(contributors)
            
            # Recent activity (last 30 days)
            thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            result = subprocess.run(
                ["git", "rev-list", "--count", f"--since={thirty_days_ago}", "HEAD"],
                capture_output=True, text=True
            )
            metrics["commits_last_30_days"] = int(result.stdout.strip()) if result.stdout.strip() else 0
            
            # Branch information
            result = subprocess.run(
                ["git", "branch", "-r"],
                capture_output=True, text=True
            )
            remote_branches = result.stdout.strip().split('\n') if result.stdout.strip() else []
            metrics["remote_branches"] = len([b for b in remote_branches if b.strip() and not 'HEAD' in b])
            
            return metrics
        except Exception as e:
            print(f"⚠️  Warning: Could not collect git metrics: {e}")
            return {}
    
    def _collect_ci_cd_metrics(self) -> Dict[str, Any]:
        """Collect CI/CD pipeline metrics."""
        try:
            metrics = {}
            
            # GitHub Actions workflow analysis
            workflows_dir = Path(".github/workflows")
            if workflows_dir.exists():
                workflows = []
                for workflow_file in workflows_dir.glob("*.yml"):
                    try:
                        import yaml
                        content = workflow_file.read_text()
                        workflow_data = yaml.safe_load(content)
                        workflows.append({
                            "name": workflow_data.get("name", workflow_file.stem),
                            "triggers": list(workflow_data.get("on", {}).keys()),
                            "jobs": len(workflow_data.get("jobs", {}))
                        })
                    except:
                        pass
                metrics["workflows"] = workflows
                
            # CI/CD features
            features = {
                "continuous_integration": any("push" in w.get("triggers", []) for w in workflows),
                "continuous_deployment": any("release" in str(w) for w in workflows),
                "automated_testing": Path("pytest.ini").exists() or Path("pyproject.toml").exists(),
                "security_scanning": any("security" in str(w) for w in workflows),
                "dependency_updates": any("dependabot" in str(w) for w in workflows),
            }
            metrics["features"] = features
            
            return metrics
        except Exception as e:
            print(f"⚠️  Warning: Could not collect CI/CD metrics: {e}")
            return {}
    
    def _collect_documentation_metrics(self) -> Dict[str, Any]:
        """Collect documentation-related metrics."""
        try:
            metrics = {}
            
            # Documentation files
            doc_files = []
            for pattern in ["*.md", "*.rst", "docs/**/*.md", "docs/**/*.rst"]:
                doc_files.extend(Path(".").glob(pattern))
            
            metrics["documentation_files"] = len(doc_files)
            
            # README analysis
            if Path("README.md").exists():
                readme_content = Path("README.md").read_text()
                metrics["readme_length"] = len(readme_content)
                metrics["readme_sections"] = readme_content.count("#")
            else:
                metrics["readme_length"] = 0
                metrics["readme_sections"] = 0
                
            # API documentation
            api_docs = list(Path(".").glob("docs/api/**/*")) if Path("docs/api").exists() else []
            metrics["api_documentation_files"] = len(api_docs)
            
            # Code documentation (docstrings)
            python_files = list(Path("src").rglob("*.py")) if Path("src").exists() else []
            total_functions = 0
            documented_functions = 0
            
            for py_file in python_files:
                try:
                    content = py_file.read_text()
                    import ast
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            total_functions += 1
                            if ast.get_docstring(node):
                                documented_functions += 1
                except:
                    pass
                    
            metrics["function_documentation_coverage"] = (
                (documented_functions / total_functions * 100) if total_functions > 0 else 0
            )
            
            return metrics
        except Exception as e:
            print(f"⚠️  Warning: Could not collect documentation metrics: {e}")
            return {}
    
    def _collect_business_metrics(self) -> Dict[str, Any]:
        """Collect business and usage metrics."""
        try:
            metrics = {}
            
            # GitHub stars, forks, etc. (if public repo)
            # This would typically come from GitHub API
            metrics["github_stars"] = 0  # Placeholder
            metrics["github_forks"] = 0  # Placeholder
            metrics["github_watchers"] = 0  # Placeholder
            
            # Package downloads (if published)
            # This would typically come from PyPI API
            metrics["pypi_downloads_last_month"] = 0  # Placeholder
            
            # Docker pulls (if published)
            # This would typically come from Docker Hub API
            metrics["docker_pulls"] = 0  # Placeholder
            
            # Issue metrics
            metrics["estimated_value"] = {
                "time_saved_per_use_hours": 2.0,
                "cost_per_generation_usd": 0.01,
                "automation_value_score": 85.0
            }
            
            return metrics
        except Exception as e:
            print(f"⚠️  Warning: Could not collect business metrics: {e}")
            return {}
    
    def _calculate_health_score(self) -> Dict[str, Any]:
        """Calculate overall repository health score."""
        try:
            weights = {
                "code_quality": 0.25,
                "security": 0.20,
                "testing": 0.20,
                "automation": 0.15,
                "documentation": 0.10,
                "performance": 0.10
            }
            
            scores = {}
            
            # Code quality score
            ruff_violations = self.metrics.get("code_quality", {}).get("ruff_violations", 0)
            scores["code_quality"] = max(0, 100 - ruff_violations * 5)
            
            # Security score
            security_issues = self.metrics.get("security", {}).get("security_issues", 0)
            vulnerable_deps = self.metrics.get("security", {}).get("vulnerable_dependencies", 0)
            scores["security"] = max(0, 100 - (security_issues + vulnerable_deps) * 10)
            
            # Testing score
            test_coverage = self.metrics.get("testing", {}).get("test_coverage_percent", 0)
            scores["testing"] = min(100, test_coverage)
            
            # Automation score
            workflows = self.metrics.get("automation", {}).get("github_workflows_count", 0)
            scripts = self.metrics.get("automation", {}).get("automation_scripts_count", 0)
            scores["automation"] = min(100, (workflows * 10) + (scripts * 5))
            
            # Documentation score
            doc_files = self.metrics.get("documentation", {}).get("documentation_files", 0)
            func_coverage = self.metrics.get("documentation", {}).get("function_documentation_coverage", 0)
            scores["documentation"] = min(100, (doc_files * 5) + func_coverage)
            
            # Performance score (inverse of build time, test time)
            build_time = self.metrics.get("performance", {}).get("build_time_seconds", 0)
            test_time = self.metrics.get("performance", {}).get("test_execution_time_seconds", 0)
            scores["performance"] = max(0, 100 - (build_time + test_time) / 10)
            
            # Calculate weighted average
            overall_score = sum(scores[component] * weights[component] for component in weights)
            
            return {
                "overall": round(overall_score, 1),
                "components": scores,
                "weights": weights
            }
        except Exception as e:
            print(f"⚠️  Warning: Could not calculate health score: {e}")
            return {"overall": 0, "components": {}, "weights": {}}
    
    def _get_directory_size(self) -> float:
        """Get total directory size in MB."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk("."):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if not os.path.islink(filepath):
                    total_size += os.path.getsize(filepath)
        return round(total_size / (1024 * 1024), 2)
    
    def _analyze_directory_structure(self) -> Dict[str, int]:
        """Analyze directory structure."""
        structure = {}
        for item in Path(".").iterdir():
            if item.is_dir() and not item.name.startswith("."):
                file_count = len(list(item.rglob("*")))
                structure[item.name] = file_count
        return structure
    
    def save_metrics(self, output_file: str = None) -> str:
        """Save metrics to file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"metrics_{timestamp}.json"
            
        output_path = Path(output_file)
        output_path.write_text(json.dumps(self.metrics, indent=2))
        
        print(f"✅ Metrics saved to {output_path}")
        return str(output_path)
    
    def generate_report(self) -> str:
        """Generate a human-readable metrics report."""
        health_score = self.metrics.get("health_score", {})
        overall_score = health_score.get("overall", 0)
        
        report = f"""
# 📊 Project Metrics Report

**Generated:** {self.timestamp}
**Overall Health Score:** {overall_score}/100

## 🏥 Health Breakdown
"""
        
        components = health_score.get("components", {})
        for component, score in components.items():
            status = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
            report += f"- **{component.replace('_', ' ').title()}:** {score:.1f}/100 {status}\n"
        
        report += f"""
## 📈 Key Metrics

### Repository
- **Lines of Code:** {self.metrics.get('repository', {}).get('total_lines_of_code', 'N/A'):,}
- **Total Files:** {self.metrics.get('repository', {}).get('total_files', 'N/A'):,}
- **Repository Size:** {self.metrics.get('repository', {}).get('size_mb', 'N/A')} MB

### Code Quality
- **Linting Issues:** {self.metrics.get('code_quality', {}).get('ruff_violations', 'N/A')}
- **Security Issues:** {self.metrics.get('security', {}).get('security_issues', 'N/A')}
- **Test Coverage:** {self.metrics.get('testing', {}).get('test_coverage_percent', 'N/A'):.1f}%

### Development
- **Total Commits:** {self.metrics.get('git', {}).get('total_commits', 'N/A'):,}
- **Contributors:** {self.metrics.get('git', {}).get('contributor_count', 'N/A')}
- **GitHub Workflows:** {self.metrics.get('automation', {}).get('github_workflows_count', 'N/A')}

### Performance
- **Build Time:** {self.metrics.get('performance', {}).get('build_time_seconds', 'N/A')}s
- **Test Time:** {self.metrics.get('performance', {}).get('test_execution_time_seconds', 'N/A')}s
"""
        
        return report


def main():
    """Main function to collect and report metrics."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Collect project metrics")
    parser.add_argument("--output", "-o", help="Output file for metrics JSON")
    parser.add_argument("--report", "-r", action="store_true", help="Generate human-readable report")
    parser.add_argument("--config", "-c", help="Path to metrics configuration file")
    
    args = parser.parse_args()
    
    try:
        collector = MetricsCollector(args.config or ".github/project-metrics.json")
        metrics = collector.collect_all_metrics()
        
        # Save metrics
        output_file = collector.save_metrics(args.output)
        
        # Generate report if requested
        if args.report:
            report = collector.generate_report()
            report_file = output_file.replace(".json", "_report.md")
            Path(report_file).write_text(report)
            print(f"📄 Report saved to {report_file}")
            
        print(f"\n🎯 Overall Health Score: {metrics['health_score']['overall']}/100")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error collecting metrics: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())