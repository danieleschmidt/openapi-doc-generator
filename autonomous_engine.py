#!/usr/bin/env python3
"""
Autonomous Backlog Management Engine
Implements WSJF-based discovery, prioritization, and execution
"""

import json
import subprocess
import yaml
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class TaskStatus(Enum):
    NEW = "NEW"
    REFINED = "REFINED"
    READY = "READY"
    DOING = "DOING"
    PR = "PR"
    DONE = "DONE"
    BLOCKED = "BLOCKED"


class RiskTier(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class BacklogItem:
    id: str
    title: str
    type: str
    description: str
    acceptance_criteria: List[str]
    effort: float
    value: int
    time_criticality: int
    risk_reduction: int
    wsjf_score: float
    status: TaskStatus
    risk_tier: RiskTier
    created_at: str
    links: List[str]
    aging_multiplier: float = 1.0
    
    def calculate_wsjf(self) -> float:
        """Calculate WSJF score with aging multiplier"""
        base_score = (self.value + self.time_criticality + self.risk_reduction) / self.effort
        return base_score * self.aging_multiplier
    
    def apply_aging(self, days_old: int) -> None:
        """Apply aging multiplier based on item age"""
        if days_old > 30:
            self.aging_multiplier = min(2.0, 1.0 + (days_old - 30) * 0.02)
            self.wsjf_score = self.calculate_wsjf()


class AutonomousBacklogEngine:
    def __init__(self, repo_root: str = "/root/repo"):
        self.repo_root = Path(repo_root)
        self.backlog_file = self.repo_root / "backlog.yml"
        self.metrics_dir = self.repo_root / "docs" / "status"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('autonomous_engine')
        
        # Load configuration
        self.scope_config = self._load_scope_config()
        
    def _load_scope_config(self) -> Dict:
        """Load automation scope configuration"""
        scope_file = self.repo_root / ".automation-scope.yaml"
        if scope_file.exists():
            with open(scope_file) as f:
                return yaml.safe_load(f)
        return {"repository_only": True}
    
    def discover_new_tasks(self) -> List[BacklogItem]:
        """Continuous discovery from multiple sources"""
        discovered = []
        
        # TODO/FIXME discovery
        discovered.extend(self._discover_code_comments())
        
        # Security issues
        discovered.extend(self._discover_security_issues())
        
        # Lint issues
        discovered.extend(self._discover_lint_issues())
        
        # Dependency issues
        discovered.extend(self._discover_dependency_issues())
        
        # Test failures
        discovered.extend(self._discover_test_issues())
        
        return discovered
    
    def _discover_code_comments(self) -> List[BacklogItem]:
        """Discover TODO/FIXME comments in code"""
        items = []
        patterns = ["TODO", "FIXME", "HACK", "BUG"]
        
        for pattern in patterns:
            try:
                result = subprocess.run([
                    "grep", "-r", "-n", "-i", pattern, str(self.repo_root / "src"),
                    "--include=*.py"
                ], capture_output=True, text=True)
                
                for line in result.stdout.strip().split('\n'):
                    if line and ':' in line:
                        file_path, line_num, content = line.split(':', 2)
                        item_id = f"DISC_{pattern}_{hash(line) % 10000:04d}"
                        
                        items.append(BacklogItem(
                            id=item_id,
                            title=f"Address {pattern} comment in {Path(file_path).name}",
                            type="code_quality",
                            description=content.strip(),
                            acceptance_criteria=[f"Resolve {pattern} comment", "Ensure tests pass"],
                            effort=1,
                            value=2,
                            time_criticality=1 if pattern in ["BUG", "FIXME"] else 0,
                            risk_reduction=1 if pattern == "BUG" else 0,
                            wsjf_score=0,
                            status=TaskStatus.NEW,
                            risk_tier=RiskTier.LOW,
                            created_at=datetime.now(timezone.utc).isoformat(),
                            links=[f"{file_path}:{line_num}"]
                        ))
            except subprocess.CalledProcessError:
                pass
                
        return items
    
    def _discover_security_issues(self) -> List[BacklogItem]:
        """Discover security issues via bandit scan"""
        items = []
        try:
            result = subprocess.run([
                "bandit", "-r", str(self.repo_root / "src"), "-f", "json"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for issue in data.get("results", []):
                    item_id = f"SEC_{hash(issue['filename'] + str(issue['line_number'])) % 10000:04d}"
                    
                    severity_map = {"HIGH": 8, "MEDIUM": 5, "LOW": 3}
                    value = severity_map.get(issue.get("issue_severity", "LOW"), 3)
                    
                    items.append(BacklogItem(
                        id=item_id,
                        title=f"Security: {issue['test_name']}",
                        type="security",
                        description=issue["issue_text"],
                        acceptance_criteria=["Fix security vulnerability", "Verify with security scan"],
                        effort=2,
                        value=value,
                        time_criticality=5 if issue.get("issue_severity") == "HIGH" else 3,
                        risk_reduction=8,
                        wsjf_score=0,
                        status=TaskStatus.NEW,
                        risk_tier=RiskTier.HIGH if issue.get("issue_severity") == "HIGH" else RiskTier.MEDIUM,
                        created_at=datetime.now(timezone.utc).isoformat(),
                        links=[f"{issue['filename']}:{issue['line_number']}"]
                    ))
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            pass
            
        return items
    
    def _discover_lint_issues(self) -> List[BacklogItem]:
        """Discover lint issues via ruff"""
        items = []
        try:
            result = subprocess.run([
                "ruff", "check", str(self.repo_root), "--output-format=json"
            ], capture_output=True, text=True)
            
            if result.stdout:
                data = json.loads(result.stdout)
                
                # Group by error type for bulk fixing
                error_groups = {}
                for issue in data:
                    error_type = issue["code"]
                    if error_type not in error_groups:
                        error_groups[error_type] = []
                    error_groups[error_type].append(issue)
                
                for error_type, issues in error_groups.items():
                    if len(issues) > 3:  # Only create backlog items for recurring issues
                        item_id = f"LINT_{error_type}_{len(issues)}"
                        
                        items.append(BacklogItem(
                            id=item_id,
                            title=f"Fix {len(issues)} {error_type} lint violations",
                            type="code_quality",
                            description=f"Fix {len(issues)} instances of {error_type}: {issues[0]['message']}",
                            acceptance_criteria=["All lint violations resolved", "Tests still pass"],
                            effort=1 if len(issues) < 10 else 2,
                            value=3,
                            time_criticality=1,
                            risk_reduction=1,
                            wsjf_score=0,
                            status=TaskStatus.NEW,
                            risk_tier=RiskTier.LOW,
                            created_at=datetime.now(timezone.utc).isoformat(),
                            links=[f"{issue['filename']}:{issue['location']['row']}" for issue in issues[:5]]
                        ))
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            pass
            
        return items
    
    def _discover_dependency_issues(self) -> List[BacklogItem]:
        """Discover dependency vulnerabilities"""
        items = []
        # This would integrate with tools like pip-audit or safety
        # For now, placeholder implementation
        return items
    
    def _discover_test_issues(self) -> List[BacklogItem]:
        """Discover failing or flaky tests"""
        items = []
        try:
            result = subprocess.run([
                "python", "-m", "pytest", "--tb=no", "-q"
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.returncode != 0:
                # Parse test failures and create backlog items
                for line in result.stdout.split('\n'):
                    if 'FAILED' in line:
                        test_name = line.split()[0] if line.split() else "unknown"
                        item_id = f"TEST_{hash(test_name) % 10000:04d}"
                        
                        items.append(BacklogItem(
                            id=item_id,
                            title=f"Fix failing test: {test_name}",
                            type="test_failure",
                            description=f"Test failure in {test_name}",
                            acceptance_criteria=["Test passes", "No regression in other tests"],
                            effort=2,
                            value=8,
                            time_criticality=8,
                            risk_reduction=5,
                            wsjf_score=0,
                            status=TaskStatus.NEW,
                            risk_tier=RiskTier.HIGH,
                            created_at=datetime.now(timezone.utc).isoformat(),
                            links=[test_name]
                        ))
        except subprocess.CalledProcessError:
            pass
            
        return items
    
    def load_backlog(self) -> List[BacklogItem]:
        """Load backlog from YAML file"""
        if not self.backlog_file.exists():
            return []
            
        with open(self.backlog_file) as f:
            data = yaml.safe_load(f)
            
        items = []
        for item_data in data.get("items", []):
            # Apply aging
            created = datetime.fromisoformat(item_data["created_at"].replace('Z', '+00:00'))
            days_old = (datetime.now(timezone.utc) - created).days
            
            item = BacklogItem(
                id=item_data["id"],
                title=item_data["title"],
                type=item_data["type"],
                description=item_data["description"],
                acceptance_criteria=item_data["acceptance_criteria"],
                effort=item_data["effort"],
                value=item_data["value"],
                time_criticality=item_data["time_criticality"],
                risk_reduction=item_data["risk_reduction"],
                wsjf_score=item_data["wsjf_score"],
                status=TaskStatus(item_data["status"]),
                risk_tier=RiskTier(item_data["risk_tier"]),
                created_at=item_data["created_at"],
                links=item_data["links"]
            )
            
            item.apply_aging(days_old)
            items.append(item)
            
        return items
    
    def save_backlog(self, items: List[BacklogItem]) -> None:
        """Save backlog to YAML file"""
        data = {
            "metadata": {
                "version": "1.0",
                "methodology": "WSJF (Weighted Shortest Job First)",
                "scoring_scale": "1-2-3-5-8-13 (Fibonacci)",
                "last_discovery": datetime.now(timezone.utc).isoformat(),
            },
            "items": []
        }
        
        for item in items:
            item_dict = asdict(item)
            item_dict["status"] = item.status.value
            item_dict["risk_tier"] = item.risk_tier.value
            data["items"].append(item_dict)
        
        with open(self.backlog_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def prioritize_backlog(self, items: List[BacklogItem]) -> List[BacklogItem]:
        """Sort backlog by WSJF score (descending)"""
        for item in items:
            item.wsjf_score = item.calculate_wsjf()
        
        return sorted(items, key=lambda x: x.wsjf_score, reverse=True)
    
    def get_next_task(self, items: List[BacklogItem]) -> Optional[BacklogItem]:
        """Get the next ready task to execute"""
        ready_items = [item for item in items if item.status == TaskStatus.READY]
        if not ready_items:
            return None
        
        # Return highest WSJF score that's in scope
        for item in sorted(ready_items, key=lambda x: x.wsjf_score, reverse=True):
            if self._is_in_scope(item):
                return item
        
        return None
    
    def _is_in_scope(self, item: BacklogItem) -> bool:
        """Check if item is within automation scope"""
        if self.scope_config.get("repository_only", True):
            return True  # All items discovered within repo are in scope
        
        # Additional scope checks would go here
        return True
    
    def generate_metrics_report(self, items: List[BacklogItem]) -> Dict:
        """Generate metrics and status report"""
        now = datetime.now(timezone.utc)
        
        # Count by status
        status_counts = {}
        for status in TaskStatus:
            status_counts[status.value] = len([i for i in items if i.status == status])
        
        # Calculate DORA-like metrics (simplified)
        completed_today = [i for i in items if i.status == TaskStatus.DONE]
        
        report = {
            "timestamp": now.isoformat(),
            "completed_ids": [i.id for i in completed_today],
            "backlog_size_by_status": status_counts,
            "total_backlog_size": len(items),
            "ready_items": len([i for i in items if i.status == TaskStatus.READY]),
            "high_priority_items": len([i for i in items if i.wsjf_score > 4.0]),
            "avg_wsjf_score": sum(i.wsjf_score for i in items) / len(items) if items else 0,
            "wsjf_snapshot": [
                {"id": i.id, "title": i.title, "wsjf": i.wsjf_score}
                for i in sorted(items, key=lambda x: x.wsjf_score, reverse=True)[:5]
            ]
        }
        
        return report
    
    def save_metrics_report(self, report: Dict) -> None:
        """Save metrics report to status directory"""
        timestamp = datetime.now(timezone.utc)
        date_str = timestamp.strftime("%Y-%m-%d")
        
        # JSON report
        json_file = self.metrics_dir / f"{date_str}.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Markdown summary
        md_file = self.metrics_dir / f"{date_str}.md"
        with open(md_file, 'w') as f:
            f.write(f"# Autonomous Backlog Status - {date_str}\n\n")
            f.write(f"**Generated:** {report['timestamp']}\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"- **Total Backlog Items:** {report['total_backlog_size']}\n")
            f.write(f"- **Ready for Execution:** {report['ready_items']}\n")
            f.write(f"- **High Priority (WSJF > 4.0):** {report['high_priority_items']}\n")
            f.write(f"- **Average WSJF Score:** {report['avg_wsjf_score']:.2f}\n\n")
            
            f.write("## Status Distribution\n\n")
            for status, count in report['backlog_size_by_status'].items():
                f.write(f"- **{status}:** {count}\n")
            
            f.write("\n## Top Priority Items\n\n")
            for item in report['wsjf_snapshot']:
                f.write(f"- **{item['id']}:** {item['title']} (WSJF: {item['wsjf']:.1f})\n")
    
    def run_discovery_cycle(self) -> None:
        """Run a complete discovery and prioritization cycle"""
        self.logger.info("Starting autonomous backlog discovery cycle")
        
        # Load existing backlog
        existing_items = self.load_backlog()
        self.logger.info(f"Loaded {len(existing_items)} existing backlog items")
        
        # Discover new tasks
        discovered_items = self.discover_new_tasks()
        self.logger.info(f"Discovered {len(discovered_items)} new potential backlog items")
        
        # Merge and deduplicate
        all_items = existing_items.copy()
        existing_ids = {item.id for item in existing_items}
        
        for item in discovered_items:
            if item.id not in existing_ids:
                all_items.append(item)
        
        # Prioritize
        prioritized_items = self.prioritize_backlog(all_items)
        
        # Save updated backlog
        self.save_backlog(prioritized_items)
        
        # Generate and save metrics
        report = self.generate_metrics_report(prioritized_items)
        self.save_metrics_report(report)
        
        self.logger.info(f"Discovery cycle complete. {len(prioritized_items)} total items prioritized.")
        
        return prioritized_items


if __name__ == "__main__":
    engine = AutonomousBacklogEngine()
    items = engine.run_discovery_cycle()
    
    print(f"\n🤖 Autonomous Backlog Management - Discovery Complete")
    print(f"📊 Total Items: {len(items)}")
    
    ready_items = [i for i in items if i.status == TaskStatus.READY]
    if ready_items:
        print(f"🚀 Ready for Execution: {len(ready_items)}")
        top_item = ready_items[0]
        print(f"🎯 Next Task: {top_item.title} (WSJF: {top_item.wsjf_score:.1f})")
    else:
        print("✅ No items ready for execution")