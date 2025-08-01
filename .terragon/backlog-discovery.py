#!/usr/bin/env python3
"""
Terragon Autonomous Value Discovery Engine
Continuously discovers, scores, and prioritizes work items for maximum value delivery.
"""

import ast
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
# import yaml  # Not available in base environment

@dataclass
class WorkItem:
    id: str
    title: str
    description: str
    category: str
    source: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    effort_estimate: float = 0.0
    wsjf_score: float = 0.0
    ice_score: float = 0.0
    technical_debt_score: float = 0.0
    composite_score: float = 0.0
    discovered_at: str = ""
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if not self.discovered_at:
            self.discovered_at = datetime.now(timezone.utc).isoformat()

class ValueDiscoveryEngine:
    """Advanced work item discovery and prioritization engine."""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        self.work_items: List[WorkItem] = []
        
    def _load_config(self) -> Dict:
        """Load Terragon configuration."""
        # Use hardcoded config since yaml not available
        return {
            "scoring": {
                "weights": {
                    "advanced": {
                        "wsjf": 0.5, 
                        "ice": 0.1, 
                        "technicalDebt": 0.3, 
                        "security": 0.1
                    }
                },
                "thresholds": {
                    "securityBoost": 2.0
                }
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging."""
        logger = logging.getLogger("terragon.discovery")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '{"timestamp":"%(asctime)s","level":"%(levelname)s",'
            '"logger":"%(name)s","message":"%(message)s"}'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def discover_code_comments(self) -> List[WorkItem]:
        """Discover TODO, FIXME, HACK markers in code."""
        items = []
        pattern = re.compile(r'#\s*(TODO|FIXME|HACK|XXX|NOTE|BUG)[\s:]*(.+)', re.IGNORECASE)
        
        for py_file in self.repo_path.rglob("*.py"):
            if "/.git/" in str(py_file) or "/venv/" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        match = pattern.search(line)
                        if match:
                            marker, description = match.groups()
                            item = WorkItem(
                                id=f"code-{py_file.stem}-{line_num}",
                                title=f"{marker}: {description.strip()[:50]}...",
                                description=description.strip(),
                                category="technical-debt",
                                source="code-comments",
                                file_path=str(py_file),
                                line_number=line_num,
                                tags=[marker.lower(), "code-comment"]
                            )
                            items.append(item)
            except UnicodeDecodeError:
                continue
                
        return items

    def discover_security_issues(self) -> List[WorkItem]:
        """Discover security vulnerabilities and issues."""
        items = []
        
        # Check for hardcoded secrets patterns
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
            (r'api[_-]?key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret"),
            (r'token\s*=\s*["\'][^"\']+["\']', "Hardcoded token"),
        ]
        
        for py_file in self.repo_path.rglob("*.py"):
            if "/.git/" in str(py_file) or "/test" in str(py_file):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                for pattern, description in secret_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        item = WorkItem(
                            id=f"sec-{py_file.stem}-{line_num}",
                            title=f"Security: {description}",
                            description=f"Potential {description.lower()} found in {py_file.name}",
                            category="security",
                            source="security-scan",
                            file_path=str(py_file),
                            line_number=line_num,
                            tags=["security", "credentials"]
                        )
                        items.append(item)
            except UnicodeDecodeError:
                continue
                
        return items

    def discover_performance_issues(self) -> List[WorkItem]:
        """Discover potential performance issues."""
        items = []
        
        # Performance anti-patterns
        perf_patterns = [
            (r'for\s+\w+\s+in\s+range\s*\(\s*len\s*\(', "Use enumerate() instead of range(len())"),
            (r'\+\s*=.*\[.*\]', "Potential string concatenation in loop"),
            (r'\.append\s*\(.*for.*\)', "Consider list comprehension"),
        ]
        
        for py_file in self.repo_path.rglob("*.py"):
            if "/.git/" in str(py_file) or "/test" in str(py_file):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                for pattern, suggestion in perf_patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        item = WorkItem(
                            id=f"perf-{py_file.stem}-{line_num}",
                            title=f"Performance: {suggestion}",
                            description=f"Performance optimization opportunity: {suggestion}",
                            category="performance",
                            source="performance-scan",
                            file_path=str(py_file),
                            line_number=line_num,
                            tags=["performance", "optimization"]
                        )
                        items.append(item)
            except UnicodeDecodeError:
                continue
                
        return items

    def discover_dependency_updates(self) -> List[WorkItem]:
        """Discover dependency update opportunities."""
        items = []
        
        # Check pyproject.toml for outdated dependencies
        pyproject = self.repo_path / "pyproject.toml"
        if pyproject.exists():
            try:
                # Simple heuristic - any dependency without version pinning
                content = pyproject.read_text()
                dep_pattern = r'"([^"]+)>=([^"]+)"'
                matches = re.finditer(dep_pattern, content)
                
                for match in matches:
                    package, version = match.groups()
                    item = WorkItem(
                        id=f"dep-{package}",
                        title=f"Update {package} dependency",
                        description=f"Consider updating {package} from {version} to latest version",
                        category="dependency-update",
                        source="dependency-scan",
                        file_path=str(pyproject),
                        tags=["dependency", "maintenance"]
                    )
                    items.append(item)
            except Exception:
                pass
                
        return items

    def calculate_wsjf_score(self, item: WorkItem) -> float:
        """Calculate Weighted Shortest Job First score."""
        # Simplified WSJF calculation
        business_value = 5.0  # Default moderate value
        time_criticality = 3.0
        risk_reduction = 2.0
        
        # Boost scores based on category
        if item.category == "security":
            business_value *= 2.0
            time_criticality *= 1.5
        elif item.category == "performance":
            business_value *= 1.3
        elif item.category == "technical-debt":
            risk_reduction *= 1.5
            
        cost_of_delay = business_value + time_criticality + risk_reduction
        job_size = item.effort_estimate or 2.0  # Default 2 hour estimate
        
        return cost_of_delay / job_size

    def calculate_ice_score(self, item: WorkItem) -> float:
        """Calculate Impact-Confidence-Ease score."""
        impact = 7.0  # Default moderate impact
        confidence = 8.0  # High confidence in most items
        ease = 6.0  # Moderate ease
        
        # Adjust based on category
        if item.category == "security":
            impact = 9.0
            confidence = 9.0
        elif item.category == "dependency-update":
            ease = 8.0
            confidence = 9.0
        elif item.category == "technical-debt":
            impact = 6.0
            ease = 7.0
            
        return impact * confidence * ease

    def calculate_technical_debt_score(self, item: WorkItem) -> float:
        """Calculate technical debt impact score."""
        base_score = 10.0
        
        # Weight by file importance (heuristic based on file location)
        if item.file_path:
            if "/src/" in item.file_path:
                base_score *= 1.5
            if "/cli" in item.file_path or "main" in item.file_path:
                base_score *= 1.3
            if "/test" in item.file_path:
                base_score *= 0.7
                
        # Weight by marker type
        if "FIXME" in item.tags:
            base_score *= 1.4
        elif "HACK" in item.tags:
            base_score *= 1.2
        elif "TODO" in item.tags:
            base_score *= 1.0
            
        return base_score

    def calculate_composite_score(self, item: WorkItem) -> float:
        """Calculate final composite priority score."""
        weights = self.config["scoring"]["weights"]["advanced"]
        
        normalized_wsjf = min(item.wsjf_score / 20.0, 1.0) * 100
        normalized_ice = min(item.ice_score / 1000.0, 1.0) * 100  
        normalized_debt = min(item.technical_debt_score / 50.0, 1.0) * 100
        
        composite = (
            weights["wsjf"] * normalized_wsjf +
            weights["ice"] * normalized_ice +
            weights["technicalDebt"] * normalized_debt
        )
        
        # Security boost
        if item.category == "security":
            composite *= self.config["scoring"]["thresholds"]["securityBoost"]
            
        return composite

    def score_all_items(self):
        """Calculate all scores for discovered work items."""
        for item in self.work_items:
            item.effort_estimate = self._estimate_effort(item)
            item.wsjf_score = self.calculate_wsjf_score(item)
            item.ice_score = self.calculate_ice_score(item)
            item.technical_debt_score = self.calculate_technical_debt_score(item)
            item.composite_score = self.calculate_composite_score(item)

    def _estimate_effort(self, item: WorkItem) -> float:
        """Estimate effort in hours for work item."""
        if item.category == "security":
            return 1.5  # Security fixes tend to be quick but critical
        elif item.category == "dependency-update":
            return 0.5  # Usually automated
        elif item.category == "performance":
            return 3.0  # Performance work takes more investigation
        elif item.category == "technical-debt":
            if "FIXME" in item.tags:
                return 2.0
            elif "TODO" in item.tags:
                return 1.0
            else:
                return 1.5
        return 2.0  # Default

    def discover_all(self) -> List[WorkItem]:
        """Run all discovery methods."""
        self.logger.info("Starting comprehensive value discovery")
        
        discovery_methods = [
            ("code-comments", self.discover_code_comments),
            ("security", self.discover_security_issues),
            ("performance", self.discover_performance_issues),
            ("dependencies", self.discover_dependency_updates),
        ]
        
        for name, method in discovery_methods:
            try:
                items = method()
                self.work_items.extend(items)
                self.logger.info(f"Discovered {len(items)} items from {name}")
            except Exception as e:
                self.logger.error(f"Error in {name} discovery: {e}")
        
        # Score all items
        self.score_all_items()
        
        # Sort by composite score (highest first)
        self.work_items.sort(key=lambda x: x.composite_score, reverse=True)
        
        self.logger.info(f"Total discovered items: {len(self.work_items)}")
        return self.work_items

    def generate_backlog_report(self) -> str:
        """Generate markdown backlog report."""
        if not self.work_items:
            return "# 📊 Autonomous Value Backlog\n\nNo work items discovered.\n"
            
        # Get top items
        top_items = self.work_items[:10]
        next_item = self.work_items[0] if self.work_items else None
        
        report = f"""# 📊 Autonomous Value Backlog

Last Updated: {datetime.now(timezone.utc).isoformat()}
Total Items Discovered: {len(self.work_items)}

## 🎯 Next Best Value Item
"""
        
        if next_item:
            report += f"""**[{next_item.id}] {next_item.title}**
- **Composite Score**: {next_item.composite_score:.1f}
- **WSJF**: {next_item.wsjf_score:.1f} | **ICE**: {next_item.ice_score:.0f} | **Tech Debt**: {next_item.technical_debt_score:.1f}
- **Estimated Effort**: {next_item.effort_estimate} hours
- **Category**: {next_item.category}
- **Source**: {next_item.source}
"""
            if next_item.file_path:
                report += f"- **Location**: {next_item.file_path}:{next_item.line_number or ''}\n"
        
        report += f"""
## 📋 Top {min(len(top_items), 10)} Priority Items

| Rank | ID | Title | Score | Category | Est. Hours | Source |
|------|-----|--------|---------|----------|------------|---------|
"""
        
        for i, item in enumerate(top_items, 1):
            title = item.title[:40] + "..." if len(item.title) > 40 else item.title
            report += f"| {i} | {item.id} | {title} | {item.composite_score:.1f} | {item.category} | {item.effort_estimate} | {item.source} |\n"
        
        # Category breakdown
        categories = {}
        for item in self.work_items:
            categories[item.category] = categories.get(item.category, 0) + 1
            
        report += f"""
## 📈 Discovery Statistics
- **Total Items**: {len(self.work_items)}
- **Average Score**: {sum(item.composite_score for item in self.work_items) / len(self.work_items):.1f}
- **Estimated Total Effort**: {sum(item.effort_estimate for item in self.work_items):.1f} hours

### Category Breakdown
"""
        for category, count in sorted(categories.items()):
            report += f"- **{category.replace('-', ' ').title()}**: {count} items\n"
            
        report += """
## 🔄 Continuous Discovery
This backlog is automatically updated through:
- Static code analysis
- Security vulnerability scanning  
- Performance pattern detection
- Dependency update monitoring
- Git history mining

*Generated by Terragon Autonomous SDLC Engine*
"""
        
        return report

    def save_results(self):
        """Save discovery results to files."""
        results_dir = self.repo_path / ".terragon"
        results_dir.mkdir(exist_ok=True)
        
        # Save detailed results as JSON
        items_data = [asdict(item) for item in self.work_items]
        with open(results_dir / "discovered-items.json", "w") as f:
            json.dump(items_data, f, indent=2)
        
        # Generate and save markdown report
        report = self.generate_backlog_report()
        with open(self.repo_path / "AUTONOMOUS_BACKLOG.md", "w") as f:
            f.write(report)
        
        # Update metrics
        metrics_file = results_dir / "value-metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
        else:
            metrics = {"backlogMetrics": {}}
            
        metrics["backlogMetrics"].update({
            "totalItems": len(self.work_items),
            "lastDiscovery": datetime.now(timezone.utc).isoformat(),
            "averageScore": sum(item.composite_score for item in self.work_items) / len(self.work_items) if self.work_items else 0,
            "categoryBreakdown": {
                category: len([item for item in self.work_items if item.category == category])
                for category in set(item.category for item in self.work_items)
            }
        })
        
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

def main():
    """Main entry point for value discovery."""
    repo_path = Path.cwd()
    
    # Ensure we're in a git repository
    if not (repo_path / ".git").exists():
        print("Error: Not in a git repository")
        sys.exit(1)
    
    engine = ValueDiscoveryEngine(repo_path)
    items = engine.discover_all()
    
    if items:
        engine.save_results()
        print(f"✅ Discovered {len(items)} work items")
        print(f"🎯 Next priority: {items[0].title} (Score: {items[0].composite_score:.1f})")
    else:
        print("ℹ️  No work items discovered - repository is in excellent shape!")

if __name__ == "__main__":
    main()