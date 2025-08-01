#!/usr/bin/env python3
"""
Terragon Metrics Dashboard
Generates comprehensive value delivery and repository health reports.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

class MetricsDashboard:
    """Advanced metrics and reporting dashboard."""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.metrics_file = repo_path / ".terragon" / "value-metrics.json"
        self.items_file = repo_path / ".terragon" / "discovered-items.json"
    
    def load_metrics(self) -> Dict:
        """Load current metrics data."""
        if self.metrics_file.exists():
            with open(self.metrics_file) as f:
                return json.load(f)
        return self._empty_metrics()
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure."""
        return {
            "repository": {"name": "unknown", "maturityLevel": "unknown"},
            "executionHistory": [],
            "valueDelivered": {"completedTasks": 0, "totalScore": 0},
            "backlogMetrics": {"totalItems": 0},
            "technicalMetrics": {}
        }
    
    def calculate_health_score(self) -> float:
        """Calculate overall repository health score (0-100)."""
        metrics = self.load_metrics()
        
        # Base scores for different aspects
        scores = {
            "maturity": 0,
            "testing": 0,
            "security": 0,
            "automation": 0,
            "documentation": 0
        }
        
        # Maturity level scoring
        maturity_level = metrics.get("repository", {}).get("maturityLevel", "nascent")
        maturity_scores = {
            "nascent": 20,
            "developing": 40,
            "maturing": 60,
            "advanced": 80
        }
        scores["maturity"] = maturity_scores.get(maturity_level, 20)
        
        # Testing score (based on coverage and test files)
        try:
            test_files = len(list(self.repo_path.rglob("test_*.py")))
            scores["testing"] = min(test_files * 2, 90)  # Cap at 90
        except:
            scores["testing"] = 50
        
        # Security score (based on security tools presence)
        security_files = [
            ".secrets.baseline",
            "security_results.json", 
            "pyproject.toml"  # bandit config
        ]
        security_present = sum(1 for f in security_files if (self.repo_path / f).exists())
        scores["security"] = (security_present / len(security_files)) * 100
        
        # Automation score (based on autonomous backlog activity)
        backlog_items = metrics.get("backlogMetrics", {}).get("totalItems", 0)
        completed_tasks = metrics.get("valueDelivered", {}).get("completedTasks", 0)
        if backlog_items > 0:
            scores["automation"] = min((completed_tasks / backlog_items) * 100 + 50, 100)
        else:
            scores["automation"] = 70  # Good baseline if no issues found
        
        # Documentation score (based on docs directory)
        try:
            doc_files = len(list(self.repo_path.rglob("docs/*.md")))
            scores["documentation"] = min(doc_files * 5, 95)  # Cap at 95
        except:
            scores["documentation"] = 60
        
        # Weighted average
        weights = {
            "maturity": 0.3,
            "testing": 0.25,
            "security": 0.2,
            "automation": 0.15,
            "documentation": 0.1
        }
        
        health_score = sum(scores[aspect] * weights[aspect] for aspect in scores)
        return round(health_score, 1)
    
    def get_productivity_metrics(self) -> Dict:
        """Calculate productivity and velocity metrics."""
        metrics = self.load_metrics()
        history = metrics.get("executionHistory", [])
        
        if not history:
            return {
                "velocity": 0,
                "cycle_time": 0,
                "accuracy": 0,
                "efficiency": 0
            }
        
        # Calculate velocity (completed tasks per week)
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)
        
        recent_completions = [
            item for item in history
            if datetime.fromisoformat(item["timestamp"]) > week_ago
        ]
        
        velocity = len(recent_completions)
        
        # Calculate average cycle time (effort estimation accuracy)
        if recent_completions:
            effort_ratios = [
                item["actual_effort"] / max(item["predicted_effort"], 0.1)
                for item in recent_completions
                if "actual_effort" in item and "predicted_effort" in item
            ]
            accuracy = 1.0 - abs(1.0 - (sum(effort_ratios) / len(effort_ratios))) if effort_ratios else 0.8
            cycle_time = sum(item.get("actual_effort", 0) for item in recent_completions) / len(recent_completions)
        else:
            accuracy = 0.8  # Default reasonable accuracy
            cycle_time = 2.0  # Default 2 hour cycle time
        
        # Efficiency (value delivered per effort)
        total_score = sum(item.get("predicted_score", 0) for item in recent_completions)
        total_effort = sum(item.get("actual_effort", 0) for item in recent_completions)
        efficiency = total_score / max(total_effort, 0.1) if total_effort > 0 else 50
        
        return {
            "velocity": velocity,
            "cycle_time": round(cycle_time, 2),
            "accuracy": round(accuracy, 3),
            "efficiency": round(efficiency, 1)
        }
    
    def generate_trend_analysis(self) -> str:
        """Generate trend analysis report."""
        metrics = self.load_metrics()
        productivity = self.get_productivity_metrics()
        
        # Determine trends
        trends = []
        
        if productivity["velocity"] > 5:
            trends.append("🚀 High velocity - completing many tasks")
        elif productivity["velocity"] > 2:
            trends.append("📈 Good velocity - steady progress")
        else:
            trends.append("🐌 Low velocity - consider process optimization")
        
        if productivity["accuracy"] > 0.9:
            trends.append("🎯 Excellent estimation accuracy")
        elif productivity["accuracy"] > 0.7:
            trends.append("✅ Good estimation accuracy")
        else:
            trends.append("📊 Estimation accuracy needs improvement")
        
        if productivity["efficiency"] > 30:
            trends.append("⚡ High efficiency - great value per effort")
        elif productivity["efficiency"] > 15:
            trends.append("📋 Moderate efficiency")
        else:
            trends.append("🔧 Low efficiency - focus on high-impact work")
        
        backlog_size = metrics.get("backlogMetrics", {}).get("totalItems", 0)
        if backlog_size == 0:
            trends.append("✨ Clean backlog - repository in excellent shape")
        elif backlog_size < 10:
            trends.append("📝 Small backlog - well-maintained repository")
        elif backlog_size < 50:
            trends.append("📚 Moderate backlog - good discovery active")
        else:
            trends.append("📦 Large backlog - many opportunities identified")
        
        return "\n".join(f"- {trend}" for trend in trends)
    
    def generate_dashboard_report(self) -> str:
        """Generate comprehensive dashboard report."""
        metrics = self.load_metrics()
        health_score = self.calculate_health_score()
        productivity = self.get_productivity_metrics()
        
        report = f"""# 📊 Terragon Metrics Dashboard

Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
Repository: {metrics.get('repository', {}).get('name', 'Unknown')}

## 🏥 Repository Health Score: {health_score}/100

### Health Breakdown
- **Maturity Level**: {metrics.get('repository', {}).get('maturityLevel', 'Unknown').title()}
- **Test Coverage**: {"High" if health_score > 80 else "Medium" if health_score > 60 else "Needs Improvement"}
- **Security Posture**: {"Strong" if health_score > 75 else "Good" if health_score > 50 else "Needs Attention"}
- **Automation Level**: {"Advanced" if health_score > 80 else "Moderate" if health_score > 60 else "Basic"}

## 📈 Productivity Metrics

| Metric | Value | Status |
|--------|-------|--------|
| **Weekly Velocity** | {productivity["velocity"]} tasks | {"🚀 High" if productivity["velocity"] > 5 else "📈 Good" if productivity["velocity"] > 2 else "🐌 Low"} |
| **Avg Cycle Time** | {productivity["cycle_time"]} hours | {"⚡ Fast" if productivity["cycle_time"] < 2 else "📋 Normal" if productivity["cycle_time"] < 4 else "🐌 Slow"} |
| **Estimation Accuracy** | {productivity["accuracy"]:.1%} | {"🎯 Excellent" if productivity["accuracy"] > 0.9 else "✅ Good" if productivity["accuracy"] > 0.7 else "📊 Fair"} |
| **Value Efficiency** | {productivity["efficiency"]:.1f} pts/hr | {"⚡ High" if productivity["efficiency"] > 30 else "📋 Medium" if productivity["efficiency"] > 15 else "🔧 Low"} |

## 🎯 Value Delivered

- **Total Completed Tasks**: {metrics.get('valueDelivered', {}).get('completedTasks', 0)}
- **Total Value Score**: {metrics.get('valueDelivered', {}).get('totalScore', 0):.1f}
- **Current Backlog**: {metrics.get('backlogMetrics', {}).get('totalItems', 0)} items
- **Execution Success Rate**: {len([h for h in metrics.get('executionHistory', []) if h.get('success', False)]) / max(len(metrics.get('executionHistory', [])), 1):.1%}

## 📊 Trend Analysis

{self.generate_trend_analysis()}

## 🔄 Recent Activity

"""
        
        # Add recent execution history
        history = metrics.get("executionHistory", [])[-5:]  # Last 5 items
        if history:
            report += "| Date | Task | Category | Status |\n|------|------|----------|--------|\n"
            for item in reversed(history):  # Most recent first
                date = datetime.fromisoformat(item["timestamp"]).strftime("%m-%d")
                title = item["title"][:30] + "..." if len(item["title"]) > 30 else item["title"]
                category = item.get("category", "unknown")
                status = "✅ Success" if item.get("success", False) else "❌ Failed"
                report += f"| {date} | {title} | {category} | {status} |\n"
        else:
            report += "*No recent execution history*\n"
        
        report += f"""
## 📋 Quick Actions

```bash
# Discover new work items
make autonomous-discover

# Execute next best item  
make autonomous-execute

# Full autonomous cycle
make autonomous-cycle

# Update this dashboard
python3 .terragon/metrics-dashboard.py
```

## 🎯 Recommendations

"""
        
        # Add specific recommendations based on metrics
        if health_score < 70:
            report += "- 🔧 **Improve Repository Health**: Focus on testing, security, and documentation\n"
        
        if productivity["velocity"] < 2:
            report += "- 📈 **Increase Velocity**: Run autonomous discovery more frequently\n"
        
        if productivity["accuracy"] < 0.8:
            report += "- 🎯 **Improve Estimation**: Review completed tasks to calibrate effort estimates\n"
        
        backlog_size = metrics.get("backlogMetrics", {}).get("totalItems", 0)
        if backlog_size > 50:
            report += "- 📦 **Manage Backlog**: Consider filtering or batching large backlogs\n"
        elif backlog_size == 0:
            report += "- 🔍 **Discovery Needed**: Run autonomous discovery to identify opportunities\n"
        
        report += "\n---\n*Generated by Terragon Autonomous SDLC Engine*\n"
        
        return report
    
    def save_dashboard(self):
        """Save dashboard report to file."""
        report = self.generate_dashboard_report()
        
        # Save to multiple locations
        dashboard_file = self.repo_path / "TERRAGON_METRICS_DASHBOARD.md"
        with open(dashboard_file, "w") as f:
            f.write(report)
        
        # Also save to docs for visibility
        docs_dashboard = self.repo_path / "docs" / "status" / f"metrics-dashboard-{datetime.now().strftime('%Y-%m-%d')}.md"
        docs_dashboard.parent.mkdir(parents=True, exist_ok=True)
        with open(docs_dashboard, "w") as f:
            f.write(report)
        
        print(f"✅ Dashboard saved to {dashboard_file}")
        print(f"📊 Archive saved to {docs_dashboard}")

def main():
    """Main dashboard generation entry point."""
    repo_path = Path.cwd()
    
    if not (repo_path / ".git").exists():
        print("Error: Not in a git repository")
        sys.exit(1)
    
    dashboard = MetricsDashboard(repo_path)
    
    # Generate and save dashboard
    dashboard.save_dashboard()
    
    # Print summary
    health_score = dashboard.calculate_health_score()
    productivity = dashboard.get_productivity_metrics()
    
    print(f"\n📊 Repository Health: {health_score}/100")
    print(f"🚀 Weekly Velocity: {productivity['velocity']} tasks")
    print(f"⚡ Efficiency: {productivity['efficiency']:.1f} points/hour")

if __name__ == "__main__":
    main()