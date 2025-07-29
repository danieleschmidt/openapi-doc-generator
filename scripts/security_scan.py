#!/usr/bin/env python3
"""Comprehensive security scanning script."""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

class SecurityScanner:
    """Comprehensive security scanner for the project."""
    
    def __init__(self, output_dir: str = "."):
        self.output_dir = Path(output_dir)
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "scans": {}
        }
    
    def run_command(self, cmd: List[str]) -> Optional[str]:
        """Run command and return output."""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Warning: {' '.join(cmd)} failed: {e}")
            return None
    
    def scan_with_bandit(self) -> Dict[str, Any]:
        """Run Bandit security scan."""
        print("Running Bandit security scan...")
        
        output_file = self.output_dir / "bandit-report.json"
        cmd = [
            "bandit", "-r", "src/", 
            "-f", "json", 
            "-o", str(output_file)
        ]
        
        result = self.run_command(cmd)
        
        try:
            with open(output_file) as f:
                bandit_results = json.load(f)
            
            return {
                "tool": "bandit",
                "status": "completed",
                "issues_found": len(bandit_results.get("results", [])),
                "report_file": str(output_file),
                "summary": {
                    "high": len([r for r in bandit_results.get("results", []) 
                               if r.get("issue_severity") == "HIGH"]),
                    "medium": len([r for r in bandit_results.get("results", []) 
                                 if r.get("issue_severity") == "MEDIUM"]),
                    "low": len([r for r in bandit_results.get("results", []) 
                              if r.get("issue_severity") == "LOW"])
                }
            }
        except (FileNotFoundError, json.JSONDecodeError):
            return {"tool": "bandit", "status": "failed", "error": "No results file"}
    
    def scan_with_safety(self) -> Dict[str, Any]:
        """Run Safety vulnerability scan."""
        print("Running Safety vulnerability scan...")
        
        output_file = self.output_dir / "safety-report.json"
        cmd = ["safety", "check", "--json", "--output", str(output_file)]
        
        # Safety returns non-zero on vulnerabilities, so don't use check=True
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            with open(output_file) as f:
                safety_results = json.load(f)
            
            vulnerabilities = safety_results.get("vulnerabilities", [])
            
            return {
                "tool": "safety",
                "status": "completed",
                "vulnerabilities_found": len(vulnerabilities),
                "report_file": str(output_file),
                "summary": {
                    "packages_scanned": len(safety_results.get("scanned_packages", [])),
                    "vulnerable_packages": len(set(v.get("package") for v in vulnerabilities))
                }
            }
        except (FileNotFoundError, json.JSONDecodeError) as e:
            return {"tool": "safety", "status": "failed", "error": str(e)}
    
    def scan_with_pip_audit(self) -> Dict[str, Any]:
        """Run pip-audit scan."""
        print("Running pip-audit scan...")
        
        output_file = self.output_dir / "pip-audit-report.json"
        cmd = ["pip-audit", "--format", "json", "--output", str(output_file)]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            with open(output_file) as f:
                audit_results = json.load(f)
            
            vulnerabilities = audit_results.get("vulnerabilities", [])
            
            return {
                "tool": "pip-audit",
                "status": "completed",
                "vulnerabilities_found": len(vulnerabilities),
                "report_file": str(output_file),
                "summary": {
                    "packages_audited": len(audit_results.get("dependencies", [])),
                    "vulnerable_packages": len(set(v.get("package") for v in vulnerabilities))
                }
            }
        except (FileNotFoundError, json.JSONDecodeError) as e:
            return {"tool": "pip-audit", "status": "failed", "error": str(e)}
    
    def scan_secrets(self) -> Dict[str, Any]:
        """Run detect-secrets scan."""
        print("Running secrets detection...")
        
        cmd = ["detect-secrets", "scan", "--all-files"]
        result = self.run_command(cmd)
        
        if result:
            try:
                secrets_data = json.loads(result)
                return {
                    "tool": "detect-secrets",
                    "status": "completed",
                    "secrets_found": len(secrets_data.get("results", {})),
                    "baseline_updated": datetime.now().isoformat()
                }
            except json.JSONDecodeError:
                pass
        
        return {"tool": "detect-secrets", "status": "failed"}
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive security summary."""
        total_issues = 0
        critical_issues = 0
        
        for scan_name, scan_result in self.results["scans"].items():
            if scan_result.get("status") == "completed":
                if "issues_found" in scan_result:
                    total_issues += scan_result["issues_found"]
                if "vulnerabilities_found" in scan_result:
                    total_issues += scan_result["vulnerabilities_found"]
                if "secrets_found" in scan_result:
                    critical_issues += scan_result["secrets_found"]
                
                # Count high severity as critical
                summary = scan_result.get("summary", {})
                if "high" in summary:
                    critical_issues += summary["high"]
        
        return {
            "overall_status": "critical" if critical_issues > 0 else "warning" if total_issues > 0 else "clean",
            "total_issues": total_issues,
            "critical_issues": critical_issues,
            "scans_completed": len([s for s in self.results["scans"].values() 
                                  if s.get("status") == "completed"]),
            "scans_failed": len([s for s in self.results["scans"].values() 
                               if s.get("status") == "failed"]),
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations based on scan results."""
        recommendations = []
        
        for scan_name, scan_result in self.results["scans"].items():
            if scan_result.get("status") == "failed":
                recommendations.append(f"Fix {scan_name} scanner configuration")
            elif scan_result.get("vulnerabilities_found", 0) > 0:
                recommendations.append(f"Address vulnerabilities found by {scan_name}")
            elif scan_result.get("secrets_found", 0) > 0:
                recommendations.append("Remove exposed secrets and rotate credentials")
        
        if not recommendations:
            recommendations.append("Maintain regular security scanning schedule")
        
        return recommendations
    
    def run_all_scans(self):
        """Run all security scans."""
        print("Starting comprehensive security scan...")
        
        self.results["scans"]["bandit"] = self.scan_with_bandit()
        self.results["scans"]["safety"] = self.scan_with_safety()
        self.results["scans"]["pip_audit"] = self.scan_with_pip_audit()
        self.results["scans"]["secrets"] = self.scan_secrets()
        
        # Generate summary
        self.results["summary"] = self.generate_summary_report()
        
        # Save comprehensive report
        report_file = self.output_dir / "security_scan_complete.json"
        with open(report_file, "w") as f:
            json.dump(self.results, f, indent=2, sort_keys=True)
        
        print(f"\nSecurity scan completed!")
        print(f"Overall status: {self.results['summary']['overall_status']}")
        print(f"Total issues: {self.results['summary']['total_issues']}")
        print(f"Critical issues: {self.results['summary']['critical_issues']}")
        print(f"Complete report: {report_file}")

def main():
    """Main function."""
    scanner = SecurityScanner()
    scanner.run_all_scans()

if __name__ == "__main__":
    main()