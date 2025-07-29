#!/usr/bin/env python3
"""Automated compliance checking script."""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

class ComplianceChecker:
    """Comprehensive compliance checker for the project."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_config(config_file)
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "compliance_checks": {},
            "overall_status": "unknown"
        }
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load compliance configuration."""
        default_config = {
            "coverage_threshold": 80,
            "security_scan_required": True,
            "license_compliance_required": True,
            "documentation_required": True,
            "allowed_licenses": [
                "MIT", "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause", "ISC"
            ],
            "blocked_licenses": [
                "GPL-3.0", "AGPL-3.0", "LGPL-3.0"
            ]
        }
        
        if config_file and Path(config_file).exists():
            with open(config_file) as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def run_command(self, cmd: List[str]) -> Optional[str]:
        """Run command and return output."""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Warning: {' '.join(cmd)} failed: {e}")
            return None
    
    def check_test_coverage(self) -> Dict[str, Any]:
        """Check test coverage compliance."""
        print("Checking test coverage...")
        
        try:
            # Run coverage report
            result = self.run_command(["coverage", "report", "--format=json"])
            if result:
                coverage_data = json.loads(result)
                total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
                
                status = "pass" if total_coverage >= self.config["coverage_threshold"] else "fail"
                
                return {
                    "check": "test_coverage",
                    "status": status,
                    "current_coverage": total_coverage,
                    "required_coverage": self.config["coverage_threshold"],
                    "details": {
                        "files_covered": len(coverage_data.get("files", {})),
                        "missing_lines": coverage_data.get("totals", {}).get("missing_lines", 0)
                    }
                }
        except (json.JSONDecodeError, KeyError):
            pass
        
        return {
            "check": "test_coverage",
            "status": "error",
            "error": "Unable to determine coverage"
        }
    
    def check_security_compliance(self) -> Dict[str, Any]:
        """Check security compliance."""
        print("Checking security compliance...")
        
        security_issues = 0
        security_details = {}
        
        # Check for security scan results
        security_files = [
            "bandit-report.json",
            "safety-report.json", 
            "pip-audit-report.json"
        ]
        
        for security_file in security_files:
            if Path(security_file).exists():
                try:
                    with open(security_file) as f:
                        security_data = json.load(f)
                    
                    if "bandit" in security_file:
                        issues = len(security_data.get("results", []))
                        high_issues = len([r for r in security_data.get("results", []) 
                                         if r.get("issue_severity") == "HIGH"])
                        security_issues += high_issues  # Only count high severity
                        security_details["bandit"] = {"total": issues, "high": high_issues}
                    
                    elif "safety" in security_file:
                        vulnerabilities = len(security_data.get("vulnerabilities", []))
                        security_issues += vulnerabilities
                        security_details["safety"] = {"vulnerabilities": vulnerabilities}
                    
                    elif "pip-audit" in security_file:
                        vulnerabilities = len(security_data.get("vulnerabilities", []))
                        security_issues += vulnerabilities
                        security_details["pip_audit"] = {"vulnerabilities": vulnerabilities}
                
                except (json.JSONDecodeError, KeyError):
                    continue
        
        status = "pass" if security_issues == 0 else "fail"
        
        return {
            "check": "security_compliance",
            "status": status,
            "security_issues": security_issues,
            "details": security_details
        }
    
    def check_license_compliance(self) -> Dict[str, Any]:
        """Check license compliance."""
        print("Checking license compliance...")
        
        try:
            # Generate license report
            result = self.run_command(["pip-licenses", "--format=json"])
            if result:
                license_data = json.loads(result)
                
                violations = []
                unknown_licenses = []
                
                for package in license_data:
                    license_name = package.get("License", "Unknown")
                    package_name = package.get("Name", "Unknown")
                    
                    if license_name in self.config["blocked_licenses"]:
                        violations.append({
                            "package": package_name,
                            "license": license_name,
                            "type": "blocked"
                        })
                    elif license_name not in self.config["allowed_licenses"] and license_name != "Unknown":
                        unknown_licenses.append({
                            "package": package_name,
                            "license": license_name,
                            "type": "unknown"
                        })
                
                status = "pass" if not violations and not unknown_licenses else "fail"
                
                return {
                    "check": "license_compliance",
                    "status": status,
                    "violations": violations,
                    "unknown_licenses": unknown_licenses,
                    "total_packages": len(license_data)
                }
        
        except (json.JSONDecodeError, KeyError):
            pass
        
        return {
            "check": "license_compliance",
            "status": "error",
            "error": "Unable to check licenses"
        }
    
    def check_documentation_compliance(self) -> Dict[str, Any]:
        """Check documentation compliance."""
        print("Checking documentation compliance...")
        
        required_docs = [
            "README.md",
            "SECURITY.md",
            "CODE_OF_CONDUCT.md",
            "CONTRIBUTING.md",
            "LICENSE"
        ]
        
        missing_docs = []
        for doc in required_docs:
            if not Path(doc).exists():
                missing_docs.append(doc)
        
        # Check for API documentation
        api_docs_exist = (
            Path("docs/").exists() or 
            Path("API.md").exists() or
            any(Path(".").glob("**/api.md"))
        )
        
        if not api_docs_exist:
            missing_docs.append("API documentation")
        
        status = "pass" if not missing_docs else "fail"
        
        return {
            "check": "documentation_compliance",
            "status": status,
            "missing_documentation": missing_docs,
            "required_docs": required_docs
        }
    
    def check_sbom_compliance(self) -> Dict[str, Any]:
        """Check SBOM (Software Bill of Materials) compliance."""
        print("Checking SBOM compliance...")
        
        sbom_files = ["sbom.json", "sbom.xml", "bom.json"]
        sbom_exists = any(Path(f).exists() for f in sbom_files)
        
        if sbom_exists:
            # Validate SBOM content
            for sbom_file in sbom_files:
                if Path(sbom_file).exists():
                    try:
                        with open(sbom_file) as f:
                            sbom_data = json.load(f)
                        
                        components = sbom_data.get("components", [])
                        metadata = sbom_data.get("metadata", {})
                        
                        return {
                            "check": "sbom_compliance",
                            "status": "pass",
                            "sbom_file": sbom_file,
                            "components_count": len(components),
                            "last_updated": metadata.get("timestamp", "unknown")
                        }
                    except (json.JSONDecodeError, KeyError):
                        continue
        
        return {
            "check": "sbom_compliance",
            "status": "fail",
            "error": "No valid SBOM found",
            "required_files": sbom_files
        }
    
    def generate_compliance_score(self) -> int:
        """Calculate overall compliance score."""
        total_checks = len(self.results["compliance_checks"])
        passed_checks = len([c for c in self.results["compliance_checks"].values() 
                           if c.get("status") == "pass"])
        
        if total_checks == 0:
            return 0
        
        return int((passed_checks / total_checks) * 100)
    
    def run_all_checks(self):
        """Run all compliance checks."""
        print("Starting compliance validation...")
        
        checks = [
            self.check_test_coverage,
            self.check_security_compliance,
            self.check_license_compliance,
            self.check_documentation_compliance,
            self.check_sbom_compliance
        ]
        
        for check_func in checks:
            try:
                result = check_func()
                check_name = result["check"]
                self.results["compliance_checks"][check_name] = result
            except Exception as e:
                print(f"Error running {check_func.__name__}: {e}")
        
        # Calculate overall status
        compliance_score = self.generate_compliance_score()
        failed_checks = [name for name, result in self.results["compliance_checks"].items() 
                        if result.get("status") == "fail"]
        
        if compliance_score >= 90:
            self.results["overall_status"] = "compliant"
        elif compliance_score >= 70:
            self.results["overall_status"] = "partially_compliant"
        else:
            self.results["overall_status"] = "non_compliant"
        
        self.results["compliance_score"] = compliance_score
        self.results["failed_checks"] = failed_checks
        
        # Save results
        report_file = Path("compliance_report.json")
        with open(report_file, "w") as f:
            json.dump(self.results, f, indent=2, sort_keys=True)
        
        print(f"\nCompliance check completed!")
        print(f"Overall status: {self.results['overall_status']}")
        print(f"Compliance score: {compliance_score}%")
        if failed_checks:
            print(f"Failed checks: {', '.join(failed_checks)}")
        print(f"Detailed report: {report_file}")
        
        # Exit with error code if not compliant
        if self.results["overall_status"] == "non_compliant":
            sys.exit(1)

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run compliance checks")
    parser.add_argument("--config", help="Compliance configuration file")
    parser.add_argument("--strict", action="store_true", 
                       help="Fail on any compliance issue")
    
    args = parser.parse_args()
    
    checker = ComplianceChecker(args.config)
    checker.run_all_checks()
    
    if args.strict and checker.results["compliance_score"] < 100:
        print("Strict mode: Failing due to compliance issues")
        sys.exit(1)

if __name__ == "__main__":
    main()