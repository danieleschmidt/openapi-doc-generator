"""
Simple test suite for autonomous SDLC components.
"""

import pytest
import asyncio
import time
from pathlib import Path


def test_imports():
    """Test that all autonomous components can be imported."""
    
    # Test AI Documentation Agent
    from openapi_doc_generator.ai_documentation_agent import (
        AIDocumentationAgent, 
        DocumentationContext,
        AdvancedReasoningEngine
    )
    
    agent = AIDocumentationAgent()
    assert agent is not None
    
    context = DocumentationContext(
        codebase_structure={"test": "structure"},
        business_domain="api",
        user_personas=["developer"],
        complexity_level="simple",
        performance_requirements={},
        security_context={},
        compliance_requirements=[]
    )
    assert context.business_domain == "api"
    
    engine = AdvancedReasoningEngine()
    assert engine is not None


def test_code_analyzer_import():
    """Test autonomous code analyzer import."""
    from openapi_doc_generator.autonomous_code_analyzer import (
        AutonomousCodeAnalyzer,
        CodePatternDetector,
        SemanticAnalyzer
    )
    
    analyzer = AutonomousCodeAnalyzer()
    assert analyzer is not None
    
    detector = CodePatternDetector()
    assert detector is not None
    
    semantic = SemanticAnalyzer()
    assert semantic is not None


def test_reliability_engine_import():
    """Test reliability engine import."""
    from openapi_doc_generator.autonomous_reliability_engine import (
        AutonomousReliabilityEngine,
        FailureType,
        SystemMetrics
    )
    
    engine = AutonomousReliabilityEngine()
    assert engine is not None
    
    # Test enum
    failure_type = FailureType.CPU_OVERLOAD
    assert failure_type.value == "cpu_overload"
    
    # Test metrics
    metrics = SystemMetrics(
        cpu_usage=50.0,
        memory_usage=60.0,
        disk_usage=40.0,
        network_latency=10.0,
        active_connections=5,
        error_rate=0.1,
        response_time=100.0
    )
    assert metrics.cpu_usage == 50.0


def test_security_guardian_import():
    """Test security guardian import."""
    from openapi_doc_generator.advanced_security_guardian import (
        AdvancedSecurityGuardian,
        ThreatType,
        ThreatLevel
    )
    
    guardian = AdvancedSecurityGuardian()
    assert guardian is not None
    
    # Test enums
    threat_type = ThreatType.SQL_INJECTION
    assert threat_type.value == "sql_injection"
    
    threat_level = ThreatLevel.HIGH
    assert threat_level.value == "high"


def test_performance_engine_import():
    """Test performance engine import."""
    from openapi_doc_generator.quantum_performance_engine import (
        QuantumPerformanceEngine,
        OptimizationStrategy,
        PerformanceSnapshot
    )
    
    engine = QuantumPerformanceEngine()
    assert engine is not None
    
    # Test optimization strategy
    strategy = OptimizationStrategy.QUANTUM_ANNEALING
    assert strategy.value == "quantum_annealing"
    
    # Test performance snapshot
    snapshot = PerformanceSnapshot(
        timestamp=time.time(),
        throughput=100.0,
        avg_latency=50.0,
        p95_latency=75.0,
        p99_latency=100.0,
        cpu_usage=30.0,
        memory_usage=40.0,
        cache_hit_rate=90.0,
        active_connections=10,
        queue_depth=2,
        error_rate=0.1
    )
    assert snapshot.throughput == 100.0


@pytest.mark.asyncio
async def test_ai_documentation_basic():
    """Test basic AI documentation functionality."""
    from openapi_doc_generator.ai_documentation_agent import (
        AIDocumentationAgent,
        DocumentationContext
    )
    
    agent = AIDocumentationAgent()
    context = DocumentationContext(
        codebase_structure={"modules": ["test"]},
        business_domain="api",
        user_personas=["developer"],
        complexity_level="simple",
        performance_requirements={},
        security_context={},
        compliance_requirements=[]
    )
    
    result = await agent.generate_documentation(context)
    
    assert result is not None
    assert result.documentation is not None
    assert len(result.documentation) > 0
    assert result.quality_score >= 0.0
    assert len(result.reasoning_trace) > 0


@pytest.mark.asyncio
async def test_security_analysis_basic():
    """Test basic security analysis."""
    from openapi_doc_generator.advanced_security_guardian import AdvancedSecurityGuardian
    
    guardian = AdvancedSecurityGuardian()
    
    # Test with clean request
    clean_request = {
        "username": "testuser",
        "action": "login",
        "source_ip": "192.168.1.100"
    }
    
    threats, responses = await guardian.analyze_request(clean_request)
    
    # Clean request should have no threats or minimal threats
    assert isinstance(threats, list)
    assert isinstance(responses, list)
    
    # Test security status
    status = guardian.get_security_status()
    assert "status" in status
    assert status["status"] == "active"


def test_performance_metrics():
    """Test performance metrics collection."""
    from openapi_doc_generator.quantum_performance_engine import QuantumPerformanceEngine
    
    engine = QuantumPerformanceEngine()
    
    # Test performance report
    report = engine.get_performance_report()
    
    # Should return report structure even with no data
    assert isinstance(report, dict)
    
    # Test cache manager
    cache_metrics = engine.cache_manager.get_cache_metrics()
    assert "hit_rate" in cache_metrics
    assert "cache_size" in cache_metrics


def test_reliability_reporting():
    """Test reliability reporting."""
    from openapi_doc_generator.autonomous_reliability_engine import AutonomousReliabilityEngine
    
    engine = AutonomousReliabilityEngine()
    
    # Test reliability report
    report = engine.get_reliability_report()
    
    assert isinstance(report, dict)
    assert "uptime_seconds" in report
    assert "system_resilience_score" in report
    assert "total_failures" in report


@pytest.mark.asyncio
async def test_integration_basic():
    """Test basic integration between components."""
    from openapi_doc_generator.ai_documentation_agent import AIDocumentationAgent, DocumentationContext
    from openapi_doc_generator.advanced_security_guardian import AdvancedSecurityGuardian
    from openapi_doc_generator.quantum_performance_engine import QuantumPerformanceEngine
    
    # Initialize components
    ai_agent = AIDocumentationAgent()
    security_guardian = AdvancedSecurityGuardian()
    performance_engine = QuantumPerformanceEngine()
    
    # Test they can work together
    doc_context = DocumentationContext(
        codebase_structure={"test": "integration"},
        business_domain="api",
        user_personas=["developer"],
        complexity_level="simple",
        performance_requirements={},
        security_context={},
        compliance_requirements=[]
    )
    
    # Generate documentation
    doc_result = await ai_agent.generate_documentation(doc_context)
    assert doc_result.quality_score > 0
    
    # Check security
    security_status = security_guardian.get_security_status()
    assert security_status["status"] == "active"
    
    # Check performance
    performance_report = performance_engine.get_performance_report()
    assert isinstance(performance_report, dict)
    
    print(f"Integration test completed successfully!")
    print(f"- Documentation quality: {doc_result.quality_score:.2f}")
    print(f"- Security status: {security_status['status']}")
    print(f"- Performance monitoring: {'active' if performance_report else 'inactive'}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])