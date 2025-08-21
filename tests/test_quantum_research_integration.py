"""
Integration tests for quantum-enhanced research modules.

This test suite validates the integration of quantum semantic analysis,
ML-enhanced schema inference, and research benchmarking with the existing
OpenAPI documentation generator infrastructure.
"""

import pytest
import ast
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from openapi_doc_generator.quantum_semantic_analyzer import (
    QuantumSemanticAnalyzer,
    QuantumFeatureEncoder,
    GraphNeuralNetworkAnalyzer,
    SemanticAnalysisResult,
    QuantumSemanticFeature
)
from openapi_doc_generator.ml_schema_inference import (
    MLEnhancedSchemaInferencer,
    BayesianTypeInferencer,
    MetaLearningSchemaInferencer,
    EvolutionarySchemaPredictor,
    ProbabilisticType
)
from openapi_doc_generator.research_benchmark_suite import (
    ResearchBenchmarkSuite,
    GroundTruthGenerator,
    StatisticalValidator,
    PerformanceBenchmarker,
    ExperimentalDataset
)
from openapi_doc_generator.documentator import APIDocumentator
from openapi_doc_generator.discovery import RouteDiscoverer


class TestQuantumSemanticIntegration:
    """Test quantum semantic analyzer integration with existing systems."""
    
    @pytest.fixture
    def sample_api_file(self):
        """Create a sample API file for testing."""
        content = '''
from flask import Flask, request, jsonify
from dataclasses import dataclass
from typing import Optional

app = Flask(__name__)

@dataclass
class User:
    id: int
    name: str
    email: Optional[str] = None

@app.route('/users', methods=['GET'])
def get_users():
    """Get all users."""
    return jsonify([])

@app.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id: int):
    """Get user by ID."""
    user = User(id=user_id, name="Test", email="test@example.com")
    return jsonify(user.__dict__)

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            return f.name
    
    def test_quantum_analyzer_initialization(self):
        """Test quantum semantic analyzer can be initialized."""
        analyzer = QuantumSemanticAnalyzer()
        assert analyzer is not None
        assert analyzer.gnn_analyzer is not None
        assert analyzer.config == {}
    
    def test_quantum_feature_encoder(self):
        """Test quantum feature encoder functionality."""
        encoder = QuantumFeatureEncoder(feature_dim=128)
        
        # Test with simple AST node
        code = "def test_function(): pass"
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        feature = encoder.encode_node(func_node)
        
        assert isinstance(feature, QuantumSemanticFeature)
        assert feature.semantic_type in ['function', 'async']
        assert 0 <= feature.confidence <= 1
        assert len(feature.embedding) == 128
        assert feature.quantum_state is not None
    
    def test_quantum_analyzer_file_analysis(self, sample_api_file):
        """Test quantum analyzer can analyze real API files."""
        analyzer = QuantumSemanticAnalyzer()
        
        result = analyzer.analyze_file(sample_api_file)
        
        assert result is not None
        assert isinstance(result, SemanticAnalysisResult)
        assert len(result.node_features) > 0
        assert result.confidence_score > 0
        
        # Check for expected semantic types
        semantic_types = [f.semantic_type for f in result.node_features.values()]
        assert 'function' in semantic_types or 'route' in semantic_types
        
        # Validate quantum metrics
        assert 'quantum_coherence' in result.quantum_metrics
        assert 'semantic_diversity' in result.quantum_metrics
    
    def test_quantum_api_insights(self, sample_api_file):
        """Test quantum analyzer can extract API insights."""
        analyzer = QuantumSemanticAnalyzer()
        result = analyzer.analyze_file(sample_api_file)
        
        insights = analyzer.get_api_insights(result)
        
        assert 'total_nodes' in insights
        assert 'semantic_clusters' in insights
        assert 'confidence_score' in insights
        assert 'potential_endpoints' in insights
        assert insights['total_nodes'] > 0
    
    def test_quantum_caching(self, sample_api_file):
        """Test quantum feature caching works correctly."""
        analyzer = QuantumSemanticAnalyzer()
        
        # First analysis
        result1 = analyzer.analyze_file(sample_api_file)
        
        # Second analysis should use cache
        result2 = analyzer.analyze_file(sample_api_file)
        
        # Results should be consistent
        assert result1.confidence_score == result2.confidence_score
        assert len(result1.node_features) == len(result2.node_features)


class TestMLSchemaInferenceIntegration:
    """Test ML-enhanced schema inference integration."""
    
    @pytest.fixture
    def sample_schema_examples(self):
        """Sample data for schema learning."""
        return [
            {'id': 1, 'name': 'John', 'email': 'john@example.com', 'age': 30},
            {'id': 2, 'name': 'Jane', 'email': 'jane@example.com', 'age': 25},
            {'id': 3, 'name': 'Bob', 'email': 'bob@example.com', 'age': 35}
        ]
    
    def test_ml_inferencer_initialization(self):
        """Test ML schema inferencer initialization."""
        inferencer = MLEnhancedSchemaInferencer()
        
        assert inferencer.bayesian_inferencer is not None
        assert inferencer.meta_learner is not None
        assert inferencer.evolution_predictor is not None
    
    def test_bayesian_type_inference(self):
        """Test Bayesian type inference functionality."""
        inferencer = BayesianTypeInferencer()
        
        # Test with constant node
        code = "x = 42"
        tree = ast.parse(code)
        assign_node = tree.body[0]
        value_node = assign_node.value
        
        prob_type = inferencer.infer_type(value_node)
        
        assert isinstance(prob_type, ProbabilisticType)
        assert prob_type.primary_type in ['int', 'float']
        assert 0 <= prob_type.confidence <= 1
        assert prob_type.uncertainty >= 0
        assert len(prob_type.evidence) > 0
    
    def test_ml_schema_inference_with_ast(self):
        """Test ML schema inference with AST input."""
        inferencer = MLEnhancedSchemaInferencer()
        
        code = '''
class User:
    def __init__(self):
        self.id: int = 1
        self.name: str = "test"
        self.email: Optional[str] = None
'''
        tree = ast.parse(code)
        
        schema = inferencer.infer_schema(tree)
        
        assert isinstance(schema, dict)
        # Should find some schema elements
        if schema:  # May be empty for simple cases
            for field_name, prob_type in schema.items():
                assert isinstance(prob_type, ProbabilisticType)
                assert prob_type.confidence > 0
    
    def test_meta_learning_from_examples(self, sample_schema_examples):
        """Test meta-learning from example schemas."""
        inferencer = MLEnhancedSchemaInferencer()
        
        # Learn from examples
        inferencer.learn_from_examples(sample_schema_examples)
        
        # Try few-shot inference
        new_examples = [{'id': 4, 'name': 'Alice', 'email': 'alice@example.com'}]
        schema = inferencer.meta_learner.infer_schema_few_shot(new_examples)
        
        assert isinstance(schema, dict)
        if schema:
            assert 'id' in schema or 'name' in schema or 'email' in schema
    
    def test_evolutionary_schema_prediction(self):
        """Test evolutionary schema prediction."""
        predictor = EvolutionarySchemaPredictor()
        
        # Create mock schema history
        schema_history = [
            {'id': {'type': 'int'}, 'name': {'type': 'str'}},
            {'id': {'type': 'int'}, 'name': {'type': 'str'}, 'email': {'type': 'str'}},
            {'id': {'type': 'int'}, 'name': {'type': 'str'}, 'email': {'type': 'Optional[str]'}}
        ]
        
        prediction = predictor.predict_evolution(schema_history)
        
        assert prediction.current_schema == schema_history[-1]
        assert isinstance(prediction.predicted_changes, list)
        assert 0 <= prediction.compatibility_score <= 1
        assert 0 <= prediction.evolution_confidence <= 1
    
    def test_schema_quality_analysis(self):
        """Test schema quality analysis."""
        inferencer = MLEnhancedSchemaInferencer()
        
        # Create mock schema
        schema = {
            'id': ProbabilisticType('int', 0.95, {}, 0.05, ['constant value']),
            'name': ProbabilisticType('str', 0.85, {'Optional[str]': 0.1}, 0.15, ['string literal']),
            'email': ProbabilisticType('Optional[str]', 0.9, {}, 0.1, ['email pattern'])
        }
        
        quality = inferencer.analyze_schema_quality(schema)
        
        assert 'average_confidence' in quality
        assert 'total_fields' in quality
        assert 'overall_quality' in quality
        assert quality['total_fields'] == 3
        assert 0 <= quality['overall_quality'] <= 1


class TestResearchBenchmarkIntegration:
    """Test research benchmark suite integration."""
    
    def test_benchmark_suite_initialization(self):
        """Test benchmark suite can be initialized."""
        suite = ResearchBenchmarkSuite()
        
        assert suite.ground_truth_generator is not None
        assert suite.benchmarker is not None
        assert suite.quantum_analyzer is not None
        assert suite.ml_inferencer is not None
    
    def test_ground_truth_generation(self):
        """Test synthetic dataset generation."""
        generator = GroundTruthGenerator()
        
        dataset = generator.create_synthetic_dataset(size=5, complexity='easy')
        
        assert isinstance(dataset, ExperimentalDataset)
        assert len(dataset.file_paths) == 5
        assert dataset.difficulty_level == 'easy'
        assert len(dataset.ground_truth_schemas) == 5
        assert len(dataset.ground_truth_semantics) == 5
    
    def test_statistical_validator(self):
        """Test statistical validation functionality."""
        validator = StatisticalValidator()
        
        # Generate mock results
        baseline_results = [0.7, 0.72, 0.68, 0.71, 0.69]
        improved_results = [0.85, 0.87, 0.83, 0.86, 0.84]
        
        validation = validator.validate_improvement(baseline_results, improved_results)
        
        assert 'baseline_mean' in validation
        assert 'improved_mean' in validation
        assert 'p_value' in validation
        assert 'is_significant' in validation
        assert 'cohens_d' in validation
        assert validation['improved_mean'] > validation['baseline_mean']
    
    def test_performance_benchmarker(self):
        """Test performance benchmarking framework."""
        benchmarker = PerformanceBenchmarker()
        generator = GroundTruthGenerator()
        
        # Create small test dataset
        dataset = generator.create_synthetic_dataset(size=2, complexity='easy')
        
        # Define simple test methods
        def mock_method1(dataset):
            return {'result': 'method1', 'accuracy': 0.8}
        
        def mock_method2(dataset):
            return {'result': 'method2', 'accuracy': 0.85}
        
        methods = {
            'method1': mock_method1,
            'method2': mock_method2
        }
        
        # Run benchmark with minimal iterations
        results = benchmarker.benchmark_methods(methods, dataset, iterations=2)
        
        assert len(results) == 2
        assert 'method1' in results
        assert 'method2' in results
        
        for method_name, result in results.items():
            assert result.metrics.execution_time > 0
            assert len(result.raw_results) == 2
    
    @pytest.mark.slow
    def test_comprehensive_benchmark_small(self):
        """Test comprehensive benchmark with minimal configuration."""
        suite = ResearchBenchmarkSuite()
        
        # Small test configuration
        config = {
            'easy_size': 2,
            'medium_size': 2,
            'hard_size': 1,
            'iterations': 1
        }
        
        # Mock the methods to avoid long execution
        with patch.object(suite, '_quantum_semantic_method') as mock_quantum, \
             patch.object(suite, '_ml_schema_method') as mock_ml, \
             patch.object(suite, '_traditional_schema_method') as mock_trad:
            
            mock_quantum.return_value = [Mock()]
            mock_ml.return_value = [{}]
            mock_trad.return_value = [{}]
            
            results = suite.run_comprehensive_benchmark(config)
            
            assert 'experiment_metadata' in results
            assert 'summary_statistics' in results
            assert 'detailed_results' in results
            assert len(results['detailed_results']) >= 3  # easy, medium, hard


class TestSystemIntegration:
    """Test integration with existing OpenAPI documentation system."""
    
    @pytest.fixture
    def api_documentator(self):
        """Create API documentator instance."""
        return APIDocumentator()
    
    def test_quantum_integration_with_documentator(self, sample_api_file, api_documentator):
        """Test quantum analyzer integration with existing documentator."""
        # Use quantum analyzer as part of documentation process
        quantum_analyzer = QuantumSemanticAnalyzer()
        
        # Analyze file with quantum methods
        quantum_result = quantum_analyzer.analyze_file(sample_api_file)
        
        # Should not interfere with normal documentation
        docs = api_documentator.analyze_app(sample_api_file)
        
        assert docs is not None
        assert quantum_result is not None
        
        # Both should work without conflicts
        assert quantum_result.confidence_score > 0
    
    def test_ml_integration_with_discovery(self, sample_api_file):
        """Test ML schema inference integration with route discovery."""
        discoverer = RouteDiscoverer()
        ml_inferencer = MLEnhancedSchemaInferencer()
        
        # Discover routes normally
        routes = discoverer.discover_routes(sample_api_file)
        
        # Enhance with ML schema inference
        if routes:
            for route in routes:
                # ML analysis should complement route discovery
                tree = ast.parse(open(sample_api_file).read())
                schema = ml_inferencer.infer_schema(tree)
                
                # Should not break existing functionality
                assert route.path is not None
                assert isinstance(schema, dict)
    
    def test_benchmark_integration_with_existing_tests(self):
        """Test benchmark suite doesn't conflict with existing test infrastructure."""
        # Import existing test modules to ensure no conflicts
        try:
            from openapi_doc_generator.cli import main as cli_main
            from openapi_doc_generator.validator import SpecValidator
            
            # Should import without issues
            assert cli_main is not None
            assert SpecValidator is not None
            
            # Research modules should also import
            suite = ResearchBenchmarkSuite()
            assert suite is not None
            
        except ImportError as e:
            pytest.fail(f"Import conflict detected: {e}")
    
    def test_quantum_memory_efficiency(self, sample_api_file):
        """Test quantum modules don't consume excessive memory."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run quantum analysis
        analyzer = QuantumSemanticAnalyzer()
        result = analyzer.analyze_file(sample_api_file)
        
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        # Should not use excessive memory (threshold: 100MB)
        assert memory_increase < 100, f"Memory usage too high: {memory_increase}MB"
        
        # Cleanup
        del analyzer, result
        gc.collect()
    
    def test_error_handling_integration(self):
        """Test error handling in research modules."""
        analyzer = QuantumSemanticAnalyzer()
        
        # Test with non-existent file
        result = analyzer.analyze_file("/nonexistent/file.py")
        assert result is None  # Should handle gracefully
        
        # Test with invalid Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("invalid python syntax {{{")
            invalid_file = f.name
        
        result = analyzer.analyze_file(invalid_file)
        assert result is None  # Should handle parsing errors gracefully


class TestConfigurationIntegration:
    """Test configuration and customization integration."""
    
    def test_quantum_analyzer_configuration(self):
        """Test quantum analyzer with custom configuration."""
        config = {
            'embedding_dim': 512,
            'quantum_levels': 4
        }
        
        analyzer = QuantumSemanticAnalyzer(config)
        assert analyzer.config == config
        assert analyzer.gnn_analyzer.embedding_dim == 512
    
    def test_ml_inferencer_configuration(self):
        """Test ML inferencer with custom configuration."""
        config = {
            'feature_dim': 256,
            'population_size': 20
        }
        
        inferencer = MLEnhancedSchemaInferencer(config)
        assert inferencer.config == config
        assert inferencer.evolution_predictor.population_size == 20
    
    def test_benchmark_suite_configuration(self):
        """Test benchmark suite with configuration."""
        suite = ResearchBenchmarkSuite()
        
        # Test with various configurations
        configs = [
            {'iterations': 1, 'easy_size': 1},
            {'iterations': 2, 'medium_size': 2}
        ]
        
        for config in configs:
            # Should handle different configurations without errors
            datasets = suite._generate_test_datasets(config)
            assert len(datasets) > 0


@pytest.mark.integration
class TestFullPipelineIntegration:
    """Test full pipeline integration of research modules."""
    
    def test_full_research_pipeline(self, sample_api_file):
        """Test complete research pipeline from analysis to benchmarking."""
        # Step 1: Quantum semantic analysis
        quantum_analyzer = QuantumSemanticAnalyzer()
        semantic_result = quantum_analyzer.analyze_file(sample_api_file)
        
        assert semantic_result is not None
        
        # Step 2: ML schema inference
        ml_inferencer = MLEnhancedSchemaInferencer()
        tree = ast.parse(open(sample_api_file).read())
        schema_result = ml_inferencer.infer_schema(tree)
        
        assert isinstance(schema_result, dict)
        
        # Step 3: Quality analysis
        if schema_result:
            quality_metrics = ml_inferencer.analyze_schema_quality(schema_result)
            assert 'overall_quality' in quality_metrics
        
        # Step 4: Integration insights
        insights = quantum_analyzer.get_api_insights(semantic_result)
        assert 'potential_endpoints' in insights
        
        # All steps should complete without conflicts
        assert True  # Pipeline completed successfully
    
    def test_concurrent_analysis(self, sample_api_file):
        """Test concurrent analysis doesn't cause conflicts."""
        import threading
        import time
        
        results = {}
        errors = []
        
        def quantum_analysis():
            try:
                analyzer = QuantumSemanticAnalyzer()
                result = analyzer.analyze_file(sample_api_file)
                results['quantum'] = result
            except Exception as e:
                errors.append(f"Quantum error: {e}")
        
        def ml_analysis():
            try:
                inferencer = MLEnhancedSchemaInferencer()
                tree = ast.parse(open(sample_api_file).read())
                result = inferencer.infer_schema(tree)
                results['ml'] = result
            except Exception as e:
                errors.append(f"ML error: {e}")
        
        # Run concurrent analyses
        quantum_thread = threading.Thread(target=quantum_analysis)
        ml_thread = threading.Thread(target=ml_analysis)
        
        quantum_thread.start()
        ml_thread.start()
        
        quantum_thread.join(timeout=10)
        ml_thread.join(timeout=10)
        
        # Check for errors
        assert len(errors) == 0, f"Concurrent execution errors: {errors}"
        assert 'quantum' in results
        assert 'ml' in results


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])  # Exit on first failure for faster feedback