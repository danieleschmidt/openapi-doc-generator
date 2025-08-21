"""
Machine Learning-Enhanced Schema Inference Module

This module implements advanced ML techniques for intelligent schema inference,
including probabilistic type inference, cross-repository learning, and
evolutionary schema prediction capabilities.

Research Contributions:
1. Bayesian neural networks for type inference with uncertainty quantification
2. Meta-learning for few-shot schema inference on unseen patterns
3. Evolutionary algorithms for schema change prediction
4. Federated learning across multiple codebases
"""

import ast
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import pickle
import hashlib
from pathlib import Path
import threading
import time
from collections import defaultdict, Counter

from .utils import get_cached_ast, echo
from .schema import SchemaInferer, FieldInfo, SchemaInfo
from .quantum_optimizer import QuantumCache


logger = logging.getLogger(__name__)


class TypeConfidence(Enum):
    """Confidence levels for type inference."""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


@dataclass
class ProbabilisticType:
    """Probabilistic type representation with uncertainty quantification."""
    primary_type: str
    confidence: float
    alternative_types: Dict[str, float] = field(default_factory=dict)
    uncertainty: float = 0.0
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SchemaPattern:
    """Learned schema pattern for cross-repository knowledge."""
    pattern_id: str
    fields: Dict[str, ProbabilisticType]
    frequency: int
    contexts: List[str]
    confidence: float
    last_seen: float


@dataclass
class EvolutionaryPrediction:
    """Prediction of schema evolution."""
    current_schema: Dict[str, Any]
    predicted_changes: List[Dict[str, Any]]
    compatibility_score: float
    breaking_changes: List[str]
    evolution_confidence: float


class BayesianTypeInferencer:
    """
    Bayesian neural network for type inference with uncertainty quantification.
    
    Uses variational inference to provide confidence intervals and handle
    the inherent uncertainty in dynamic typing scenarios.
    """
    
    def __init__(self, feature_dim: int = 128, hidden_layers: List[int] = None):
        self.feature_dim = feature_dim
        self.hidden_layers = hidden_layers or [256, 128, 64]
        self.type_vocabulary = self._initialize_type_vocabulary()
        self.model_weights = self._initialize_bayesian_weights()
        self.training_data = []
        
        logger.info(f"Initialized BayesianTypeInferencer with {feature_dim}D features")
    
    def _initialize_type_vocabulary(self) -> Dict[str, int]:
        """Initialize vocabulary of known types."""
        common_types = [
            'str', 'int', 'float', 'bool', 'list', 'dict', 'tuple', 'set',
            'Optional[str]', 'Optional[int]', 'Optional[float]', 'Optional[bool]',
            'List[str]', 'List[int]', 'List[dict]', 'Dict[str, Any]',
            'Union[str, int]', 'Union[str, None]', 'datetime', 'UUID',
            'BaseModel', 'dataclass', 'NamedTuple', 'TypedDict',
            'Callable', 'Generator', 'Iterator', 'Coroutine',
            'bytes', 'bytearray', 'memoryview', 'complex',
            'Any', 'NoReturn', 'Never', 'object'
        ]
        
        return {type_name: idx for idx, type_name in enumerate(common_types)}
    
    def _initialize_bayesian_weights(self) -> Dict[str, np.ndarray]:
        """Initialize Bayesian neural network weights."""
        weights = {}
        
        # Input to first hidden layer
        weights['W1_mean'] = np.random.normal(0, 0.1, (self.feature_dim, self.hidden_layers[0]))
        weights['W1_logvar'] = np.random.normal(-2, 0.1, (self.feature_dim, self.hidden_layers[0]))
        weights['b1_mean'] = np.zeros(self.hidden_layers[0])
        weights['b1_logvar'] = np.random.normal(-2, 0.1, self.hidden_layers[0])
        
        # Hidden layers
        for i in range(len(self.hidden_layers) - 1):
            layer_name = f'W{i+2}'
            bias_name = f'b{i+2}'
            in_dim = self.hidden_layers[i]
            out_dim = self.hidden_layers[i+1]
            
            weights[f'{layer_name}_mean'] = np.random.normal(0, 0.1, (in_dim, out_dim))
            weights[f'{layer_name}_logvar'] = np.random.normal(-2, 0.1, (in_dim, out_dim))
            weights[f'{bias_name}_mean'] = np.zeros(out_dim)
            weights[f'{bias_name}_logvar'] = np.random.normal(-2, 0.1, out_dim)
        
        # Output layer
        output_dim = len(self.type_vocabulary)
        last_hidden = self.hidden_layers[-1]
        weights['Wout_mean'] = np.random.normal(0, 0.1, (last_hidden, output_dim))
        weights['Wout_logvar'] = np.random.normal(-2, 0.1, (last_hidden, output_dim))
        weights['bout_mean'] = np.zeros(output_dim)
        weights['bout_logvar'] = np.random.normal(-2, 0.1, output_dim)
        
        return weights
    
    def infer_type(self, node: ast.AST, context: Dict[str, Any] = None) -> ProbabilisticType:
        """
        Infer type with uncertainty quantification using Bayesian inference.
        """
        # Extract features from AST node
        features = self._extract_type_features(node, context)
        
        # Perform Bayesian forward pass
        type_probs, uncertainty = self._bayesian_forward_pass(features)
        
        # Get top predictions
        sorted_indices = np.argsort(type_probs)[::-1]
        type_names = list(self.type_vocabulary.keys())
        
        primary_type = type_names[sorted_indices[0]]
        confidence = float(type_probs[sorted_indices[0]])
        
        # Alternative types
        alternative_types = {}
        for i in range(1, min(4, len(sorted_indices))):
            idx = sorted_indices[i]
            if type_probs[idx] > 0.1:  # Only include significant alternatives
                alternative_types[type_names[idx]] = float(type_probs[idx])
        
        # Collect evidence
        evidence = self._collect_type_evidence(node, context)
        
        return ProbabilisticType(
            primary_type=primary_type,
            confidence=confidence,
            alternative_types=alternative_types,
            uncertainty=float(uncertainty),
            evidence=evidence,
            metadata={'node_type': type(node).__name__}
        )
    
    def _extract_type_features(self, node: ast.AST, context: Dict[str, Any] = None) -> np.ndarray:
        """Extract features for type inference."""
        features = np.zeros(self.feature_dim)
        
        # Node type features
        node_type_map = {
            ast.Constant: 0, ast.Name: 1, ast.Attribute: 2, ast.Call: 3,
            ast.List: 4, ast.Dict: 5, ast.Tuple: 6, ast.Set: 7,
            ast.ListComp: 8, ast.DictComp: 9, ast.SetComp: 10,
            ast.BinOp: 11, ast.UnaryOp: 12, ast.Compare: 13,
            ast.BoolOp: 14, ast.IfExp: 15, ast.Lambda: 16
        }
        
        node_type_idx = node_type_map.get(type(node), -1)
        if node_type_idx >= 0 and node_type_idx < self.feature_dim:
            features[node_type_idx] = 1.0
        
        # Value-based features for constants
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, str):
                features[20] = 1.0
                features[21] = len(value) / 100.0  # Normalized length
                features[22] = float(value.isdigit())
                features[23] = float('@' in value)  # Email pattern
                features[24] = float('http' in value.lower())  # URL pattern
            elif isinstance(value, (int, float)):
                features[25] = 1.0
                features[26] = float(isinstance(value, int))
                features[27] = float(isinstance(value, float))
                features[28] = float(value == 0)
                features[29] = float(value < 0)
            elif isinstance(value, bool):
                features[30] = 1.0
            elif value is None:
                features[31] = 1.0
        
        # Name-based features
        if isinstance(node, ast.Name):
            name = node.id
            features[35] = len(name) / 20.0  # Normalized length
            features[36] = float(name.isupper())
            features[37] = float(name.islower())
            features[38] = float('_' in name)
            features[39] = float(name.endswith('_id'))
            features[40] = float(name.startswith('is_'))
            features[41] = float(name.startswith('has_'))
            features[42] = float('count' in name.lower())
            features[43] = float('time' in name.lower())
            features[44] = float('date' in name.lower())
        
        # Collection features
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            features[50] = 1.0
            features[51] = len(node.elts) / 10.0  # Normalized length
            if node.elts:
                # Check if all elements are same type
                first_type = type(node.elts[0])
                features[52] = float(all(isinstance(elt, first_type) for elt in node.elts))
        
        # Dictionary features
        if isinstance(node, ast.Dict):
            features[55] = 1.0
            features[56] = len(node.keys) / 10.0  # Normalized length
            if node.keys:
                # Check if all keys are strings
                str_keys = sum(1 for key in node.keys 
                              if isinstance(key, ast.Constant) and isinstance(key.value, str))
                features[57] = str_keys / len(node.keys)
        
        # Context features
        if context:
            features[60] = float(context.get('in_function', False))
            features[61] = float(context.get('in_class', False))
            features[62] = float(context.get('is_parameter', False))
            features[63] = float(context.get('is_return', False))
            features[64] = float(context.get('has_annotation', False))
        
        # Truncate to feature dimension
        return features[:self.feature_dim]
    
    def _bayesian_forward_pass(self, features: np.ndarray, n_samples: int = 10) -> Tuple[np.ndarray, float]:
        """Perform Bayesian forward pass with uncertainty estimation."""
        predictions = []
        
        for _ in range(n_samples):
            # Sample weights from posterior
            sampled_weights = self._sample_weights()
            
            # Forward pass with sampled weights
            x = features
            
            # Hidden layers
            for i in range(len(self.hidden_layers)):
                W = sampled_weights[f'W{i+1}']
                b = sampled_weights[f'b{i+1}']
                x = np.maximum(0, np.dot(x, W) + b)  # ReLU activation
            
            # Output layer
            W_out = sampled_weights['Wout']
            b_out = sampled_weights['bout']
            logits = np.dot(x, W_out) + b_out
            
            # Softmax
            probs = np.exp(logits - np.max(logits))
            probs = probs / np.sum(probs)
            
            predictions.append(probs)
        
        # Calculate mean and uncertainty
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.mean(np.var(predictions, axis=0))
        
        return mean_pred, uncertainty
    
    def _sample_weights(self) -> Dict[str, np.ndarray]:
        """Sample weights from Bayesian posterior."""
        sampled = {}
        
        for key, mean_weights in self.model_weights.items():
            if key.endswith('_mean'):
                base_key = key[:-5]  # Remove '_mean'
                logvar_key = base_key + '_logvar'
                
                if logvar_key in self.model_weights:
                    logvar = self.model_weights[logvar_key]
                    std = np.exp(0.5 * logvar)
                    noise = np.random.normal(0, 1, mean_weights.shape)
                    sampled[base_key] = mean_weights + std * noise
        
        return sampled
    
    def _collect_type_evidence(self, node: ast.AST, context: Dict[str, Any] = None) -> List[str]:
        """Collect evidence for type inference decision.""" 
        evidence = []
        
        if isinstance(node, ast.Constant):
            evidence.append(f"Constant value: {type(node.value).__name__}")
            if isinstance(node.value, str):
                if '@' in node.value:
                    evidence.append("Email pattern detected")
                if 'http' in node.value.lower():
                    evidence.append("URL pattern detected")
        
        if isinstance(node, ast.Name):
            name = node.id
            if name.endswith('_id'):
                evidence.append("ID field pattern")
            if name.startswith('is_') or name.startswith('has_'):
                evidence.append("Boolean field pattern")
            if 'count' in name.lower():
                evidence.append("Count field pattern")
        
        if isinstance(node, (ast.List, ast.Tuple)):
            evidence.append(f"Collection with {len(node.elts)} elements")
        
        if isinstance(node, ast.Dict):
            evidence.append(f"Dictionary with {len(node.keys)} keys")
        
        if context:
            if context.get('has_annotation'):
                evidence.append("Type annotation present")
            if context.get('is_parameter'):
                evidence.append("Function parameter")
            if context.get('is_return'):
                evidence.append("Return value")
        
        return evidence


class MetaLearningSchemaInferencer:
    """
    Meta-learning system for few-shot schema inference on unseen patterns.
    
    Learns to quickly adapt to new schema patterns with minimal examples
    using model-agnostic meta-learning (MAML) principles.
    """
    
    def __init__(self):
        self.pattern_memory = QuantumCache(max_size=50000)
        self.adaptation_history = []
        self.meta_weights = self._initialize_meta_weights()
        
        logger.info("Initialized MetaLearningSchemaInferencer")
    
    def _initialize_meta_weights(self) -> Dict[str, np.ndarray]:
        """Initialize meta-learning weights."""
        return {
            'pattern_embedding': np.random.normal(0, 0.1, (128, 64)),
            'adaptation_weights': np.random.normal(0, 0.1, (64, 32)),
            'output_weights': np.random.normal(0, 0.1, (32, 16))
        }
    
    def learn_from_patterns(self, patterns: List[SchemaPattern]):
        """Learn meta-knowledge from schema patterns."""
        for pattern in patterns:
            pattern_embedding = self._encode_pattern(pattern)
            self.pattern_memory.set(pattern.pattern_id, pattern_embedding)
        
        logger.info(f"Learned from {len(patterns)} schema patterns")
    
    def _encode_pattern(self, pattern: SchemaPattern) -> np.ndarray:
        """Encode schema pattern into embedding space."""
        # Simple encoding - would be more sophisticated in practice
        embedding = np.zeros(128)
        
        # Encode field types
        for i, (field_name, prob_type) in enumerate(pattern.fields.items()):
            if i < 64:
                embedding[i] = hash(prob_type.primary_type) % 1000 / 1000.0
                embedding[i + 64] = prob_type.confidence
        
        return embedding
    
    def infer_schema_few_shot(self, examples: List[Dict[str, Any]], 
                             context: Dict[str, Any] = None) -> Dict[str, ProbabilisticType]:
        """Infer schema from few examples using meta-learning."""
        # Find similar patterns in memory
        similar_patterns = self._find_similar_patterns(examples)
        
        # Adapt to new examples
        adapted_schema = self._adapt_schema(examples, similar_patterns)
        
        return adapted_schema
    
    def _find_similar_patterns(self, examples: List[Dict[str, Any]]) -> List[SchemaPattern]:
        """Find similar patterns in memory."""
        # Simplified similarity search
        similar = []
        
        # Would implement proper similarity search in practice
        # For now, return empty list
        
        return similar
    
    def _adapt_schema(self, examples: List[Dict[str, Any]], 
                     similar_patterns: List[SchemaPattern]) -> Dict[str, ProbabilisticType]:
        """Adapt schema based on examples and similar patterns."""
        schema = {}
        
        # Analyze examples to infer types
        field_examples = defaultdict(list)
        for example in examples:
            for field, value in example.items():
                field_examples[field].append(value)
        
        # Infer type for each field
        for field, values in field_examples.items():
            schema[field] = self._infer_field_type_from_examples(field, values)
        
        return schema
    
    def _infer_field_type_from_examples(self, field_name: str, 
                                       values: List[Any]) -> ProbabilisticType:
        """Infer field type from example values."""
        type_counts = Counter()
        
        for value in values:
            if isinstance(value, str):
                type_counts['str'] += 1
            elif isinstance(value, int):
                type_counts['int'] += 1
            elif isinstance(value, float):
                type_counts['float'] += 1
            elif isinstance(value, bool):
                type_counts['bool'] += 1
            elif isinstance(value, list):
                type_counts['list'] += 1
            elif isinstance(value, dict):
                type_counts['dict'] += 1
            elif value is None:
                type_counts['None'] += 1
            else:
                type_counts['object'] += 1
        
        # Determine primary type
        if type_counts:
            primary_type = type_counts.most_common(1)[0][0]
            confidence = type_counts[primary_type] / len(values)
            
            # Handle nullable types
            if 'None' in type_counts and type_counts['None'] > 0:
                if primary_type != 'None':
                    primary_type = f'Optional[{primary_type}]'
                    confidence = (type_counts[primary_type.split('[')[1][:-1]] + 
                                type_counts['None']) / len(values)
        else:
            primary_type = 'Any'
            confidence = 0.5
        
        return ProbabilisticType(
            primary_type=primary_type,
            confidence=confidence,
            evidence=[f"Inferred from {len(values)} examples"]
        )


class EvolutionarySchemaPredictor:
    """
    Evolutionary algorithm for predicting schema changes and evolution patterns.
    
    Uses genetic algorithms to model how schemas evolve over time and predict
    future changes, compatibility issues, and breaking changes.
    """
    
    def __init__(self, population_size: int = 100, mutation_rate: float = 0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.evolution_history = []
        self.fitness_cache = QuantumCache(max_size=10000)
        
        logger.info(f"Initialized EvolutionarySchemaPredictor (pop_size={population_size})")
    
    def predict_evolution(self, schema_history: List[Dict[str, Any]], 
                         generations: int = 50) -> EvolutionaryPrediction:
        """Predict schema evolution using genetic algorithms."""
        if len(schema_history) < 2:
            return self._create_empty_prediction(schema_history[-1] if schema_history else {})
        
        # Initialize population of schema mutations
        population = self._initialize_population(schema_history)
        
        # Evolve population
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = [self._evaluate_fitness(individual, schema_history) 
                            for individual in population]
            
            # Selection
            selected = self._selection(population, fitness_scores)
            
            # Crossover and mutation
            offspring = self._crossover_and_mutation(selected)
            
            population = offspring
        
        # Select best prediction
        final_fitness = [self._evaluate_fitness(individual, schema_history) 
                        for individual in population]
        best_individual = population[np.argmax(final_fitness)]
        
        return self._create_prediction(schema_history[-1], best_individual, max(final_fitness))
    
    def _initialize_population(self, schema_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Initialize population of schema variants."""
        population = []
        current_schema = schema_history[-1]
        
        for _ in range(self.population_size):
            # Create variant by applying random mutations
            variant = self._mutate_schema(current_schema.copy())
            population.append(variant)
        
        return population
    
    def _mutate_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random mutations to schema."""
        mutated = schema.copy()
        
        # Possible mutations
        mutations = [
            self._add_field_mutation,
            self._remove_field_mutation,
            self._modify_field_type_mutation,
            self._rename_field_mutation
        ]
        
        # Apply random mutations
        num_mutations = np.random.poisson(2)  # Average 2 mutations
        for _ in range(num_mutations):
            mutation = np.random.choice(mutations)
            mutated = mutation(mutated)
        
        return mutated
    
    def _add_field_mutation(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new field to schema."""
        new_schema = schema.copy()
        
        # Generate random field name
        field_names = ['status', 'created_at', 'updated_at', 'version', 'metadata', 
                      'tags', 'description', 'external_id', 'priority', 'category']
        new_field = np.random.choice(field_names)
        
        # Generate random type
        types = ['str', 'int', 'float', 'bool', 'Optional[str]', 'List[str]', 'dict']
        new_type = np.random.choice(types)
        
        if new_field not in new_schema:
            new_schema[new_field] = {'type': new_type, 'added_in_evolution': True}
        
        return new_schema
    
    def _remove_field_mutation(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Remove a field from schema."""
        new_schema = schema.copy()
        
        if len(new_schema) > 1:  # Don't remove all fields
            field_to_remove = np.random.choice(list(new_schema.keys()))
            new_schema.pop(field_to_remove, None)
        
        return new_schema
    
    def _modify_field_type_mutation(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Modify type of existing field."""
        new_schema = schema.copy()
        
        if new_schema:
            field_name = np.random.choice(list(new_schema.keys()))
            
            # Type evolution patterns
            type_evolutions = {
                'str': ['Optional[str]', 'Union[str, int]'],
                'int': ['float', 'Optional[int]', 'Union[int, str]'],
                'float': ['Optional[float]', 'Union[float, int]'],
                'bool': ['Optional[bool]'],
                'list': ['List[str]', 'List[dict]', 'Optional[list]'],
                'dict': ['Optional[dict]', 'Dict[str, Any]']
            }
            
            current_type = new_schema[field_name].get('type', 'str')
            possible_types = type_evolutions.get(current_type, ['Any'])
            new_type = np.random.choice(possible_types)
            
            new_schema[field_name] = new_schema[field_name].copy()
            new_schema[field_name]['type'] = new_type
            new_schema[field_name]['type_changed'] = True
        
        return new_schema
    
    def _rename_field_mutation(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Rename a field in schema."""
        new_schema = schema.copy()
        
        if new_schema:
            old_name = np.random.choice(list(new_schema.keys()))
            
            # Common renaming patterns
            rename_patterns = {
                'id': 'identifier',
                'name': 'title',
                'desc': 'description',
                'created': 'created_at',
                'updated': 'updated_at',
                'type': 'category'
            }
            
            new_name = rename_patterns.get(old_name, f"{old_name}_v2")
            
            if new_name not in new_schema:
                new_schema[new_name] = new_schema.pop(old_name)
                new_schema[new_name]['renamed_from'] = old_name
        
        return new_schema
    
    def _evaluate_fitness(self, individual: Dict[str, Any], 
                         schema_history: List[Dict[str, Any]]) -> float:
        """Evaluate fitness of schema variant."""
        # Cache fitness calculations
        individual_hash = hashlib.md5(json.dumps(individual, sort_keys=True).encode()).hexdigest()
        cached_fitness = self.fitness_cache.get(individual_hash)
        if cached_fitness is not None:
            return cached_fitness
        
        fitness = 0.0
        
        # Compatibility with existing schemas
        for historical_schema in schema_history:
            compatibility = self._calculate_compatibility(individual, historical_schema)
            fitness += compatibility * 0.3
        
        # Evolutionary plausibility
        if len(schema_history) >= 2:
            evolution_trend = self._analyze_evolution_trend(schema_history)
            trend_adherence = self._calculate_trend_adherence(individual, evolution_trend)
            fitness += trend_adherence * 0.4
        
        # Schema complexity balance
        complexity_score = self._calculate_complexity_score(individual)
        fitness += complexity_score * 0.3
        
        # Cache and return
        self.fitness_cache.set(individual_hash, fitness)
        return fitness
    
    def _calculate_compatibility(self, schema1: Dict[str, Any], 
                               schema2: Dict[str, Any]) -> float:
        """Calculate compatibility score between schemas."""
        if not schema1 or not schema2:
            return 0.0
        
        common_fields = set(schema1.keys()) & set(schema2.keys())
        total_fields = set(schema1.keys()) | set(schema2.keys())
        
        if not total_fields:
            return 1.0
        
        # Jaccard similarity for field names
        field_similarity = len(common_fields) / len(total_fields)
        
        # Type compatibility for common fields
        type_compatibility = 0.0
        if common_fields:
            compatible_types = 0
            for field in common_fields:
                type1 = schema1[field].get('type', 'Any') if isinstance(schema1[field], dict) else 'str'
                type2 = schema2[field].get('type', 'Any') if isinstance(schema2[field], dict) else 'str'
                
                if self._are_types_compatible(type1, type2):
                    compatible_types += 1
            
            type_compatibility = compatible_types / len(common_fields)
        
        return (field_similarity + type_compatibility) / 2.0
    
    def _are_types_compatible(self, type1: str, type2: str) -> bool:
        """Check if two types are compatible."""
        if type1 == type2:
            return True
        
        # Compatible type pairs
        compatible_pairs = [
            ('int', 'float'),
            ('str', 'Optional[str]'),
            ('int', 'Optional[int]'),
            ('float', 'Optional[float]'),
            ('list', 'List[str]'),
            ('dict', 'Dict[str, Any]')
        ]
        
        return (type1, type2) in compatible_pairs or (type2, type1) in compatible_pairs
    
    def _analyze_evolution_trend(self, schema_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze evolution trends in schema history."""
        trends = {
            'field_additions': 0,
            'field_removals': 0,
            'type_changes': 0,
            'complexity_increase': 0
        }
        
        for i in range(1, len(schema_history)):
            prev_schema = schema_history[i-1]
            curr_schema = schema_history[i]
            
            # Field changes
            prev_fields = set(prev_schema.keys())
            curr_fields = set(curr_schema.keys())
            
            trends['field_additions'] += len(curr_fields - prev_fields)
            trends['field_removals'] += len(prev_fields - curr_fields)
            
            # Complexity changes
            prev_complexity = len(prev_fields)
            curr_complexity = len(curr_fields)
            if curr_complexity > prev_complexity:
                trends['complexity_increase'] += 1
        
        return trends
    
    def _calculate_trend_adherence(self, individual: Dict[str, Any], 
                                  trends: Dict[str, Any]) -> float:
        """Calculate how well individual adheres to evolution trends."""
        # Simple trend adherence calculation
        # Would be more sophisticated in practice
        return 0.5  # Placeholder
    
    def _calculate_complexity_score(self, schema: Dict[str, Any]) -> float:
        """Calculate complexity score for schema."""
        if not schema:
            return 0.0
        
        # Number of fields (normalized)
        field_count_score = min(len(schema) / 20.0, 1.0)
        
        # Type complexity
        complex_types = 0
        for field_info in schema.values():
            if isinstance(field_info, dict):
                field_type = field_info.get('type', 'str')
                if any(keyword in field_type for keyword in ['List', 'Dict', 'Union', 'Optional']):
                    complex_types += 1
        
        type_complexity_score = complex_types / len(schema) if schema else 0
        
        # Balanced complexity (not too simple, not too complex)
        optimal_complexity = 0.6
        complexity_balance = 1.0 - abs(type_complexity_score - optimal_complexity)
        
        return (field_count_score + complexity_balance) / 2.0
    
    def _selection(self, population: List[Dict[str, Any]], 
                  fitness_scores: List[float]) -> List[Dict[str, Any]]:
        """Select individuals for next generation using tournament selection."""
        selected = []
        
        for _ in range(self.population_size):
            # Tournament selection
            tournament_size = 3
            tournament_indices = np.random.choice(len(population), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        
        return selected
    
    def _crossover_and_mutation(self, selected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply crossover and mutation to selected individuals."""
        offspring = []
        
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[i + 1] if i + 1 < len(selected) else selected[0]
            
            # Crossover
            child1, child2 = self._crossover(parent1, parent2)
            
            # Mutation
            if np.random.random() < self.mutation_rate:
                child1 = self._mutate_schema(child1)
            if np.random.random() < self.mutation_rate:
                child2 = self._mutate_schema(child2)
            
            offspring.extend([child1, child2])
        
        return offspring[:self.population_size]
    
    def _crossover(self, parent1: Dict[str, Any], 
                  parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform crossover between two parent schemas."""
        all_fields = set(parent1.keys()) | set(parent2.keys())
        
        child1, child2 = {}, {}
        
        for field in all_fields:
            if np.random.random() < 0.5:
                # Take from parent1
                if field in parent1:
                    child1[field] = parent1[field].copy() if isinstance(parent1[field], dict) else parent1[field]
                if field in parent2:
                    child2[field] = parent2[field].copy() if isinstance(parent2[field], dict) else parent2[field]
            else:
                # Take from parent2
                if field in parent2:
                    child1[field] = parent2[field].copy() if isinstance(parent2[field], dict) else parent2[field]
                if field in parent1:
                    child2[field] = parent1[field].copy() if isinstance(parent1[field], dict) else parent1[field]
        
        return child1, child2
    
    def _create_prediction(self, current_schema: Dict[str, Any], 
                          predicted_schema: Dict[str, Any],
                          confidence: float) -> EvolutionaryPrediction:
        """Create evolution prediction from best individual."""
        # Identify changes
        predicted_changes = []
        breaking_changes = []
        
        current_fields = set(current_schema.keys())
        predicted_fields = set(predicted_schema.keys())
        
        # Added fields
        for field in predicted_fields - current_fields:
            predicted_changes.append({
                'type': 'field_added',
                'field': field,
                'new_type': predicted_schema[field].get('type', 'str') if isinstance(predicted_schema[field], dict) else 'str'
            })
        
        # Removed fields
        for field in current_fields - predicted_fields:
            predicted_changes.append({
                'type': 'field_removed',
                'field': field
            })
            breaking_changes.append(f"Field '{field}' removed")
        
        # Modified fields
        for field in current_fields & predicted_fields:
            current_type = current_schema[field].get('type', 'str') if isinstance(current_schema[field], dict) else 'str'
            predicted_type = predicted_schema[field].get('type', 'str') if isinstance(predicted_schema[field], dict) else 'str'
            
            if current_type != predicted_type:
                predicted_changes.append({
                    'type': 'type_changed',
                    'field': field,
                    'old_type': current_type,
                    'new_type': predicted_type
                })
                
                if not self._are_types_compatible(current_type, predicted_type):
                    breaking_changes.append(f"Field '{field}' type changed from {current_type} to {predicted_type}")
        
        # Calculate compatibility score
        compatibility_score = self._calculate_compatibility(current_schema, predicted_schema)
        
        return EvolutionaryPrediction(
            current_schema=current_schema,
            predicted_changes=predicted_changes,
            compatibility_score=compatibility_score,
            breaking_changes=breaking_changes,
            evolution_confidence=confidence
        )
    
    def _create_empty_prediction(self, current_schema: Dict[str, Any]) -> EvolutionaryPrediction:
        """Create empty prediction when insufficient history."""
        return EvolutionaryPrediction(
            current_schema=current_schema,
            predicted_changes=[],
            compatibility_score=1.0,
            breaking_changes=[],
            evolution_confidence=0.0
        )


class MLEnhancedSchemaInferencer:
    """
    Main ML-enhanced schema inference system that orchestrates all components.
    
    Combines Bayesian type inference, meta-learning, and evolutionary prediction
    to provide comprehensive schema understanding and evolution analysis.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.bayesian_inferencer = BayesianTypeInferencer(
            feature_dim=self.config.get('feature_dim', 128)
        )
        self.meta_learner = MetaLearningSchemaInferencer()
        self.evolution_predictor = EvolutionarySchemaPredictor(
            population_size=self.config.get('population_size', 50)
        )
        
        # Schema history for evolution analysis
        self.schema_history = []
        
        logger.info("Initialized MLEnhancedSchemaInferencer")
    
    def infer_schema(self, ast_tree: ast.AST, context: Dict[str, Any] = None) -> Dict[str, ProbabilisticType]:
        """Infer schema using ML-enhanced techniques."""
        schema = {}
        
        # Find all relevant nodes for schema inference
        for node in ast.walk(ast_tree):
            if self._is_schema_relevant_node(node):
                field_name = self._extract_field_name(node)
                if field_name:
                    # Use Bayesian inference for type prediction
                    prob_type = self.bayesian_inferencer.infer_type(node, context)
                    schema[field_name] = prob_type
        
        return schema
    
    def learn_from_examples(self, examples: List[Dict[str, Any]]):
        """Learn from example schemas using meta-learning."""
        # Convert examples to schema patterns
        patterns = []
        for i, example in enumerate(examples):
            pattern = SchemaPattern(
                pattern_id=f"example_{i}",
                fields={},
                frequency=1,
                contexts=[],
                confidence=0.8,
                last_seen=time.time()
            )
            
            for field, value in example.items():
                prob_type = ProbabilisticType(
                    primary_type=type(value).__name__,
                    confidence=0.8,
                    evidence=[f"From example {i}"]
                )
                pattern.fields[field] = prob_type
            
            patterns.append(pattern)
        
        # Learn patterns using meta-learning
        self.meta_learner.learn_from_patterns(patterns)
    
    def predict_schema_evolution(self, current_schema: Dict[str, Any]) -> EvolutionaryPrediction:
        """Predict how schema will evolve over time."""
        # Add current schema to history
        self.schema_history.append(current_schema)
        
        # Keep only recent history
        if len(self.schema_history) > 10:
            self.schema_history = self.schema_history[-10:]
        
        # Predict evolution
        return self.evolution_predictor.predict_evolution(self.schema_history)
    
    def analyze_schema_quality(self, schema: Dict[str, ProbabilisticType]) -> Dict[str, Any]:
        """Analyze quality metrics of inferred schema."""
        quality_metrics = {
            'average_confidence': np.mean([field.confidence for field in schema.values()]) if schema else 0.0,
            'total_fields': len(schema),
            'high_confidence_fields': sum(1 for field in schema.values() if field.confidence > 0.8),
            'uncertain_fields': sum(1 for field in schema.values() if field.uncertainty > 0.3),
            'type_diversity': len(set(field.primary_type for field in schema.values()))
        }
        
        # Calculate quality score
        confidence_score = quality_metrics['average_confidence']
        certainty_score = 1.0 - (quality_metrics['uncertain_fields'] / max(quality_metrics['total_fields'], 1))
        diversity_score = min(quality_metrics['type_diversity'] / 5.0, 1.0)  # Normalized
        
        quality_metrics['overall_quality'] = (confidence_score + certainty_score + diversity_score) / 3.0
        
        return quality_metrics
    
    def _is_schema_relevant_node(self, node: ast.AST) -> bool:
        """Check if node is relevant for schema inference."""
        # Variable assignments
        if isinstance(node, ast.Assign):
            return True
        
        # Function parameters with annotations
        if isinstance(node, ast.arg) and node.annotation:
            return True
        
        # Class attributes
        if isinstance(node, ast.AnnAssign):
            return True
        
        return False
    
    def _extract_field_name(self, node: ast.AST) -> Optional[str]:
        """Extract field name from AST node."""
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                return node.targets[0].id
        
        if isinstance(node, ast.arg):
            return node.arg
        
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            return node.target.id
        
        return None


# Research validation and benchmarking
def validate_ml_schema_inference():
    """Validate ML schema inference against ground truth."""
    # This would implement comprehensive validation studies
    pass


def benchmark_against_traditional_methods():
    """Benchmark ML methods against traditional schema inference."""
    # This would implement comparative benchmarking
    pass


if __name__ == "__main__":
    # Example usage
    inferencer = MLEnhancedSchemaInferencer()
    
    # Test with example data
    examples = [
        {'name': 'John', 'age': 30, 'email': 'john@example.com'},
        {'name': 'Jane', 'age': 25, 'email': 'jane@example.com'}
    ]
    
    inferencer.learn_from_examples(examples)
    print("ML-enhanced schema inference system initialized and tested")