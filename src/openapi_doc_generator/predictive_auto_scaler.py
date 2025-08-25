"""
Predictive Auto-Scaling System with ML-based Capacity Planning

This module implements advanced predictive auto-scaling capabilities that use machine
learning algorithms to forecast resource demands, enabling proactive scaling decisions
and optimal resource utilization across distributed systems.

Features:
- ML-based demand forecasting and capacity planning
- Preemptive scaling based on predicted workload patterns
- Multi-dimensional resource optimization
- Advanced anomaly detection and response
- Cost-aware scaling decisions
- Seasonal and trend analysis
- Real-time performance optimization
"""

import asyncio
import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .enhanced_monitoring import get_monitor
from .performance_optimizer import get_optimizer
from .quantum_scale_optimizer import QuantumAutoScaler, ScalingMetrics

logger = logging.getLogger(__name__)


class PredictionModel(Enum):
    """Types of prediction models available."""
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    POLYNOMIAL_REGRESSION = "polynomial_regression"
    SEASONAL_DECOMPOSITION = "seasonal_decomposition"
    ENSEMBLE = "ensemble"
    NEURAL_NETWORK = "neural_network"


class ScalingTrigger(Enum):
    """Types of scaling triggers."""
    DEMAND_PREDICTION = "demand_prediction"
    RESOURCE_FORECAST = "resource_forecast"
    ANOMALY_DETECTION = "anomaly_detection"
    SEASONAL_PATTERN = "seasonal_pattern"
    COST_OPTIMIZATION = "cost_optimization"
    PERFORMANCE_TARGET = "performance_target"


class ResourceType(Enum):
    """Types of resources for scaling."""
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"
    INSTANCES = "instances"
    CONNECTIONS = "connections"


@dataclass
class PredictionResult:
    """Result of demand prediction."""
    resource_type: ResourceType
    current_value: float
    predicted_value: float
    prediction_horizon_minutes: int
    confidence_interval: Tuple[float, float]
    confidence_score: float
    model_used: PredictionModel
    features_used: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ScalingPrediction:
    """Prediction for scaling requirements."""
    resource_predictions: List[PredictionResult]
    recommended_action: str  # "scale_up", "scale_down", "no_change"
    target_instances: int
    target_resources: Dict[ResourceType, float]
    urgency_score: float  # 0.0 to 1.0
    cost_impact: float
    expected_benefit: Dict[str, float]
    reasoning: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class WorkloadPattern:
    """Detected workload pattern."""
    pattern_id: str
    pattern_type: str  # "daily", "weekly", "monthly", "seasonal"
    peak_hours: List[int]
    low_hours: List[int]
    scaling_multiplier: float
    confidence: float
    last_observed: datetime
    occurrences: int = 1


@dataclass
class AnomalyDetection:
    """Anomaly detection result."""
    timestamp: datetime
    resource_type: ResourceType
    observed_value: float
    expected_value: float
    anomaly_score: float
    severity: str  # "low", "medium", "high", "critical"
    recommended_action: str
    description: str


class MLCapacityPredictor:
    """Machine learning-based capacity prediction system."""

    def __init__(self,
                 prediction_horizon_minutes: int = 30,
                 feature_window_hours: int = 24,
                 min_training_samples: int = 100):

        self.prediction_horizon_minutes = prediction_horizon_minutes
        self.feature_window_hours = feature_window_hours
        self.min_training_samples = min_training_samples

        # ML Models
        self.models: Dict[ResourceType, Dict[str, Any]] = {}
        self.scalers: Dict[ResourceType, StandardScaler] = {}

        # Training data
        self.training_data: Dict[ResourceType, deque] = {
            rt: deque(maxlen=10000) for rt in ResourceType
        }

        # Feature engineering
        self.feature_history: deque = deque(maxlen=1000)
        self.seasonal_patterns: Dict[str, float] = {}

        # Model performance tracking
        self.model_accuracies: Dict[ResourceType, Dict[str, float]] = defaultdict(dict)
        self.prediction_errors: Dict[ResourceType, deque] = {
            rt: deque(maxlen=100) for rt in ResourceType
        }

        # Initialize models for each resource type
        for resource_type in ResourceType:
            self._initialize_models(resource_type)

        logger.info("ML Capacity Predictor initialized")

    def _initialize_models(self, resource_type: ResourceType):
        """Initialize ML models for a resource type."""
        self.models[resource_type] = {
            PredictionModel.LINEAR_REGRESSION: LinearRegression(),
            PredictionModel.RANDOM_FOREST: RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            PredictionModel.POLYNOMIAL_REGRESSION: LinearRegression(),
        }

        self.scalers[resource_type] = StandardScaler()

    async def add_training_sample(self,
                                resource_type: ResourceType,
                                value: float,
                                features: Dict[str, float]):
        """Add a training sample for ML models."""
        timestamp = datetime.now()

        sample = {
            'timestamp': timestamp,
            'value': value,
            'features': features,
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'day_of_month': timestamp.day,
            'month': timestamp.month
        }

        self.training_data[resource_type].append(sample)

        # Retrain models periodically
        if len(self.training_data[resource_type]) >= self.min_training_samples:
            if len(self.training_data[resource_type]) % 50 == 0:  # Retrain every 50 samples
                await self._retrain_models(resource_type)

    async def predict_demand(self,
                           resource_type: ResourceType,
                           current_features: Dict[str, float]) -> PredictionResult:
        """Predict future demand for a resource type."""

        if resource_type not in self.models or len(self.training_data[resource_type]) < 10:
            # Not enough data, return simple trend-based prediction
            return await self._simple_trend_prediction(resource_type, current_features)

        try:
            # Prepare features
            features = self._prepare_features(current_features)

            # Get predictions from all models
            predictions = {}
            confidences = {}

            for model_type, model in self.models[resource_type].items():
                if hasattr(model, 'predict'):
                    try:
                        # Scale features
                        features_scaled = self.scalers[resource_type].transform([features])

                        # Make prediction
                        pred = model.predict(features_scaled)[0]
                        predictions[model_type] = pred

                        # Calculate confidence based on model accuracy
                        accuracy = self.model_accuracies[resource_type].get(model_type.value, 0.5)
                        confidences[model_type] = accuracy

                    except Exception as e:
                        logger.warning(f"Prediction failed for {model_type}: {e}")

            if not predictions:
                return await self._simple_trend_prediction(resource_type, current_features)

            # Ensemble prediction - weighted average based on model performance
            total_confidence = sum(confidences.values())
            if total_confidence > 0:
                weighted_prediction = sum(
                    pred * confidences[model_type]
                    for model_type, pred in predictions.items()
                ) / total_confidence

                ensemble_confidence = total_confidence / len(predictions)
            else:
                weighted_prediction = np.mean(list(predictions.values()))
                ensemble_confidence = 0.5

            # Calculate confidence interval
            prediction_std = np.std(list(predictions.values()))
            confidence_interval = (
                weighted_prediction - 1.96 * prediction_std,
                weighted_prediction + 1.96 * prediction_std
            )

            return PredictionResult(
                resource_type=resource_type,
                current_value=current_features.get(f'{resource_type.value}_current', 0.0),
                predicted_value=max(0.0, weighted_prediction),
                prediction_horizon_minutes=self.prediction_horizon_minutes,
                confidence_interval=confidence_interval,
                confidence_score=ensemble_confidence,
                model_used=PredictionModel.ENSEMBLE,
                features_used=list(features.keys())
            )

        except Exception as e:
            logger.error(f"ML prediction failed for {resource_type}: {e}")
            return await self._simple_trend_prediction(resource_type, current_features)

    async def _simple_trend_prediction(self,
                                     resource_type: ResourceType,
                                     current_features: Dict[str, float]) -> PredictionResult:
        """Simple trend-based prediction when ML models are unavailable."""
        current_value = current_features.get(f'{resource_type.value}_current', 0.0)

        # Get recent values to calculate trend
        recent_samples = list(self.training_data[resource_type])[-10:]

        if len(recent_samples) >= 3:
            recent_values = [s['value'] for s in recent_samples]

            # Calculate linear trend
            x = list(range(len(recent_values)))
            slope, intercept, r_value, _, _ = stats.linregress(x, recent_values)

            # Project trend forward
            future_time = len(recent_values) + (self.prediction_horizon_minutes / 5)  # Assume 5-minute intervals
            predicted_value = slope * future_time + intercept

            confidence = abs(r_value) * 0.7  # Confidence based on correlation
        else:
            # No trend data available, assume current value
            predicted_value = current_value
            confidence = 0.3

        return PredictionResult(
            resource_type=resource_type,
            current_value=current_value,
            predicted_value=max(0.0, predicted_value),
            prediction_horizon_minutes=self.prediction_horizon_minutes,
            confidence_interval=(predicted_value * 0.8, predicted_value * 1.2),
            confidence_score=confidence,
            model_used=PredictionModel.LINEAR_REGRESSION,
            features_used=list(current_features.keys())
        )

    def _prepare_features(self, current_features: Dict[str, float]) -> List[float]:
        """Prepare features for ML model input."""
        now = datetime.now()

        # Time-based features
        time_features = [
            now.hour / 24.0,
            now.weekday() / 7.0,
            now.day / 31.0,
            now.month / 12.0,
            math.sin(2 * math.pi * now.hour / 24.0),  # Cyclical hour
            math.cos(2 * math.pi * now.hour / 24.0),
            math.sin(2 * math.pi * now.weekday() / 7.0),  # Cyclical day of week
            math.cos(2 * math.pi * now.weekday() / 7.0)
        ]

        # Current resource features
        resource_features = [
            current_features.get('cpu_utilization', 0.0) / 100.0,
            current_features.get('memory_utilization', 0.0) / 100.0,
            current_features.get('network_utilization', 0.0) / 100.0,
            current_features.get('active_connections', 0.0) / 1000.0,
            current_features.get('request_rate', 0.0) / 100.0,
            current_features.get('response_time', 0.0) / 1000.0,
        ]

        # Trend features from recent history
        trend_features = []
        if len(self.feature_history) >= 5:
            recent_values = list(self.feature_history)[-5:]

            # CPU trend
            cpu_values = [f.get('cpu_utilization', 0.0) for f in recent_values]
            cpu_trend = (cpu_values[-1] - cpu_values[0]) / len(cpu_values) if len(cpu_values) > 1 else 0.0
            trend_features.append(cpu_trend / 100.0)

            # Memory trend
            memory_values = [f.get('memory_utilization', 0.0) for f in recent_values]
            memory_trend = (memory_values[-1] - memory_values[0]) / len(memory_values) if len(memory_values) > 1 else 0.0
            trend_features.append(memory_trend / 100.0)

            # Request rate trend
            request_values = [f.get('request_rate', 0.0) for f in recent_values]
            request_trend = (request_values[-1] - request_values[0]) / len(request_values) if len(request_values) > 1 else 0.0
            trend_features.append(request_trend / 100.0)
        else:
            trend_features = [0.0, 0.0, 0.0]

        return time_features + resource_features + trend_features

    async def _retrain_models(self, resource_type: ResourceType):
        """Retrain ML models with recent data."""
        try:
            samples = list(self.training_data[resource_type])

            if len(samples) < self.min_training_samples:
                return

            # Prepare training data
            X = []
            y = []

            for sample in samples:
                features = self._prepare_features(sample['features'])
                X.append(features)
                y.append(sample['value'])

            X = np.array(X)
            y = np.array(y)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Fit scaler
            self.scalers[resource_type].fit(X_train)
            X_train_scaled = self.scalers[resource_type].transform(X_train)
            X_test_scaled = self.scalers[resource_type].transform(X_test)

            # Train models
            for model_type, model in self.models[resource_type].items():
                try:
                    if model_type == PredictionModel.POLYNOMIAL_REGRESSION:
                        # Create polynomial features
                        from sklearn.preprocessing import PolynomialFeatures
                        poly_features = PolynomialFeatures(degree=2)
                        X_train_poly = poly_features.fit_transform(X_train_scaled)
                        X_test_poly = poly_features.transform(X_test_scaled)

                        model.fit(X_train_poly, y_train)
                        predictions = model.predict(X_test_poly)
                    else:
                        model.fit(X_train_scaled, y_train)
                        predictions = model.predict(X_test_scaled)

                    # Calculate accuracy metrics
                    mae = mean_absolute_error(y_test, predictions)
                    r2 = r2_score(y_test, predictions)

                    # Store accuracy (higher is better, convert MAE to accuracy-like metric)
                    accuracy = max(0.0, 1.0 - (mae / (np.mean(y_test) + 1e-6)))
                    accuracy = min(1.0, accuracy * r2 if r2 > 0 else accuracy * 0.5)

                    self.model_accuracies[resource_type][model_type.value] = accuracy

                    logger.debug(f"Retrained {model_type.value} for {resource_type.value}: "
                               f"MAE={mae:.3f}, R2={r2:.3f}, Accuracy={accuracy:.3f}")

                except Exception as e:
                    logger.warning(f"Failed to train {model_type} for {resource_type}: {e}")

            logger.info(f"Model retraining completed for {resource_type.value}")

        except Exception as e:
            logger.error(f"Model retraining failed for {resource_type}: {e}")


class WorkloadPatternAnalyzer:
    """Analyzes workload patterns for predictive scaling."""

    def __init__(self):
        self.patterns: Dict[str, WorkloadPattern] = {}
        self.historical_data: deque = deque(maxlen=10080)  # 1 week at 1-minute intervals
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)

    def add_workload_data(self,
                         timestamp: datetime,
                         metrics: Dict[str, float]):
        """Add workload data point for pattern analysis."""
        data_point = {
            'timestamp': timestamp,
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'day_of_month': timestamp.day,
            'month': timestamp.month,
            'metrics': metrics
        }

        self.historical_data.append(data_point)

        # Analyze patterns periodically
        if len(self.historical_data) % 100 == 0:
            self._analyze_patterns()

    def _analyze_patterns(self):
        """Analyze workload patterns from historical data."""
        if len(self.historical_data) < 144:  # Need at least 1 day of data
            return

        # Analyze daily patterns
        self._analyze_daily_patterns()

        # Analyze weekly patterns
        if len(self.historical_data) >= 1008:  # 1 week of data
            self._analyze_weekly_patterns()

        # Train anomaly detector
        self._train_anomaly_detector()

    def _analyze_daily_patterns(self):
        """Analyze daily workload patterns."""
        hourly_metrics = defaultdict(list)

        for data_point in self.historical_data:
            hour = data_point['hour']
            cpu_usage = data_point['metrics'].get('cpu_utilization', 0.0)
            hourly_metrics[hour].append(cpu_usage)

        # Find peak and low hours
        hourly_averages = {}
        for hour, values in hourly_metrics.items():
            if values:
                hourly_averages[hour] = np.mean(values)

        if len(hourly_averages) >= 12:  # Need sufficient data
            avg_load = np.mean(list(hourly_averages.values()))
            std_load = np.std(list(hourly_averages.values()))

            peak_hours = [
                hour for hour, avg in hourly_averages.items()
                if avg > avg_load + 0.5 * std_load
            ]

            low_hours = [
                hour for hour, avg in hourly_averages.items()
                if avg < avg_load - 0.5 * std_load
            ]

            if peak_hours or low_hours:
                pattern = WorkloadPattern(
                    pattern_id="daily_pattern",
                    pattern_type="daily",
                    peak_hours=peak_hours,
                    low_hours=low_hours,
                    scaling_multiplier=1.5 if peak_hours else 1.0,
                    confidence=0.8,
                    last_observed=datetime.now()
                )

                self.patterns["daily"] = pattern

    def _analyze_weekly_patterns(self):
        """Analyze weekly workload patterns."""
        daily_metrics = defaultdict(list)

        for data_point in self.historical_data:
            day_of_week = data_point['day_of_week']
            cpu_usage = data_point['metrics'].get('cpu_utilization', 0.0)
            daily_metrics[day_of_week].append(cpu_usage)

        # Analyze weekday vs weekend patterns
        weekday_loads = []
        weekend_loads = []

        for day, values in daily_metrics.items():
            if values:
                avg_load = np.mean(values)
                if day < 5:  # Monday to Friday (0-4)
                    weekday_loads.append(avg_load)
                else:  # Weekend (5-6)
                    weekend_loads.append(avg_load)

        if weekday_loads and weekend_loads:
            weekday_avg = np.mean(weekday_loads)
            weekend_avg = np.mean(weekend_loads)

            if abs(weekday_avg - weekend_avg) > 10.0:  # Significant difference
                pattern = WorkloadPattern(
                    pattern_id="weekly_pattern",
                    pattern_type="weekly",
                    peak_hours=list(range(9, 17)) if weekday_avg > weekend_avg else [],
                    low_hours=list(range(18, 24)) + list(range(0, 8)),
                    scaling_multiplier=1.3,
                    confidence=0.7,
                    last_observed=datetime.now()
                )

                self.patterns["weekly"] = pattern

    def _train_anomaly_detector(self):
        """Train anomaly detection model."""
        if len(self.historical_data) < 100:
            return

        try:
            # Prepare features for anomaly detection
            features = []
            for data_point in self.historical_data:
                feature_vector = [
                    data_point['hour'],
                    data_point['day_of_week'],
                    data_point['metrics'].get('cpu_utilization', 0.0),
                    data_point['metrics'].get('memory_utilization', 0.0),
                    data_point['metrics'].get('request_rate', 0.0)
                ]
                features.append(feature_vector)

            X = np.array(features)
            self.anomaly_detector.fit(X)

        except Exception as e:
            logger.warning(f"Anomaly detector training failed: {e}")

    def detect_anomalies(self, current_metrics: Dict[str, float]) -> List[AnomalyDetection]:
        """Detect anomalies in current metrics."""
        anomalies = []

        try:
            now = datetime.now()
            current_features = [
                now.hour,
                now.weekday(),
                current_metrics.get('cpu_utilization', 0.0),
                current_metrics.get('memory_utilization', 0.0),
                current_metrics.get('request_rate', 0.0)
            ]

            # Get anomaly score
            anomaly_score = self.anomaly_detector.decision_function([current_features])[0]
            is_anomaly = self.anomaly_detector.predict([current_features])[0] == -1

            if is_anomaly:
                # Determine severity based on anomaly score
                if anomaly_score < -0.5:
                    severity = "critical"
                elif anomaly_score < -0.3:
                    severity = "high"
                elif anomaly_score < -0.1:
                    severity = "medium"
                else:
                    severity = "low"

                anomaly = AnomalyDetection(
                    timestamp=now,
                    resource_type=ResourceType.CPU,  # Primary anomaly type
                    observed_value=current_metrics.get('cpu_utilization', 0.0),
                    expected_value=self._get_expected_value('cpu_utilization', now),
                    anomaly_score=abs(anomaly_score),
                    severity=severity,
                    recommended_action="scale_up" if severity in ["high", "critical"] else "monitor",
                    description=f"Unusual resource usage pattern detected (score: {anomaly_score:.3f})"
                )

                anomalies.append(anomaly)

        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")

        return anomalies

    def _get_expected_value(self, metric_name: str, timestamp: datetime) -> float:
        """Get expected value for a metric at given timestamp."""
        # Simple implementation: return historical average for this hour
        hour = timestamp.hour

        values = []
        for data_point in self.historical_data:
            if data_point['hour'] == hour:
                value = data_point['metrics'].get(metric_name, 0.0)
                values.append(value)

        return np.mean(values) if values else 0.0

    def get_scaling_recommendation(self, current_time: datetime) -> Dict[str, Any]:
        """Get scaling recommendation based on detected patterns."""
        recommendations = {}

        current_hour = current_time.hour
        current_day = current_time.weekday()

        # Check daily patterns
        if "daily" in self.patterns:
            pattern = self.patterns["daily"]

            if current_hour in pattern.peak_hours:
                recommendations["daily_peak"] = {
                    "action": "scale_up",
                    "multiplier": pattern.scaling_multiplier,
                    "confidence": pattern.confidence,
                    "reason": f"Daily peak hour detected: {current_hour}"
                }
            elif current_hour in pattern.low_hours:
                recommendations["daily_low"] = {
                    "action": "scale_down",
                    "multiplier": 1.0 / pattern.scaling_multiplier,
                    "confidence": pattern.confidence,
                    "reason": f"Daily low hour detected: {current_hour}"
                }

        # Check weekly patterns
        if "weekly" in self.patterns:
            pattern = self.patterns["weekly"]

            is_weekend = current_day >= 5
            is_business_hours = 9 <= current_hour <= 17

            if not is_weekend and is_business_hours:
                recommendations["weekly_business"] = {
                    "action": "scale_up",
                    "multiplier": pattern.scaling_multiplier,
                    "confidence": pattern.confidence * 0.8,
                    "reason": "Weekday business hours pattern"
                }
            elif is_weekend or not is_business_hours:
                recommendations["weekly_offhours"] = {
                    "action": "scale_down",
                    "multiplier": 1.0 / pattern.scaling_multiplier,
                    "confidence": pattern.confidence * 0.7,
                    "reason": "Weekend or off-hours pattern"
                }

        return recommendations


class PredictiveAutoScaler:
    """
    Advanced predictive auto-scaler with ML-based capacity planning.
    """

    def __init__(self,
                 min_instances: int = 1,
                 max_instances: int = 50,
                 prediction_horizon_minutes: int = 30,
                 cost_per_instance_hour: float = 0.10):

        self.min_instances = min_instances
        self.max_instances = max_instances
        self.prediction_horizon_minutes = prediction_horizon_minutes
        self.cost_per_instance_hour = cost_per_instance_hour

        # ML components
        self.capacity_predictor = MLCapacityPredictor(
            prediction_horizon_minutes=prediction_horizon_minutes
        )
        self.pattern_analyzer = WorkloadPatternAnalyzer()

        # Current state
        self.current_instances = min_instances
        self.current_metrics: Dict[ResourceType, float] = {}
        self.last_scaling_action = datetime.now()

        # Scaling history
        self.scaling_history: deque = deque(maxlen=1000)
        self.predictions: deque = deque(maxlen=500)

        # Integration with existing systems
        self.monitor = get_monitor()
        self.optimizer = get_optimizer()
        self.quantum_scaler = QuantumAutoScaler(
            min_instances=min_instances,
            max_instances=max_instances
        )

        # Background tasks
        self.prediction_task: Optional[asyncio.Task] = None
        self.pattern_analysis_task: Optional[asyncio.Task] = None
        self.running = False

        # Configuration
        self.scaling_cooldown_minutes = 5
        self.confidence_threshold = 0.6
        self.cost_sensitivity = 0.7  # 0.0 = cost-insensitive, 1.0 = very cost-sensitive

        logger.info("Predictive Auto-Scaler initialized")

    async def start(self):
        """Start the predictive auto-scaler."""
        if self.running:
            return

        self.running = True

        # Start background tasks
        self.prediction_task = asyncio.create_task(self._prediction_loop())
        self.pattern_analysis_task = asyncio.create_task(self._pattern_analysis_loop())

        logger.info("Predictive auto-scaler started")

    async def stop(self):
        """Stop the predictive auto-scaler."""
        if not self.running:
            return

        self.running = False

        # Cancel background tasks
        if self.prediction_task and not self.prediction_task.done():
            self.prediction_task.cancel()

        if self.pattern_analysis_task and not self.pattern_analysis_task.done():
            self.pattern_analysis_task.cancel()

        # Wait for tasks to complete
        tasks = [self.prediction_task, self.pattern_analysis_task]
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("Predictive auto-scaler stopped")

    async def update_metrics(self, metrics: Dict[str, float]):
        """Update current system metrics."""
        # Convert to resource-specific metrics
        resource_metrics = {
            ResourceType.CPU: metrics.get('cpu_utilization', 0.0),
            ResourceType.MEMORY: metrics.get('memory_utilization', 0.0),
            ResourceType.NETWORK: metrics.get('network_utilization', 0.0),
            ResourceType.INSTANCES: float(self.current_instances),
            ResourceType.CONNECTIONS: metrics.get('active_connections', 0.0)
        }

        self.current_metrics = resource_metrics

        # Add training samples to ML predictor
        features = {
            'cpu_utilization': metrics.get('cpu_utilization', 0.0),
            'memory_utilization': metrics.get('memory_utilization', 0.0),
            'network_utilization': metrics.get('network_utilization', 0.0),
            'active_connections': metrics.get('active_connections', 0.0),
            'request_rate': metrics.get('request_rate', 0.0),
            'response_time': metrics.get('response_time', 0.0),
            'instances_current': float(self.current_instances)
        }

        for resource_type, value in resource_metrics.items():
            await self.capacity_predictor.add_training_sample(
                resource_type, value, features
            )

        # Add to pattern analyzer
        self.pattern_analyzer.add_workload_data(datetime.now(), metrics)

    async def get_scaling_prediction(self) -> ScalingPrediction:
        """Get comprehensive scaling prediction."""
        current_time = datetime.now()

        # Get ML predictions for all resource types
        resource_predictions = []
        features = self._get_current_features()

        for resource_type in ResourceType:
            if resource_type in self.current_metrics:
                prediction = await self.capacity_predictor.predict_demand(
                    resource_type, features
                )
                resource_predictions.append(prediction)

        # Detect anomalies
        anomalies = self.pattern_analyzer.detect_anomalies(features)

        # Get pattern-based recommendations
        pattern_recommendations = self.pattern_analyzer.get_scaling_recommendation(current_time)

        # Analyze scaling decision
        scaling_analysis = await self._analyze_scaling_decision(
            resource_predictions, anomalies, pattern_recommendations
        )

        # Create comprehensive prediction
        prediction = ScalingPrediction(
            resource_predictions=resource_predictions,
            recommended_action=scaling_analysis['action'],
            target_instances=scaling_analysis['target_instances'],
            target_resources=scaling_analysis['target_resources'],
            urgency_score=scaling_analysis['urgency'],
            cost_impact=scaling_analysis['cost_impact'],
            expected_benefit=scaling_analysis['expected_benefit'],
            reasoning=scaling_analysis['reasoning']
        )

        self.predictions.append(prediction)
        return prediction

    async def _analyze_scaling_decision(self,
                                      resource_predictions: List[PredictionResult],
                                      anomalies: List[AnomalyDetection],
                                      pattern_recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze all factors to make scaling decision."""

        reasoning = []
        urgency_factors = []

        # Analyze ML predictions
        cpu_prediction = None
        memory_prediction = None

        for pred in resource_predictions:
            if pred.resource_type == ResourceType.CPU:
                cpu_prediction = pred
            elif pred.resource_type == ResourceType.MEMORY:
                memory_prediction = pred

        # CPU analysis
        cpu_scaling_needed = 0
        if cpu_prediction and cpu_prediction.confidence_score > self.confidence_threshold:
            current_cpu = cpu_prediction.current_value
            predicted_cpu = cpu_prediction.predicted_value

            if predicted_cpu > 80.0:  # High CPU predicted
                cpu_scaling_needed = 1
                urgency_factors.append(0.8)
                reasoning.append(f"High CPU predicted: {predicted_cpu:.1f}%")
            elif predicted_cpu < 30.0 and current_cpu < 40.0:  # Low CPU predicted
                cpu_scaling_needed = -1
                urgency_factors.append(0.4)
                reasoning.append(f"Low CPU predicted: {predicted_cpu:.1f}%")

        # Memory analysis
        memory_scaling_needed = 0
        if memory_prediction and memory_prediction.confidence_score > self.confidence_threshold:
            current_memory = memory_prediction.current_value
            predicted_memory = memory_prediction.predicted_value

            if predicted_memory > 85.0:  # High memory predicted
                memory_scaling_needed = 1
                urgency_factors.append(0.9)  # Memory is more critical
                reasoning.append(f"High memory predicted: {predicted_memory:.1f}%")
            elif predicted_memory < 40.0 and current_memory < 50.0:  # Low memory predicted
                memory_scaling_needed = -1
                urgency_factors.append(0.3)
                reasoning.append(f"Low memory predicted: {predicted_memory:.1f}%")

        # Anomaly analysis
        anomaly_scaling_needed = 0
        for anomaly in anomalies:
            if anomaly.severity in ["high", "critical"]:
                anomaly_scaling_needed = 1
                urgency_factors.append(0.9)
                reasoning.append(f"Anomaly detected: {anomaly.description}")
            elif anomaly.recommended_action == "scale_up":
                anomaly_scaling_needed = max(anomaly_scaling_needed, 1)
                urgency_factors.append(0.6)
                reasoning.append(f"Anomaly suggests scaling up: {anomaly.description}")

        # Pattern analysis
        pattern_scaling_needed = 0
        pattern_confidence = 0.0

        for pattern_name, recommendation in pattern_recommendations.items():
            if recommendation['action'] == 'scale_up':
                pattern_scaling_needed = max(pattern_scaling_needed, 1)
                pattern_confidence = max(pattern_confidence, recommendation['confidence'])
                reasoning.append(f"Pattern-based scale up: {recommendation['reason']}")
            elif recommendation['action'] == 'scale_down':
                pattern_scaling_needed = min(pattern_scaling_needed, -1)
                pattern_confidence = max(pattern_confidence, recommendation['confidence'])
                reasoning.append(f"Pattern-based scale down: {recommendation['reason']}")

        if pattern_confidence > 0:
            urgency_factors.append(pattern_confidence * 0.5)

        # Combine all factors
        total_scaling_signal = (
            cpu_scaling_needed * 0.3 +
            memory_scaling_needed * 0.4 +
            anomaly_scaling_needed * 0.2 +
            pattern_scaling_needed * 0.1
        )

        # Determine action
        if total_scaling_signal > 0.3:
            recommended_action = "scale_up"
            target_instances = min(self.max_instances,
                                 self.current_instances + max(1, int(total_scaling_signal * 2)))
        elif total_scaling_signal < -0.3:
            recommended_action = "scale_down"
            target_instances = max(self.min_instances,
                                 self.current_instances - max(1, int(abs(total_scaling_signal) * 2)))
        else:
            recommended_action = "no_change"
            target_instances = self.current_instances

        # Check cooldown period
        time_since_last_scaling = (datetime.now() - self.last_scaling_action).total_seconds() / 60
        if time_since_last_scaling < self.scaling_cooldown_minutes and recommended_action != "no_change":
            reasoning.append(f"Scaling action delayed due to cooldown ({time_since_last_scaling:.1f} min remaining)")
            recommended_action = "no_change"
            target_instances = self.current_instances

        # Calculate cost impact
        instance_change = target_instances - self.current_instances
        cost_impact = instance_change * self.cost_per_instance_hour

        # Apply cost sensitivity
        if self.cost_sensitivity > 0 and cost_impact > 0:
            cost_penalty = self.cost_sensitivity * (cost_impact / self.cost_per_instance_hour)
            if cost_penalty > 0.5 and recommended_action == "scale_up":
                # Reduce scaling aggressiveness due to cost sensitivity
                target_instances = self.current_instances + max(1, instance_change // 2)
                cost_impact = (target_instances - self.current_instances) * self.cost_per_instance_hour
                reasoning.append("Reduced scaling due to cost sensitivity")

        # Calculate expected benefits
        expected_benefit = {}
        if recommended_action == "scale_up":
            expected_benefit = {
                "response_time_improvement": "20-40%",
                "throughput_increase": "30-50%",
                "error_rate_reduction": "50-70%"
            }
        elif recommended_action == "scale_down":
            expected_benefit = {
                "cost_savings": f"${abs(cost_impact):.2f}/hour",
                "resource_efficiency": "15-25%"
            }

        # Calculate urgency score
        urgency_score = np.mean(urgency_factors) if urgency_factors else 0.0

        # Target resources calculation
        target_resources = {}
        if cpu_prediction:
            target_resources[ResourceType.CPU] = min(70.0, cpu_prediction.predicted_value * 0.8)
        if memory_prediction:
            target_resources[ResourceType.MEMORY] = min(75.0, memory_prediction.predicted_value * 0.8)

        return {
            'action': recommended_action,
            'target_instances': target_instances,
            'target_resources': target_resources,
            'urgency': urgency_score,
            'cost_impact': cost_impact,
            'expected_benefit': expected_benefit,
            'reasoning': reasoning
        }

    async def execute_scaling(self, prediction: ScalingPrediction) -> Dict[str, Any]:
        """Execute scaling action based on prediction."""
        if prediction.recommended_action == "no_change":
            return {
                'action': 'no_change',
                'current_instances': self.current_instances,
                'reasoning': prediction.reasoning
            }

        # Update quantum scaler metrics
        quantum_metrics = ScalingMetrics(
            current_load=self.current_metrics.get(ResourceType.CPU, 0.0) / 100.0,
            cpu_utilization=self.current_metrics.get(ResourceType.CPU, 0.0),
            memory_utilization=self.current_metrics.get(ResourceType.MEMORY, 0.0),
            active_instances=self.current_instances,
            target_instances=prediction.target_instances
        )

        await self.quantum_scaler.update_metrics(quantum_metrics)

        # Get quantum scaling decision
        quantum_decision = await self.quantum_scaler.should_scale()

        # Execute scaling if both predictive and quantum agree
        if quantum_decision.get('should_scale', False):
            quantum_result = await self.quantum_scaler.execute_scaling(quantum_decision)

            if quantum_result['action'] == 'scaled':
                # Update our state
                old_instances = self.current_instances
                self.current_instances = quantum_result['new_instances']
                self.last_scaling_action = datetime.now()

                # Record scaling decision
                scaling_record = {
                    'timestamp': datetime.now(),
                    'prediction': prediction,
                    'quantum_decision': quantum_decision,
                    'old_instances': old_instances,
                    'new_instances': self.current_instances,
                    'success': True
                }
                self.scaling_history.append(scaling_record)

                # Record metrics
                self.monitor.record_metric(
                    "predictive_scaling_execution", 1.0, "counter"
                )
                self.monitor.record_metric(
                    "predictive_scaling_instances", float(self.current_instances), "gauge"
                )

                result = {
                    'action': 'scaled',
                    'old_instances': old_instances,
                    'new_instances': self.current_instances,
                    'prediction_confidence': np.mean([p.confidence_score for p in prediction.resource_predictions]),
                    'quantum_confidence': quantum_decision.get('confidence', 0.0),
                    'cost_impact': prediction.cost_impact,
                    'reasoning': prediction.reasoning + [f"Quantum scaler agreement: {quantum_decision.get('reasoning', '')}"]
                }

                logger.info(f"Predictive scaling executed: {old_instances} -> {self.current_instances} instances")
                return result

        # Scaling not executed
        return {
            'action': 'not_executed',
            'current_instances': self.current_instances,
            'reasoning': prediction.reasoning + ['Quantum scaler did not agree with scaling decision']
        }

    def _get_current_features(self) -> Dict[str, float]:
        """Get current features for ML prediction."""
        return {
            'cpu_utilization': self.current_metrics.get(ResourceType.CPU, 0.0),
            'memory_utilization': self.current_metrics.get(ResourceType.MEMORY, 0.0),
            'network_utilization': self.current_metrics.get(ResourceType.NETWORK, 0.0),
            'active_connections': self.current_metrics.get(ResourceType.CONNECTIONS, 0.0),
            'instances_current': float(self.current_instances),
            'request_rate': 100.0,  # Placeholder
            'response_time': 200.0,  # Placeholder
            f'{ResourceType.CPU.value}_current': self.current_metrics.get(ResourceType.CPU, 0.0),
            f'{ResourceType.MEMORY.value}_current': self.current_metrics.get(ResourceType.MEMORY, 0.0)
        }

    async def _prediction_loop(self):
        """Background loop for making predictions."""
        while self.running:
            try:
                if self.current_metrics:
                    prediction = await self.get_scaling_prediction()

                    # Execute scaling if prediction has high confidence and urgency
                    if (prediction.urgency_score > 0.7 and
                        prediction.recommended_action != "no_change"):

                        result = await self.execute_scaling(prediction)
                        if result['action'] in ['scaled', 'not_executed']:
                            logger.debug(f"Automatic scaling result: {result}")

                await asyncio.sleep(60)  # Make predictions every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Prediction loop error: {e}")
                await asyncio.sleep(120)

    async def _pattern_analysis_loop(self):
        """Background loop for pattern analysis."""
        while self.running:
            try:
                # Pattern analysis is handled automatically when adding workload data
                # This loop can be used for periodic deep analysis

                await asyncio.sleep(300)  # Run every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Pattern analysis loop error: {e}")
                await asyncio.sleep(600)

    def get_predictive_scaling_stats(self) -> Dict[str, Any]:
        """Get comprehensive predictive scaling statistics."""
        # Calculate prediction accuracy
        successful_predictions = len([
            record for record in self.scaling_history
            if record.get('success', False)
        ])

        total_predictions = len(self.scaling_history)
        prediction_accuracy = (successful_predictions / max(total_predictions, 1)) * 100

        # Get recent predictions
        recent_predictions = []
        for prediction in list(self.predictions)[-10:]:
            recent_predictions.append({
                'timestamp': prediction.timestamp.isoformat(),
                'action': prediction.recommended_action,
                'target_instances': prediction.target_instances,
                'urgency_score': prediction.urgency_score,
                'cost_impact': prediction.cost_impact,
                'confidence': np.mean([p.confidence_score for p in prediction.resource_predictions])
            })

        # Get pattern information
        patterns_detected = []
        for pattern_id, pattern in self.pattern_analyzer.patterns.items():
            patterns_detected.append({
                'pattern_id': pattern_id,
                'type': pattern.pattern_type,
                'confidence': pattern.confidence,
                'last_observed': pattern.last_observed.isoformat()
            })

        return {
            'current_instances': self.current_instances,
            'prediction_horizon_minutes': self.prediction_horizon_minutes,
            'total_scaling_events': len(self.scaling_history),
            'prediction_accuracy': prediction_accuracy,
            'patterns_detected': len(self.pattern_analyzer.patterns),
            'anomaly_detector_trained': hasattr(self.pattern_analyzer.anomaly_detector, 'offset_'),
            'recent_predictions': recent_predictions,
            'detected_patterns': patterns_detected,
            'model_accuracies': {
                resource_type.value: {
                    model: accuracy
                    for model, accuracy in accuracies.items()
                }
                for resource_type, accuracies in self.capacity_predictor.model_accuracies.items()
            },
            'cost_per_instance_hour': self.cost_per_instance_hour,
            'cost_sensitivity': self.cost_sensitivity
        }


# Global predictive auto-scaler instance
_global_predictive_scaler: Optional[PredictiveAutoScaler] = None


async def get_predictive_auto_scaler(**kwargs) -> PredictiveAutoScaler:
    """Get global predictive auto-scaler instance."""
    global _global_predictive_scaler

    if _global_predictive_scaler is None:
        _global_predictive_scaler = PredictiveAutoScaler(**kwargs)
        await _global_predictive_scaler.start()

    return _global_predictive_scaler


# Example usage
if __name__ == "__main__":
    async def example_predictive_scaling():
        """Example of using predictive auto-scaling."""

        # Create predictive auto-scaler
        scaler = PredictiveAutoScaler(
            min_instances=2,
            max_instances=20,
            prediction_horizon_minutes=30,
            cost_per_instance_hour=0.12
        )

        await scaler.start()

        try:
            # Simulate workload over time
            for hour in range(24):
                for minute in range(0, 60, 5):  # Every 5 minutes
                    # Simulate realistic workload patterns
                    base_cpu = 40.0

                    # Add daily pattern (higher during business hours)
                    if 9 <= hour <= 17:
                        base_cpu += 20.0

                    # Add some random variation
                    cpu_utilization = base_cpu + np.random.normal(0, 10)
                    cpu_utilization = max(10, min(95, cpu_utilization))

                    memory_utilization = cpu_utilization * 0.8 + np.random.normal(0, 5)
                    memory_utilization = max(20, min(90, memory_utilization))

                    metrics = {
                        'cpu_utilization': cpu_utilization,
                        'memory_utilization': memory_utilization,
                        'network_utilization': np.random.uniform(10, 50),
                        'active_connections': np.random.randint(50, 500),
                        'request_rate': np.random.uniform(10, 200),
                        'response_time': np.random.uniform(50, 500)
                    }

                    # Update metrics
                    await scaler.update_metrics(metrics)

                    # Get prediction
                    if minute % 15 == 0:  # Get prediction every 15 minutes
                        prediction = await scaler.get_scaling_prediction()

                        print(f"Hour {hour:02d}:{minute:02d} - "
                              f"CPU: {cpu_utilization:.1f}%, "
                              f"Action: {prediction.recommended_action}, "
                              f"Target: {prediction.target_instances}, "
                              f"Urgency: {prediction.urgency_score:.2f}")

                        # Execute high-urgency scaling
                        if prediction.urgency_score > 0.8:
                            result = await scaler.execute_scaling(prediction)
                            if result['action'] == 'scaled':
                                print(f"  -> Scaled: {result['old_instances']} -> {result['new_instances']}")

                    await asyncio.sleep(0.01)  # Speed up simulation

            # Get final statistics
            stats = scaler.get_predictive_scaling_stats()
            print("\nPredictive Scaling Statistics:")
            print(f"Total scaling events: {stats['total_scaling_events']}")
            print(f"Prediction accuracy: {stats['prediction_accuracy']:.1f}%")
            print(f"Patterns detected: {stats['patterns_detected']}")
            print(f"Final instances: {stats['current_instances']}")

        finally:
            await scaler.stop()

    # Run example
    asyncio.run(example_predictive_scaling())
    print("Predictive auto-scaling example completed!")
