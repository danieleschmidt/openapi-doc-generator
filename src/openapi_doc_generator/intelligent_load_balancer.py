"""
Advanced Intelligent Load Balancing System

This module implements sophisticated load balancing algorithms with real-time metrics,
machine learning-based predictions, and quantum-inspired distribution strategies
for optimal resource allocation across distributed systems.

Features:
- Real-time performance metrics collection
- Quantum-inspired load distribution algorithms  
- ML-based capacity planning and prediction
- Dynamic health monitoring and failover
- Geographic awareness for global deployments
- Advanced routing with sticky sessions and affinity
"""

import asyncio
import hashlib
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from .enhanced_monitoring import get_monitor
from .performance_optimizer import get_optimizer
from .quantum_scale_optimizer import QuantumLoadBalancer

logger = logging.getLogger(__name__)


class LoadBalancingAlgorithm(Enum):
    """Advanced load balancing algorithms."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    IP_HASH = "ip_hash"
    GEOGRAPHIC = "geographic"
    QUANTUM_WEIGHTED = "quantum_weighted"
    ML_PREDICTIVE = "ml_predictive"
    HYBRID_ADAPTIVE = "hybrid_adaptive"


class HealthStatus(Enum):
    """Instance health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class InstanceType(Enum):
    """Types of instances for specialized handling."""
    COMPUTE = "compute"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    HYBRID = "hybrid"
    EDGE = "edge"
    QUANTUM = "quantum"


@dataclass
class LoadBalancingMetrics:
    """Comprehensive metrics for load balancing decisions."""
    # Performance metrics
    response_time_ms: float = 0.0
    throughput_rps: float = 0.0
    active_connections: int = 0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    network_utilization: float = 0.0

    # Load metrics
    current_load: float = 0.0
    predicted_load: float = 0.0
    load_trend: float = 0.0
    queue_depth: int = 0

    # Health metrics
    error_rate: float = 0.0
    success_rate: float = 100.0
    availability: float = 100.0
    last_health_check: Optional[datetime] = None

    # Geographic metrics
    region: str = "default"
    availability_zone: str = "default"
    latency_ms: float = 0.0

    # Quantum metrics
    quantum_coherence: float = 1.0
    entanglement_strength: float = 0.0

    # Capacity metrics
    max_capacity: int = 100
    current_capacity_usage: float = 0.0

    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class InstanceConfiguration:
    """Configuration for load balanced instances."""
    instance_id: str
    address: str
    port: int
    weight: float = 1.0
    instance_type: InstanceType = InstanceType.HYBRID
    region: str = "default"
    availability_zone: str = "default"
    tags: Dict[str, str] = field(default_factory=dict)

    # Capacity configuration
    max_connections: int = 1000
    max_rps: int = 1000
    max_cpu_percent: float = 90.0
    max_memory_percent: float = 90.0

    # Health check configuration
    health_check_url: str = "/health"
    health_check_interval: float = 30.0
    health_check_timeout: float = 5.0
    unhealthy_threshold: int = 3
    healthy_threshold: int = 2

    # Quantum configuration
    quantum_enabled: bool = False
    quantum_coherence_required: float = 0.8


@dataclass
class LoadBalancingDecision:
    """Decision made by the load balancer."""
    selected_instance: str
    algorithm_used: str
    decision_factors: Dict[str, float]
    confidence: float
    predicted_performance: Dict[str, float]
    fallback_instances: List[str]
    session_affinity: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class IntelligentLoadBalancer:
    """
    Advanced load balancer with intelligent routing, real-time metrics,
    and machine learning-based predictions.
    """

    def __init__(self,
                 algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.HYBRID_ADAPTIVE,
                 enable_session_affinity: bool = True,
                 enable_health_monitoring: bool = True,
                 enable_ml_predictions: bool = True):

        self.algorithm = algorithm
        self.enable_session_affinity = enable_session_affinity
        self.enable_health_monitoring = enable_health_monitoring
        self.enable_ml_predictions = enable_ml_predictions

        # Instance management
        self.instances: Dict[str, InstanceConfiguration] = {}
        self.instance_metrics: Dict[str, LoadBalancingMetrics] = {}
        self.instance_health: Dict[str, HealthStatus] = {}

        # Session management
        self.session_affinity_map: Dict[str, str] = {}  # session_id -> instance_id
        self.sticky_sessions: Dict[str, datetime] = {}  # session_id -> last_access
        self.session_ttl: float = 3600.0  # 1 hour

        # Performance tracking
        self.request_history: deque = deque(maxlen=10000)
        self.instance_performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.decision_history: deque = deque(maxlen=1000)

        # Machine learning components
        self.ml_predictor = MLLoadPredictor() if enable_ml_predictions else None
        self.pattern_detector = LoadPatternDetector()

        # Quantum components
        self.quantum_balancer = QuantumLoadBalancer([])

        # Monitoring and optimization
        self.monitor = get_monitor()
        self.optimizer = get_optimizer()

        # Background tasks
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.metrics_collector_task: Optional[asyncio.Task] = None
        self.ml_trainer_task: Optional[asyncio.Task] = None
        self.running = False

        # Thread safety
        self.lock = threading.RLock()

        logger.info(f"Intelligent load balancer initialized with algorithm: {algorithm.value}")

    async def start(self):
        """Start the load balancer and background tasks."""
        if self.running:
            return

        self.running = True

        # Start background tasks
        if self.enable_health_monitoring:
            self.health_monitor_task = asyncio.create_task(self._health_monitoring_loop())

        self.metrics_collector_task = asyncio.create_task(self._metrics_collection_loop())

        if self.enable_ml_predictions and self.ml_predictor:
            self.ml_trainer_task = asyncio.create_task(self._ml_training_loop())

        logger.info("Intelligent load balancer started")

    async def stop(self):
        """Stop the load balancer and cleanup."""
        if not self.running:
            return

        self.running = False

        # Cancel background tasks
        tasks = [self.health_monitor_task, self.metrics_collector_task, self.ml_trainer_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()

        # Wait for tasks to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("Intelligent load balancer stopped")

    def add_instance(self, config: InstanceConfiguration):
        """Add a new instance to the load balancer."""
        with self.lock:
            self.instances[config.instance_id] = config
            self.instance_metrics[config.instance_id] = LoadBalancingMetrics(
                region=config.region,
                availability_zone=config.availability_zone,
                max_capacity=config.max_connections
            )
            self.instance_health[config.instance_id] = HealthStatus.HEALTHY

            # Update quantum balancer
            quantum_instances = list(self.instances.keys())
            self.quantum_balancer = QuantumLoadBalancer(quantum_instances)

        logger.info(f"Added instance: {config.instance_id} at {config.address}:{config.port}")

    def remove_instance(self, instance_id: str):
        """Remove an instance from the load balancer."""
        with self.lock:
            if instance_id in self.instances:
                del self.instances[instance_id]
                del self.instance_metrics[instance_id]
                del self.instance_health[instance_id]

                # Remove from session affinity mappings
                sessions_to_remove = [
                    session_id for session_id, mapped_instance
                    in self.session_affinity_map.items()
                    if mapped_instance == instance_id
                ]
                for session_id in sessions_to_remove:
                    del self.session_affinity_map[session_id]

                # Update quantum balancer
                quantum_instances = list(self.instances.keys())
                if quantum_instances:
                    self.quantum_balancer = QuantumLoadBalancer(quantum_instances)

        logger.info(f"Removed instance: {instance_id}")

    async def select_instance(self,
                            request_context: Optional[Dict[str, Any]] = None,
                            session_id: Optional[str] = None,
                            client_ip: Optional[str] = None,
                            requirements: Optional[Dict[str, Any]] = None) -> LoadBalancingDecision:
        """
        Select the optimal instance for handling a request using intelligent algorithms.
        """
        with self.lock:
            if not self.instances:
                raise RuntimeError("No healthy instances available")

            # Check session affinity first
            if self.enable_session_affinity and session_id:
                affinity_instance = await self._check_session_affinity(session_id)
                if affinity_instance:
                    decision = LoadBalancingDecision(
                        selected_instance=affinity_instance,
                        algorithm_used="session_affinity",
                        decision_factors={"session_affinity": 1.0},
                        confidence=0.9,
                        predicted_performance={"response_time_ms": 50.0},
                        fallback_instances=[]
                    )
                    await self._record_decision(decision, request_context)
                    return decision

            # Get healthy instances
            healthy_instances = await self._get_healthy_instances(requirements)

            if not healthy_instances:
                raise RuntimeError("No healthy instances meeting requirements")

            # Select algorithm based on configuration
            if self.algorithm == LoadBalancingAlgorithm.HYBRID_ADAPTIVE:
                decision = await self._hybrid_adaptive_selection(
                    healthy_instances, request_context, requirements
                )
            elif self.algorithm == LoadBalancingAlgorithm.ML_PREDICTIVE:
                decision = await self._ml_predictive_selection(
                    healthy_instances, request_context, requirements
                )
            elif self.algorithm == LoadBalancingAlgorithm.QUANTUM_WEIGHTED:
                decision = await self._quantum_weighted_selection(
                    healthy_instances, request_context, requirements
                )
            elif self.algorithm == LoadBalancingAlgorithm.GEOGRAPHIC:
                decision = await self._geographic_selection(
                    healthy_instances, request_context, client_ip
                )
            elif self.algorithm == LoadBalancingAlgorithm.LEAST_RESPONSE_TIME:
                decision = await self._least_response_time_selection(healthy_instances)
            elif self.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
                decision = await self._least_connections_selection(healthy_instances)
            elif self.algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
                decision = await self._weighted_round_robin_selection(healthy_instances)
            elif self.algorithm == LoadBalancingAlgorithm.IP_HASH:
                decision = await self._ip_hash_selection(healthy_instances, client_ip)
            else:  # ROUND_ROBIN
                decision = await self._round_robin_selection(healthy_instances)

            # Establish session affinity if enabled
            if self.enable_session_affinity and session_id:
                self.session_affinity_map[session_id] = decision.selected_instance
                self.sticky_sessions[session_id] = datetime.now()
                decision.session_affinity = session_id

            await self._record_decision(decision, request_context)
            return decision

    async def _check_session_affinity(self, session_id: str) -> Optional[str]:
        """Check if session has affinity to a specific instance."""
        if session_id not in self.session_affinity_map:
            return None

        instance_id = self.session_affinity_map[session_id]

        # Check if session is still valid
        if session_id in self.sticky_sessions:
            last_access = self.sticky_sessions[session_id]
            if datetime.now() - last_access > timedelta(seconds=self.session_ttl):
                # Session expired
                del self.session_affinity_map[session_id]
                del self.sticky_sessions[session_id]
                return None

        # Check if instance is still healthy
        if (instance_id in self.instances and
            self.instance_health.get(instance_id) in [HealthStatus.HEALTHY, HealthStatus.WARNING]):
            # Update last access time
            self.sticky_sessions[session_id] = datetime.now()
            return instance_id

        # Instance not healthy, remove affinity
        del self.session_affinity_map[session_id]
        if session_id in self.sticky_sessions:
            del self.sticky_sessions[session_id]

        return None

    async def _get_healthy_instances(self,
                                   requirements: Optional[Dict[str, Any]] = None) -> List[str]:
        """Get list of healthy instances that meet requirements."""
        healthy_instances = []

        for instance_id, config in self.instances.items():
            # Check health status
            health = self.instance_health.get(instance_id, HealthStatus.OFFLINE)
            if health in [HealthStatus.OFFLINE, HealthStatus.CRITICAL]:
                continue

            # Check requirements
            if requirements:
                # Instance type requirement
                required_type = requirements.get('instance_type')
                if required_type and config.instance_type != InstanceType(required_type):
                    continue

                # Region requirement
                required_region = requirements.get('region')
                if required_region and config.region != required_region:
                    continue

                # Quantum coherence requirement
                required_coherence = requirements.get('min_quantum_coherence', 0.0)
                if required_coherence > 0.0:
                    metrics = self.instance_metrics.get(instance_id)
                    if not metrics or metrics.quantum_coherence < required_coherence:
                        continue

                # Capacity requirement
                max_load = requirements.get('max_load_percent', 100.0)
                metrics = self.instance_metrics.get(instance_id)
                if metrics and metrics.current_capacity_usage > max_load:
                    continue

            healthy_instances.append(instance_id)

        return healthy_instances

    async def _hybrid_adaptive_selection(self,
                                       instances: List[str],
                                       request_context: Optional[Dict[str, Any]],
                                       requirements: Optional[Dict[str, Any]]) -> LoadBalancingDecision:
        """
        Hybrid adaptive selection combining multiple algorithms based on current conditions.
        """
        # Analyze current system state
        system_load = await self._calculate_system_load()

        # Choose best algorithm based on conditions
        if system_load < 0.3:
            # Low load - use simple round robin for even distribution
            decision = await self._round_robin_selection(instances)
            decision.algorithm_used = "hybrid_adaptive:round_robin"

        elif system_load > 0.8:
            # High load - use ML predictions for optimal performance
            if self.ml_predictor:
                decision = await self._ml_predictive_selection(instances, request_context, requirements)
                decision.algorithm_used = "hybrid_adaptive:ml_predictive"
            else:
                decision = await self._least_response_time_selection(instances)
                decision.algorithm_used = "hybrid_adaptive:least_response_time"

        else:
            # Medium load - use quantum-weighted for balanced optimization
            decision = await self._quantum_weighted_selection(instances, request_context, requirements)
            decision.algorithm_used = "hybrid_adaptive:quantum_weighted"

        # Add confidence adjustment based on system conditions
        confidence_adjustment = 1.0 - (system_load * 0.2)  # Reduce confidence at high load
        decision.confidence *= confidence_adjustment

        return decision

    async def _ml_predictive_selection(self,
                                     instances: List[str],
                                     request_context: Optional[Dict[str, Any]],
                                     requirements: Optional[Dict[str, Any]]) -> LoadBalancingDecision:
        """ML-based predictive instance selection."""
        if not self.ml_predictor:
            # Fallback to quantum weighted
            return await self._quantum_weighted_selection(instances, request_context, requirements)

        # Get ML predictions for each instance
        predictions = {}
        for instance_id in instances:
            metrics = self.instance_metrics.get(instance_id)
            if metrics:
                context_features = self._extract_request_features(request_context)
                prediction = await self.ml_predictor.predict_performance(
                    instance_id, metrics, context_features
                )
                predictions[instance_id] = prediction

        if not predictions:
            return await self._quantum_weighted_selection(instances, request_context, requirements)

        # Select instance with best predicted performance
        best_instance = min(predictions.keys(),
                          key=lambda i: predictions[i].get('response_time_ms', float('inf')))

        decision_factors = {
            f"ml_response_time_{instance_id}": predictions[instance_id].get('response_time_ms', 0.0)
            for instance_id in predictions.keys()
        }

        return LoadBalancingDecision(
            selected_instance=best_instance,
            algorithm_used="ml_predictive",
            decision_factors=decision_factors,
            confidence=predictions[best_instance].get('confidence', 0.7),
            predicted_performance=predictions[best_instance],
            fallback_instances=[i for i in instances if i != best_instance][:3]
        )

    async def _quantum_weighted_selection(self,
                                        instances: List[str],
                                        request_context: Optional[Dict[str, Any]],
                                        requirements: Optional[Dict[str, Any]]) -> LoadBalancingDecision:
        """Quantum-inspired weighted selection with superposition and entanglement."""
        # Create quantum task for selection
        from .quantum_scaler import QuantumTask

        task = QuantumTask(
            task_id=f"lb_selection_{time.time()}",
            operation="load_balancing",
            priority=1.0,
            estimated_duration=0.1,
            estimated_memory=10.0,
            quantum_complexity=2.0,
            metadata={'instances': instances, 'context': request_context}
        )

        # Use quantum load balancer for selection
        selected_index = self.quantum_balancer.select_optimal_worker(task)
        selected_instance = instances[selected_index % len(instances)]

        # Calculate quantum factors
        quantum_factors = {}
        for i, instance_id in enumerate(instances):
            metrics = self.instance_metrics.get(instance_id, LoadBalancingMetrics())

            # Quantum weight calculation
            coherence_factor = metrics.quantum_coherence
            load_factor = 1.0 - (metrics.current_capacity_usage / 100.0)
            performance_factor = 1.0 / (1.0 + metrics.response_time_ms / 100.0)

            quantum_weight = (coherence_factor * 0.4 +
                            load_factor * 0.3 +
                            performance_factor * 0.3)

            quantum_factors[f"quantum_weight_{instance_id}"] = quantum_weight

        return LoadBalancingDecision(
            selected_instance=selected_instance,
            algorithm_used="quantum_weighted",
            decision_factors=quantum_factors,
            confidence=0.85,
            predicted_performance={"response_time_ms": 75.0, "throughput_rps": 500.0},
            fallback_instances=[i for i in instances if i != selected_instance][:2]
        )

    async def _geographic_selection(self,
                                  instances: List[str],
                                  request_context: Optional[Dict[str, Any]],
                                  client_ip: Optional[str]) -> LoadBalancingDecision:
        """Geographic-aware instance selection."""
        # Determine client location (simplified implementation)
        client_region = self._determine_client_region(client_ip) if client_ip else "default"

        # Score instances based on geographic proximity
        scores = {}
        for instance_id in instances:
            config = self.instances[instance_id]
            metrics = self.instance_metrics.get(instance_id, LoadBalancingMetrics())

            # Geographic proximity score
            region_match = 1.0 if config.region == client_region else 0.5
            latency_score = 1.0 / (1.0 + metrics.latency_ms / 100.0)
            availability_score = metrics.availability / 100.0

            total_score = region_match * 0.5 + latency_score * 0.3 + availability_score * 0.2
            scores[instance_id] = total_score

        best_instance = max(scores.keys(), key=lambda i: scores[i])

        return LoadBalancingDecision(
            selected_instance=best_instance,
            algorithm_used="geographic",
            decision_factors=scores,
            confidence=0.8,
            predicted_performance={"latency_ms": self.instance_metrics[best_instance].latency_ms},
            fallback_instances=sorted(instances, key=lambda i: scores[i], reverse=True)[1:4]
        )

    async def _least_response_time_selection(self, instances: List[str]) -> LoadBalancingDecision:
        """Select instance with lowest average response time."""
        response_times = {}

        for instance_id in instances:
            metrics = self.instance_metrics.get(instance_id, LoadBalancingMetrics())
            response_times[instance_id] = metrics.response_time_ms

        best_instance = min(response_times.keys(), key=lambda i: response_times[i])

        return LoadBalancingDecision(
            selected_instance=best_instance,
            algorithm_used="least_response_time",
            decision_factors=response_times,
            confidence=0.75,
            predicted_performance={"response_time_ms": response_times[best_instance]},
            fallback_instances=sorted(instances, key=lambda i: response_times[i])[1:3]
        )

    async def _least_connections_selection(self, instances: List[str]) -> LoadBalancingDecision:
        """Select instance with fewest active connections."""
        connections = {}

        for instance_id in instances:
            metrics = self.instance_metrics.get(instance_id, LoadBalancingMetrics())
            connections[instance_id] = metrics.active_connections

        best_instance = min(connections.keys(), key=lambda i: connections[i])

        return LoadBalancingDecision(
            selected_instance=best_instance,
            algorithm_used="least_connections",
            decision_factors=connections,
            confidence=0.7,
            predicted_performance={"active_connections": connections[best_instance]},
            fallback_instances=sorted(instances, key=lambda i: connections[i])[1:3]
        )

    async def _weighted_round_robin_selection(self, instances: List[str]) -> LoadBalancingDecision:
        """Weighted round robin selection based on instance weights."""
        weights = []

        for instance_id in instances:
            config = self.instances[instance_id]
            metrics = self.instance_metrics.get(instance_id, LoadBalancingMetrics())

            # Adjust weight based on current performance
            performance_factor = 1.0 - (metrics.current_capacity_usage / 200.0)  # 0.5 to 1.0
            effective_weight = config.weight * max(0.1, performance_factor)
            weights.append(effective_weight)

        # Select based on weights
        total_weight = sum(weights)
        if total_weight == 0:
            selected_instance = instances[0]
        else:
            cumulative_weights = np.cumsum(weights)
            random_value = np.random.random() * total_weight
            selected_index = np.searchsorted(cumulative_weights, random_value)
            selected_instance = instances[selected_index]

        return LoadBalancingDecision(
            selected_instance=selected_instance,
            algorithm_used="weighted_round_robin",
            decision_factors={f"weight_{i}": w for i, w in zip(instances, weights)},
            confidence=0.65,
            predicted_performance={"weight": weights[instances.index(selected_instance)]},
            fallback_instances=[i for i in instances if i != selected_instance][:2]
        )

    async def _ip_hash_selection(self,
                               instances: List[str],
                               client_ip: Optional[str]) -> LoadBalancingDecision:
        """Consistent hash-based selection using client IP."""
        if not client_ip:
            # Fallback to round robin
            return await self._round_robin_selection(instances)

        # Hash client IP to select instance
        ip_hash = hashlib.sha256(client_ip.encode()).hexdigest()
        hash_value = int(ip_hash[:8], 16)
        selected_index = hash_value % len(instances)
        selected_instance = instances[selected_index]

        return LoadBalancingDecision(
            selected_instance=selected_instance,
            algorithm_used="ip_hash",
            decision_factors={"ip_hash": hash_value, "client_ip": client_ip},
            confidence=0.6,
            predicted_performance={"consistency": 1.0},
            fallback_instances=[instances[(selected_index + 1) % len(instances)]]
        )

    async def _round_robin_selection(self, instances: List[str]) -> LoadBalancingDecision:
        """Simple round robin selection."""
        # Use timestamp-based rotation
        current_time = int(time.time())
        selected_index = current_time % len(instances)
        selected_instance = instances[selected_index]

        return LoadBalancingDecision(
            selected_instance=selected_instance,
            algorithm_used="round_robin",
            decision_factors={"rotation_index": selected_index},
            confidence=0.5,
            predicted_performance={"fairness": 1.0},
            fallback_instances=[instances[(selected_index + 1) % len(instances)]]
        )

    async def _calculate_system_load(self) -> float:
        """Calculate overall system load across all instances."""
        if not self.instances:
            return 0.0

        total_load = 0.0
        instance_count = 0

        for instance_id in self.instances:
            metrics = self.instance_metrics.get(instance_id)
            if metrics:
                # Combine multiple load factors
                cpu_load = metrics.cpu_utilization / 100.0
                memory_load = metrics.memory_utilization / 100.0
                capacity_load = metrics.current_capacity_usage / 100.0

                instance_load = (cpu_load + memory_load + capacity_load) / 3.0
                total_load += instance_load
                instance_count += 1

        return total_load / max(instance_count, 1)

    def _determine_client_region(self, client_ip: str) -> str:
        """Determine client region from IP address (simplified implementation)."""
        # In a real implementation, this would use a GeoIP database
        # For now, return a simple hash-based region assignment
        ip_hash = hashlib.sha256(client_ip.encode()).hexdigest()
        regions = ["us-east", "us-west", "eu-central", "asia-pacific"]
        region_index = int(ip_hash[:2], 16) % len(regions)
        return regions[region_index]

    def _extract_request_features(self, request_context: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Extract numerical features from request context for ML."""
        if not request_context:
            return {}

        features = {}

        # Extract numerical features
        for key, value in request_context.items():
            if isinstance(value, (int, float)):
                features[key] = float(value)
            elif isinstance(value, str):
                # Convert string features to numerical
                features[f"{key}_hash"] = float(hash(value) % 10000) / 10000.0
                features[f"{key}_length"] = float(len(value))

        return features

    async def _record_decision(self,
                             decision: LoadBalancingDecision,
                             request_context: Optional[Dict[str, Any]]):
        """Record load balancing decision for analysis."""
        self.decision_history.append(decision)

        # Record metrics
        self.monitor.record_metric(
            f"load_balancer_decision_{decision.algorithm_used}",
            1.0,
            "counter"
        )

        self.monitor.record_metric(
            "load_balancer_confidence",
            decision.confidence,
            "gauge"
        )

    async def update_instance_metrics(self,
                                    instance_id: str,
                                    metrics: LoadBalancingMetrics):
        """Update metrics for a specific instance."""
        if instance_id in self.instances:
            self.instance_metrics[instance_id] = metrics
            self.instance_performance_history[instance_id].append(metrics)

            # Update quantum balancer if needed
            if hasattr(self.quantum_balancer, 'update_worker_performance'):
                # Convert metrics to quantum balancer format
                worker_index = list(self.instances.keys()).index(instance_id)
                success_rate = metrics.success_rate / 100.0
                avg_duration = metrics.response_time_ms / 1000.0

                self.quantum_balancer.update_worker_performance(
                    worker_index,
                    f"request_{time.time()}",
                    success_rate > 0.95,
                    avg_duration
                )

    async def _health_monitoring_loop(self):
        """Background task for monitoring instance health."""
        while self.running:
            try:
                for instance_id, config in self.instances.items():
                    try:
                        # Perform health check (simplified)
                        health_status = await self._perform_health_check(instance_id, config)
                        self.instance_health[instance_id] = health_status

                    except Exception as e:
                        logger.warning(f"Health check failed for {instance_id}: {e}")
                        self.instance_health[instance_id] = HealthStatus.CRITICAL

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)

    async def _perform_health_check(self,
                                   instance_id: str,
                                   config: InstanceConfiguration) -> HealthStatus:
        """Perform health check on a specific instance."""
        try:
            # Simulate health check (in production, would make HTTP request)
            metrics = self.instance_metrics.get(instance_id, LoadBalancingMetrics())

            # Determine health based on metrics
            if metrics.error_rate > 20.0:
                return HealthStatus.CRITICAL
            elif metrics.error_rate > 10.0 or metrics.cpu_utilization > 90.0:
                return HealthStatus.WARNING
            elif metrics.cpu_utilization > 80.0 or metrics.memory_utilization > 80.0:
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.HEALTHY

        except Exception as e:
            logger.error(f"Health check error for {instance_id}: {e}")
            return HealthStatus.CRITICAL

    async def _metrics_collection_loop(self):
        """Background task for collecting performance metrics."""
        while self.running:
            try:
                for instance_id in self.instances:
                    try:
                        # Simulate metrics collection (in production, would query monitoring systems)
                        metrics = await self._collect_instance_metrics(instance_id)
                        if metrics:
                            await self.update_instance_metrics(instance_id, metrics)

                    except Exception as e:
                        logger.warning(f"Metrics collection failed for {instance_id}: {e}")

                await asyncio.sleep(10)  # Collect every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(30)

    async def _collect_instance_metrics(self, instance_id: str) -> Optional[LoadBalancingMetrics]:
        """Collect metrics for a specific instance."""
        try:
            # Simulate metrics collection with some realistic variations
            base_metrics = self.instance_metrics.get(instance_id, LoadBalancingMetrics())

            # Add some random variation to simulate real metrics
            import random

            return LoadBalancingMetrics(
                response_time_ms=base_metrics.response_time_ms * (0.8 + random.random() * 0.4),
                throughput_rps=max(0, base_metrics.throughput_rps * (0.9 + random.random() * 0.2)),
                active_connections=max(0, int(base_metrics.active_connections * (0.8 + random.random() * 0.4))),
                cpu_utilization=max(0, min(100, base_metrics.cpu_utilization + random.uniform(-5, 5))),
                memory_utilization=max(0, min(100, base_metrics.memory_utilization + random.uniform(-3, 3))),
                current_load=max(0, min(1, base_metrics.current_load + random.uniform(-0.1, 0.1))),
                error_rate=max(0, base_metrics.error_rate * (0.5 + random.random())),
                success_rate=min(100, base_metrics.success_rate + random.uniform(-1, 1)),
                availability=min(100, base_metrics.availability + random.uniform(-0.5, 0.5)),
                quantum_coherence=max(0.5, min(1.0, base_metrics.quantum_coherence + random.uniform(-0.02, 0.02))),
                region=base_metrics.region,
                availability_zone=base_metrics.availability_zone,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Error collecting metrics for {instance_id}: {e}")
            return None

    async def _ml_training_loop(self):
        """Background task for training ML models."""
        if not self.ml_predictor:
            return

        while self.running:
            try:
                # Train ML models with recent data
                if len(self.decision_history) >= 100:  # Minimum data for training
                    await self.ml_predictor.train_models(
                        list(self.decision_history)[-1000:],  # Last 1000 decisions
                        self.instance_performance_history
                    )

                await asyncio.sleep(300)  # Train every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"ML training error: {e}")
                await asyncio.sleep(600)

    def get_load_balancing_stats(self) -> Dict[str, Any]:
        """Get comprehensive load balancing statistics."""
        with self.lock:
            return {
                "algorithm": self.algorithm.value,
                "total_instances": len(self.instances),
                "healthy_instances": len([
                    i for i, health in self.instance_health.items()
                    if health in [HealthStatus.HEALTHY, HealthStatus.WARNING]
                ]),
                "session_affinity_enabled": self.enable_session_affinity,
                "active_sessions": len(self.session_affinity_map),
                "total_decisions": len(self.decision_history),
                "recent_decisions": [
                    {
                        "algorithm": d.algorithm_used,
                        "selected_instance": d.selected_instance,
                        "confidence": d.confidence,
                        "timestamp": d.timestamp.isoformat()
                    }
                    for d in list(self.decision_history)[-10:]  # Last 10 decisions
                ],
                "instance_health": {
                    instance_id: health.value
                    for instance_id, health in self.instance_health.items()
                },
                "instance_metrics_summary": {
                    instance_id: {
                        "response_time_ms": metrics.response_time_ms,
                        "cpu_utilization": metrics.cpu_utilization,
                        "memory_utilization": metrics.memory_utilization,
                        "success_rate": metrics.success_rate,
                        "availability": metrics.availability
                    }
                    for instance_id, metrics in self.instance_metrics.items()
                }
            }


class MLLoadPredictor:
    """Machine learning-based load prediction for intelligent routing."""

    def __init__(self):
        self.models: Dict[str, Any] = {}  # Instance-specific models
        self.feature_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.training_data: deque = deque(maxlen=5000)

    async def predict_performance(self,
                                instance_id: str,
                                current_metrics: LoadBalancingMetrics,
                                request_features: Dict[str, float]) -> Dict[str, float]:
        """Predict performance metrics for an instance."""
        # Simplified ML prediction (in production, would use scikit-learn or similar)

        # Base prediction from current metrics
        base_response_time = current_metrics.response_time_ms
        base_throughput = current_metrics.throughput_rps

        # Apply some learned adjustments based on patterns
        load_factor = current_metrics.current_load
        adjustment_factor = 1.0 + (load_factor * 0.5)  # Higher load = higher response time

        predicted_response_time = base_response_time * adjustment_factor
        predicted_throughput = base_throughput / adjustment_factor

        # Calculate confidence based on prediction consistency
        confidence = 0.8 - (load_factor * 0.2)  # Lower confidence at higher loads

        return {
            "response_time_ms": predicted_response_time,
            "throughput_rps": predicted_throughput,
            "confidence": confidence
        }

    async def train_models(self,
                          decision_history: List[LoadBalancingDecision],
                          performance_history: Dict[str, deque]):
        """Train ML models with historical data."""
        # Simplified training implementation
        # In production, would implement proper ML training pipeline

        for instance_id, metrics_history in performance_history.items():
            if len(metrics_history) < 10:
                continue

            # Extract patterns from metrics history
            recent_metrics = list(metrics_history)[-50:]  # Last 50 data points

            # Calculate performance trends
            response_times = [m.response_time_ms for m in recent_metrics]
            if len(response_times) > 1:
                trend = np.polyfit(range(len(response_times)), response_times, 1)[0]
                self.models[instance_id] = {"response_time_trend": trend}

        logger.debug(f"Trained ML models for {len(self.models)} instances")


class LoadPatternDetector:
    """Detects patterns in load balancing decisions for optimization."""

    def __init__(self):
        self.pattern_history: deque = deque(maxlen=10000)
        self.detected_patterns: Dict[str, Any] = {}

    def analyze_patterns(self, decision_history: List[LoadBalancingDecision]):
        """Analyze load balancing patterns."""
        if len(decision_history) < 50:
            return

        # Detect algorithm effectiveness
        algorithm_performance = defaultdict(list)

        for decision in decision_history[-100:]:  # Last 100 decisions
            algorithm_performance[decision.algorithm_used].append(decision.confidence)

        # Calculate average confidence by algorithm
        for algorithm, confidences in algorithm_performance.items():
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                self.detected_patterns[f"{algorithm}_avg_confidence"] = avg_confidence

        # Detect time-based patterns
        hourly_patterns = defaultdict(list)

        for decision in decision_history[-1000:]:  # Last 1000 decisions
            hour = decision.timestamp.hour
            hourly_patterns[hour].append(decision.selected_instance)

        # Store hourly distribution patterns
        for hour, instances in hourly_patterns.items():
            if instances:
                instance_counts = defaultdict(int)
                for instance in instances:
                    instance_counts[instance] += 1

                # Most popular instance for this hour
                popular_instance = max(instance_counts.items(), key=lambda x: x[1])
                self.detected_patterns[f"hour_{hour}_popular_instance"] = popular_instance[0]


# Factory function for creating intelligent load balancer
async def create_intelligent_load_balancer(
    algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.HYBRID_ADAPTIVE,
    instances: Optional[List[InstanceConfiguration]] = None
) -> IntelligentLoadBalancer:
    """Create and configure an intelligent load balancer."""

    lb = IntelligentLoadBalancer(
        algorithm=algorithm,
        enable_session_affinity=True,
        enable_health_monitoring=True,
        enable_ml_predictions=True
    )

    # Add instances if provided
    if instances:
        for instance_config in instances:
            lb.add_instance(instance_config)

    # Start background tasks
    await lb.start()

    return lb


# Global intelligent load balancer instance
_global_intelligent_lb: Optional[IntelligentLoadBalancer] = None


async def get_intelligent_load_balancer() -> IntelligentLoadBalancer:
    """Get global intelligent load balancer instance."""
    global _global_intelligent_lb

    if _global_intelligent_lb is None:
        _global_intelligent_lb = await create_intelligent_load_balancer()

    return _global_intelligent_lb


# Example usage and testing
if __name__ == "__main__":
    async def example_usage():
        """Example of using the intelligent load balancer."""

        # Create sample instances
        instances = [
            InstanceConfiguration(
                instance_id="instance_1",
                address="192.168.1.10",
                port=8080,
                weight=1.0,
                instance_type=InstanceType.COMPUTE,
                region="us-east",
                availability_zone="us-east-1a"
            ),
            InstanceConfiguration(
                instance_id="instance_2",
                address="192.168.1.11",
                port=8080,
                weight=1.5,
                instance_type=InstanceType.MEMORY,
                region="us-east",
                availability_zone="us-east-1b"
            ),
            InstanceConfiguration(
                instance_id="instance_3",
                address="192.168.1.12",
                port=8080,
                weight=0.8,
                instance_type=InstanceType.QUANTUM,
                region="us-west",
                availability_zone="us-west-1a",
                quantum_enabled=True
            )
        ]

        # Create load balancer
        lb = await create_intelligent_load_balancer(
            algorithm=LoadBalancingAlgorithm.HYBRID_ADAPTIVE,
            instances=instances
        )

        try:
            # Simulate load balancing decisions
            for i in range(20):
                request_context = {
                    "request_id": f"req_{i}",
                    "path": f"/api/endpoint_{i % 5}",
                    "method": "GET",
                    "size_bytes": np.random.randint(100, 10000)
                }

                decision = await lb.select_instance(
                    request_context=request_context,
                    session_id=f"session_{i % 3}",
                    client_ip=f"192.168.{i % 10}.{i % 100}",
                    requirements={"instance_type": "compute"} if i % 4 == 0 else None
                )

                print(f"Request {i}: {decision.selected_instance} "
                      f"(algorithm: {decision.algorithm_used}, "
                      f"confidence: {decision.confidence:.2f})")

                # Simulate some delay
                await asyncio.sleep(0.1)

            # Get statistics
            stats = lb.get_load_balancing_stats()
            print("\nLoad Balancer Statistics:")
            print(f"Total instances: {stats['total_instances']}")
            print(f"Healthy instances: {stats['healthy_instances']}")
            print(f"Total decisions: {stats['total_decisions']}")
            print(f"Active sessions: {stats['active_sessions']}")

        finally:
            await lb.stop()

    # Run example
    asyncio.run(example_usage())
    print("Intelligent load balancer example completed!")
