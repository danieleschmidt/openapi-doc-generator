"""
Advanced Resource Pool Optimization System

This module implements sophisticated resource pool management with intelligent
connection pooling, dynamic resource allocation, and advanced optimization
techniques for maximum efficiency and performance.

Features:
- Intelligent connection pooling with adaptive sizing
- Dynamic resource allocation and lifecycle management
- Advanced memory and CPU resource optimization
- Connection health monitoring and automatic recovery
- Load-aware resource distribution
- Resource usage analytics and optimization recommendations
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import psutil

from .enhanced_monitoring import get_monitor
from .performance_optimizer import get_optimizer

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources managed by the pool."""
    DATABASE_CONNECTION = "database_connection"
    HTTP_CONNECTION = "http_connection"
    THREAD = "thread"
    PROCESS = "process"
    MEMORY_BUFFER = "memory_buffer"
    FILE_HANDLE = "file_handle"
    NETWORK_SOCKET = "network_socket"
    CACHE_ENTRY = "cache_entry"


class ResourceState(Enum):
    """States of pooled resources."""
    AVAILABLE = "available"
    IN_USE = "in_use"
    VALIDATING = "validating"
    FAILED = "failed"
    EXPIRED = "expired"
    INITIALIZING = "initializing"


class PoolStrategy(Enum):
    """Resource pool management strategies."""
    FIXED_SIZE = "fixed_size"
    DYNAMIC = "dynamic"
    ELASTIC = "elastic"
    ADAPTIVE = "adaptive"
    PREDICTIVE = "predictive"


@dataclass
class ResourceMetrics:
    """Metrics for individual resources."""
    resource_id: str
    resource_type: ResourceType
    created_at: datetime
    last_used: datetime
    usage_count: int = 0
    failure_count: int = 0
    total_active_time: float = 0.0
    average_usage_duration: float = 0.0
    health_score: float = 1.0
    memory_usage_kb: float = 0.0

    def update_usage(self, duration: float, success: bool = True):
        """Update usage statistics."""
        self.usage_count += 1
        self.last_used = datetime.now()
        self.total_active_time += duration
        self.average_usage_duration = self.total_active_time / self.usage_count

        if not success:
            self.failure_count += 1
            self.health_score = max(0.0, self.health_score - 0.1)
        else:
            # Gradually improve health score on successful use
            self.health_score = min(1.0, self.health_score + 0.01)


@dataclass
class PoolConfiguration:
    """Configuration for resource pools."""
    resource_type: ResourceType
    min_size: int = 5
    max_size: int = 100
    initial_size: int = 10

    # Lifecycle settings
    max_age_seconds: float = 3600.0  # 1 hour
    max_idle_time_seconds: float = 300.0  # 5 minutes
    validation_interval_seconds: float = 60.0  # 1 minute

    # Scaling settings
    scale_up_threshold: float = 0.8  # 80% utilization
    scale_down_threshold: float = 0.3  # 30% utilization
    scale_up_increment: int = 2
    scale_down_increment: int = 1
    scaling_cooldown_seconds: float = 30.0

    # Health and monitoring
    health_check_enabled: bool = True
    health_check_timeout: float = 5.0
    max_failures_before_removal: int = 3

    # Strategy
    strategy: PoolStrategy = PoolStrategy.ADAPTIVE


@dataclass
class PooledResource:
    """A resource managed by the pool."""
    resource_id: str
    resource_type: ResourceType
    resource: Any
    state: ResourceState = ResourceState.INITIALIZING
    metrics: ResourceMetrics = field(init=False)
    created_at: datetime = field(default_factory=datetime.now)
    last_health_check: Optional[datetime] = None

    def __post_init__(self):
        self.metrics = ResourceMetrics(
            resource_id=self.resource_id,
            resource_type=self.resource_type,
            created_at=self.created_at,
            last_used=self.created_at
        )


class ResourceHealthChecker:
    """Health checker for pooled resources."""

    def __init__(self):
        self.health_check_functions: Dict[ResourceType, Callable] = {}
        self.default_timeout = 5.0

    def register_health_check(self,
                            resource_type: ResourceType,
                            check_function: Callable[[Any], bool]):
        """Register a health check function for a resource type."""
        self.health_check_functions[resource_type] = check_function
        logger.debug(f"Registered health check for {resource_type.value}")

    async def check_health(self,
                          pooled_resource: PooledResource,
                          timeout: Optional[float] = None) -> bool:
        """Check the health of a pooled resource."""
        if pooled_resource.resource_type not in self.health_check_functions:
            return True  # Assume healthy if no check function

        check_function = self.health_check_functions[pooled_resource.resource_type]
        timeout = timeout or self.default_timeout

        try:
            # Run health check with timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(check_function, pooled_resource.resource),
                timeout=timeout
            )

            pooled_resource.last_health_check = datetime.now()

            if result:
                pooled_resource.metrics.health_score = min(1.0, pooled_resource.metrics.health_score + 0.05)
            else:
                pooled_resource.metrics.health_score = max(0.0, pooled_resource.metrics.health_score - 0.2)

            return result

        except asyncio.TimeoutError:
            logger.warning(f"Health check timeout for resource {pooled_resource.resource_id}")
            pooled_resource.metrics.health_score = max(0.0, pooled_resource.metrics.health_score - 0.3)
            return False
        except Exception as e:
            logger.error(f"Health check error for resource {pooled_resource.resource_id}: {e}")
            pooled_resource.metrics.health_score = max(0.0, pooled_resource.metrics.health_score - 0.2)
            return False


class ResourceFactory:
    """Factory for creating pooled resources."""

    def __init__(self):
        self.factory_functions: Dict[ResourceType, Callable] = {}
        self.cleanup_functions: Dict[ResourceType, Callable] = {}

    def register_factory(self,
                        resource_type: ResourceType,
                        factory_function: Callable[[], Any],
                        cleanup_function: Optional[Callable[[Any], None]] = None):
        """Register factory and cleanup functions for a resource type."""
        self.factory_functions[resource_type] = factory_function
        if cleanup_function:
            self.cleanup_functions[resource_type] = cleanup_function

        logger.debug(f"Registered factory for {resource_type.value}")

    async def create_resource(self,
                            resource_type: ResourceType,
                            resource_id: str) -> Optional[PooledResource]:
        """Create a new pooled resource."""
        if resource_type not in self.factory_functions:
            logger.error(f"No factory registered for {resource_type.value}")
            return None

        try:
            factory_function = self.factory_functions[resource_type]

            # Create resource with timeout
            resource = await asyncio.wait_for(
                asyncio.to_thread(factory_function),
                timeout=10.0
            )

            pooled_resource = PooledResource(
                resource_id=resource_id,
                resource_type=resource_type,
                resource=resource,
                state=ResourceState.AVAILABLE
            )

            logger.debug(f"Created resource {resource_id} of type {resource_type.value}")
            return pooled_resource

        except asyncio.TimeoutError:
            logger.error(f"Resource creation timeout for {resource_id}")
            return None
        except Exception as e:
            logger.error(f"Resource creation failed for {resource_id}: {e}")
            return None

    async def cleanup_resource(self, pooled_resource: PooledResource):
        """Clean up a pooled resource."""
        if pooled_resource.resource_type in self.cleanup_functions:
            try:
                cleanup_function = self.cleanup_functions[pooled_resource.resource_type]
                await asyncio.to_thread(cleanup_function, pooled_resource.resource)
                logger.debug(f"Cleaned up resource {pooled_resource.resource_id}")
            except Exception as e:
                logger.warning(f"Cleanup failed for resource {pooled_resource.resource_id}: {e}")


class IntelligentResourcePool:
    """
    Intelligent resource pool with adaptive sizing and advanced optimization.
    """

    def __init__(self,
                 config: PoolConfiguration,
                 resource_factory: ResourceFactory,
                 health_checker: ResourceHealthChecker):

        self.config = config
        self.resource_factory = resource_factory
        self.health_checker = health_checker

        # Resource management
        self.resources: Dict[str, PooledResource] = {}
        self.available_resources: deque = deque()
        self.in_use_resources: Set[str] = set()

        # Pool state
        self.current_size = 0
        self.target_size = config.initial_size
        self.last_scaling_action = datetime.now()

        # Statistics
        self.stats = {
            'total_created': 0,
            'total_destroyed': 0,
            'total_requests': 0,
            'total_hits': 0,
            'total_misses': 0,
            'total_failures': 0,
            'average_wait_time': 0.0,
            'current_utilization': 0.0
        }

        # Performance tracking
        self.usage_history: deque = deque(maxlen=1000)
        self.wait_times: deque = deque(maxlen=100)
        self.utilization_history: deque = deque(maxlen=100)

        # Background tasks
        self.maintenance_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        self.running = False

        # Optimization
        self.monitor = get_monitor()
        self.optimizer = get_optimizer()

        # Thread safety
        self.lock = asyncio.Lock()

        logger.info(f"Resource pool initialized: {config.resource_type.value}, "
                   f"size {config.min_size}-{config.max_size}")

    async def start(self):
        """Start the resource pool and background tasks."""
        if self.running:
            return

        self.running = True

        # Initialize pool with initial resources
        await self._initialize_pool()

        # Start background tasks
        self.maintenance_task = asyncio.create_task(self._maintenance_loop())

        if self.config.health_check_enabled:
            self.health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info(f"Resource pool started: {self.config.resource_type.value}")

    async def stop(self):
        """Stop the resource pool and cleanup all resources."""
        if not self.running:
            return

        self.running = False

        # Cancel background tasks
        if self.maintenance_task and not self.maintenance_task.done():
            self.maintenance_task.cancel()

        if self.health_check_task and not self.health_check_task.done():
            self.health_check_task.cancel()

        # Wait for tasks to complete
        tasks = [self.maintenance_task, self.health_check_task]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Cleanup all resources
        await self._cleanup_all_resources()

        logger.info(f"Resource pool stopped: {self.config.resource_type.value}")

    async def acquire_resource(self, timeout: Optional[float] = 10.0) -> Optional[Any]:
        """Acquire a resource from the pool."""
        async with self.lock:
            start_time = time.time()
            self.stats['total_requests'] += 1

            try:
                # Try to get an available resource
                pooled_resource = await self._get_available_resource(timeout)

                if pooled_resource:
                    # Mark as in use
                    pooled_resource.state = ResourceState.IN_USE
                    self.in_use_resources.add(pooled_resource.resource_id)

                    # Update statistics
                    self.stats['total_hits'] += 1
                    wait_time = time.time() - start_time
                    self.wait_times.append(wait_time)
                    self.stats['average_wait_time'] = sum(self.wait_times) / len(self.wait_times)

                    # Update utilization
                    self._update_utilization()

                    # Record usage start
                    usage_start = time.time()
                    pooled_resource.metrics.last_used = datetime.now()

                    # Create usage context for tracking
                    usage_context = {
                        'resource_id': pooled_resource.resource_id,
                        'start_time': usage_start,
                        'resource': pooled_resource
                    }

                    # Return a wrapped resource that tracks usage
                    return ResourceWrapper(pooled_resource.resource, usage_context, self)

                else:
                    # No resource available
                    self.stats['total_misses'] += 1
                    logger.warning(f"Failed to acquire resource from pool {self.config.resource_type.value}")
                    return None

            except Exception as e:
                self.stats['total_failures'] += 1
                logger.error(f"Error acquiring resource: {e}")
                return None

    async def release_resource(self, resource_wrapper: 'ResourceWrapper'):
        """Release a resource back to the pool."""
        async with self.lock:
            try:
                usage_context = resource_wrapper.usage_context
                pooled_resource = usage_context['resource']

                # Calculate usage duration
                usage_duration = time.time() - usage_context['start_time']

                # Update resource metrics
                pooled_resource.metrics.update_usage(usage_duration, success=True)

                # Mark as available
                pooled_resource.state = ResourceState.AVAILABLE
                self.in_use_resources.discard(pooled_resource.resource_id)
                self.available_resources.append(pooled_resource.resource_id)

                # Update utilization
                self._update_utilization()

                # Record usage history
                self.usage_history.append({
                    'timestamp': datetime.now(),
                    'resource_id': pooled_resource.resource_id,
                    'duration': usage_duration,
                    'success': True
                })

                # Record metrics
                self.monitor.record_metric(
                    f"resource_pool_{self.config.resource_type.value}_usage_duration",
                    usage_duration, "timer"
                )

                logger.debug(f"Released resource {pooled_resource.resource_id} after {usage_duration:.3f}s")

            except Exception as e:
                logger.error(f"Error releasing resource: {e}")

    async def _get_available_resource(self, timeout: Optional[float]) -> Optional[PooledResource]:
        """Get an available resource, creating one if needed."""
        deadline = time.time() + (timeout or 10.0)

        while time.time() < deadline:
            # Try to get from available resources
            if self.available_resources:
                resource_id = self.available_resources.popleft()
                if resource_id in self.resources:
                    pooled_resource = self.resources[resource_id]

                    # Validate resource
                    if await self._validate_resource(pooled_resource):
                        return pooled_resource
                    else:
                        # Resource failed validation, remove it
                        await self._remove_resource(resource_id)

            # Try to create new resource if under limit
            if self.current_size < self.config.max_size:
                pooled_resource = await self._create_resource()
                if pooled_resource:
                    return pooled_resource

            # Wait a bit before trying again
            await asyncio.sleep(0.1)

        return None

    async def _validate_resource(self, pooled_resource: PooledResource) -> bool:
        """Validate that a resource is still usable."""
        # Check age
        age = (datetime.now() - pooled_resource.created_at).total_seconds()
        if age > self.config.max_age_seconds:
            logger.debug(f"Resource {pooled_resource.resource_id} expired due to age")
            return False

        # Check idle time
        idle_time = (datetime.now() - pooled_resource.metrics.last_used).total_seconds()
        if idle_time > self.config.max_idle_time_seconds:
            logger.debug(f"Resource {pooled_resource.resource_id} expired due to idle time")
            return False

        # Check failure count
        if pooled_resource.metrics.failure_count >= self.config.max_failures_before_removal:
            logger.debug(f"Resource {pooled_resource.resource_id} removed due to failures")
            return False

        # Check health score
        if pooled_resource.metrics.health_score < 0.3:
            logger.debug(f"Resource {pooled_resource.resource_id} removed due to low health score")
            return False

        return True

    async def _create_resource(self) -> Optional[PooledResource]:
        """Create a new resource."""
        resource_id = f"{self.config.resource_type.value}_{self.stats['total_created'] + 1}_{time.time()}"

        pooled_resource = await self.resource_factory.create_resource(
            self.config.resource_type, resource_id
        )

        if pooled_resource:
            self.resources[resource_id] = pooled_resource
            self.current_size += 1
            self.stats['total_created'] += 1

            logger.debug(f"Created new resource: {resource_id}")
            return pooled_resource

        return None

    async def _remove_resource(self, resource_id: str):
        """Remove a resource from the pool."""
        if resource_id in self.resources:
            pooled_resource = self.resources[resource_id]

            # Cleanup the resource
            await self.resource_factory.cleanup_resource(pooled_resource)

            # Remove from tracking
            del self.resources[resource_id]
            self.in_use_resources.discard(resource_id)

            # Remove from available queue if present
            try:
                self.available_resources.remove(resource_id)
            except ValueError:
                pass  # Not in queue

            self.current_size -= 1
            self.stats['total_destroyed'] += 1

            logger.debug(f"Removed resource: {resource_id}")

    async def _initialize_pool(self):
        """Initialize the pool with initial resources."""
        for _ in range(self.config.initial_size):
            pooled_resource = await self._create_resource()
            if pooled_resource:
                self.available_resources.append(pooled_resource.resource_id)

        self.target_size = self.config.initial_size
        logger.info(f"Initialized pool with {self.current_size} resources")

    def _update_utilization(self):
        """Update current utilization statistics."""
        if self.current_size > 0:
            utilization = len(self.in_use_resources) / self.current_size
            self.stats['current_utilization'] = utilization
            self.utilization_history.append(utilization)
        else:
            self.stats['current_utilization'] = 0.0

    async def _maintenance_loop(self):
        """Background maintenance loop."""
        while self.running:
            try:
                await self._perform_maintenance()
                await asyncio.sleep(30)  # Run maintenance every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Maintenance loop error: {e}")
                await asyncio.sleep(60)

    async def _perform_maintenance(self):
        """Perform pool maintenance tasks."""
        async with self.lock:
            # Remove expired resources
            expired_resources = []
            for resource_id, pooled_resource in self.resources.items():
                if not await self._validate_resource(pooled_resource):
                    expired_resources.append(resource_id)

            for resource_id in expired_resources:
                await self._remove_resource(resource_id)

            # Adjust pool size based on strategy
            await self._adjust_pool_size()

            # Update statistics
            await self._update_statistics()

    async def _adjust_pool_size(self):
        """Adjust pool size based on configuration strategy."""
        if self.config.strategy == PoolStrategy.FIXED_SIZE:
            # Maintain fixed size
            while self.current_size < self.config.initial_size:
                pooled_resource = await self._create_resource()
                if pooled_resource:
                    self.available_resources.append(pooled_resource.resource_id)
                else:
                    break

        elif self.config.strategy == PoolStrategy.DYNAMIC:
            # Dynamic sizing based on utilization
            await self._dynamic_sizing()

        elif self.config.strategy == PoolStrategy.ADAPTIVE:
            # Adaptive sizing with learning
            await self._adaptive_sizing()

        elif self.config.strategy == PoolStrategy.PREDICTIVE:
            # Predictive sizing based on patterns
            await self._predictive_sizing()

    async def _dynamic_sizing(self):
        """Dynamic pool sizing based on current utilization."""
        current_utilization = self.stats['current_utilization']

        # Check if we should scale up
        if current_utilization > self.config.scale_up_threshold:
            if self.current_size < self.config.max_size:
                # Check cooldown
                time_since_scaling = (datetime.now() - self.last_scaling_action).total_seconds()
                if time_since_scaling > self.config.scaling_cooldown_seconds:

                    # Scale up
                    for _ in range(min(self.config.scale_up_increment,
                                     self.config.max_size - self.current_size)):
                        pooled_resource = await self._create_resource()
                        if pooled_resource:
                            self.available_resources.append(pooled_resource.resource_id)
                        else:
                            break

                    self.last_scaling_action = datetime.now()
                    logger.info(f"Scaled up pool to {self.current_size} resources "
                               f"(utilization: {current_utilization:.2f})")

        # Check if we should scale down
        elif current_utilization < self.config.scale_down_threshold:
            if self.current_size > self.config.min_size:
                # Check cooldown
                time_since_scaling = (datetime.now() - self.last_scaling_action).total_seconds()
                if time_since_scaling > self.config.scaling_cooldown_seconds:

                    # Scale down by removing idle resources
                    resources_to_remove = min(self.config.scale_down_increment,
                                            self.current_size - self.config.min_size,
                                            len(self.available_resources))

                    for _ in range(resources_to_remove):
                        if self.available_resources:
                            resource_id = self.available_resources.popleft()
                            await self._remove_resource(resource_id)

                    self.last_scaling_action = datetime.now()
                    logger.info(f"Scaled down pool to {self.current_size} resources "
                               f"(utilization: {current_utilization:.2f})")

    async def _adaptive_sizing(self):
        """Adaptive sizing with learning from usage patterns."""
        # Analyze usage patterns
        if len(self.utilization_history) >= 10:
            avg_utilization = sum(self.utilization_history) / len(self.utilization_history)
            utilization_trend = self._calculate_trend(list(self.utilization_history))

            # Predict future utilization
            predicted_utilization = avg_utilization + utilization_trend * 5  # 5 periods ahead

            # Adjust target size based on prediction
            if predicted_utilization > 0.8:
                self.target_size = min(self.config.max_size,
                                     int(self.current_size * 1.2))
            elif predicted_utilization < 0.3:
                self.target_size = max(self.config.min_size,
                                     int(self.current_size * 0.8))

            # Gradually adjust to target size
            if self.current_size < self.target_size:
                pooled_resource = await self._create_resource()
                if pooled_resource:
                    self.available_resources.append(pooled_resource.resource_id)
            elif self.current_size > self.target_size and self.available_resources:
                resource_id = self.available_resources.popleft()
                await self._remove_resource(resource_id)

    async def _predictive_sizing(self):
        """Predictive sizing based on time patterns."""
        # Analyze hourly patterns
        current_hour = datetime.now().hour

        # Simple predictive model based on hour of day
        if 9 <= current_hour <= 17:  # Business hours
            recommended_size = int(self.config.max_size * 0.8)
        elif 18 <= current_hour <= 22:  # Evening hours
            recommended_size = int(self.config.max_size * 0.5)
        else:  # Night hours
            recommended_size = int(self.config.max_size * 0.3)

        recommended_size = max(self.config.min_size,
                              min(self.config.max_size, recommended_size))

        # Gradually adjust towards recommended size
        if self.current_size < recommended_size:
            pooled_resource = await self._create_resource()
            if pooled_resource:
                self.available_resources.append(pooled_resource.resource_id)
        elif self.current_size > recommended_size and self.available_resources:
            resource_id = self.available_resources.popleft()
            await self._remove_resource(resource_id)

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend from a list of values."""
        if len(values) < 2:
            return 0.0

        # Simple linear regression
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x_squared_sum = sum(i * i for i in range(n))

        denominator = n * x_squared_sum - x_sum * x_sum
        if denominator == 0:
            return 0.0

        slope = (n * xy_sum - x_sum * y_sum) / denominator
        return slope

    async def _update_statistics(self):
        """Update pool statistics."""
        # Record pool metrics
        self.monitor.record_metric(
            f"resource_pool_{self.config.resource_type.value}_size",
            float(self.current_size), "gauge"
        )

        self.monitor.record_metric(
            f"resource_pool_{self.config.resource_type.value}_utilization",
            self.stats['current_utilization'], "gauge"
        )

        self.monitor.record_metric(
            f"resource_pool_{self.config.resource_type.value}_wait_time",
            self.stats['average_wait_time'], "gauge"
        )

    async def _health_check_loop(self):
        """Background health check loop."""
        while self.running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.config.validation_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(self.config.validation_interval_seconds * 2)

    async def _perform_health_checks(self):
        """Perform health checks on pool resources."""
        async with self.lock:
            unhealthy_resources = []

            for resource_id, pooled_resource in self.resources.items():
                # Skip resources currently in use
                if resource_id in self.in_use_resources:
                    continue

                # Skip recently checked resources
                if (pooled_resource.last_health_check and
                    (datetime.now() - pooled_resource.last_health_check).total_seconds() <
                    self.config.validation_interval_seconds / 2):
                    continue

                # Perform health check
                is_healthy = await self.health_checker.check_health(
                    pooled_resource, self.config.health_check_timeout
                )

                if not is_healthy:
                    unhealthy_resources.append(resource_id)

            # Remove unhealthy resources
            for resource_id in unhealthy_resources:
                await self._remove_resource(resource_id)
                logger.warning(f"Removed unhealthy resource: {resource_id}")

    async def _cleanup_all_resources(self):
        """Cleanup all resources in the pool."""
        async with self.lock:
            resource_ids = list(self.resources.keys())
            for resource_id in resource_ids:
                await self._remove_resource(resource_id)

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics."""
        return {
            'resource_type': self.config.resource_type.value,
            'strategy': self.config.strategy.value,
            'current_size': self.current_size,
            'target_size': self.target_size,
            'min_size': self.config.min_size,
            'max_size': self.config.max_size,
            'available_resources': len(self.available_resources),
            'in_use_resources': len(self.in_use_resources),
            'current_utilization': self.stats['current_utilization'],
            'average_wait_time': self.stats['average_wait_time'],
            'total_requests': self.stats['total_requests'],
            'hit_rate': (self.stats['total_hits'] / max(self.stats['total_requests'], 1)) * 100,
            'total_created': self.stats['total_created'],
            'total_destroyed': self.stats['total_destroyed'],
            'resource_health_scores': {
                resource_id: resource.metrics.health_score
                for resource_id, resource in self.resources.items()
            },
            'recent_utilization_trend': list(self.utilization_history)[-10:],
        }


class ResourceWrapper:
    """Wrapper for pooled resources that tracks usage."""

    def __init__(self, resource: Any, usage_context: Dict[str, Any], pool: IntelligentResourcePool):
        self._resource = resource
        self.usage_context = usage_context
        self._pool = pool
        self._released = False

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped resource."""
        return getattr(self._resource, name)

    async def __aenter__(self):
        """Async context manager entry."""
        return self._resource

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if not self._released:
            await self._pool.release_resource(self)
            self._released = True

    def __enter__(self):
        """Context manager entry."""
        return self._resource

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if not self._released:
            asyncio.create_task(self._pool.release_resource(self))
            self._released = True

    async def release(self):
        """Manually release the resource."""
        if not self._released:
            await self._pool.release_resource(self)
            self._released = True


class ResourcePoolManager:
    """
    Manager for multiple resource pools with optimization and coordination.
    """

    def __init__(self):
        self.pools: Dict[ResourceType, IntelligentResourcePool] = {}
        self.resource_factory = ResourceFactory()
        self.health_checker = ResourceHealthChecker()

        # Global optimization
        self.monitor = get_monitor()
        self.optimizer = get_optimizer()

        # Background tasks
        self.optimization_task: Optional[asyncio.Task] = None
        self.running = False

        logger.info("Resource Pool Manager initialized")

    async def start(self):
        """Start all resource pools and management tasks."""
        if self.running:
            return

        self.running = True

        # Start all pools
        for pool in self.pools.values():
            await pool.start()

        # Start global optimization
        self.optimization_task = asyncio.create_task(self._global_optimization_loop())

        logger.info("Resource Pool Manager started")

    async def stop(self):
        """Stop all resource pools and cleanup."""
        if not self.running:
            return

        self.running = False

        # Cancel optimization task
        if self.optimization_task and not self.optimization_task.done():
            self.optimization_task.cancel()
            await asyncio.gather(self.optimization_task, return_exceptions=True)

        # Stop all pools
        for pool in self.pools.values():
            await pool.stop()

        logger.info("Resource Pool Manager stopped")

    def create_pool(self, config: PoolConfiguration) -> IntelligentResourcePool:
        """Create a new resource pool."""
        pool = IntelligentResourcePool(config, self.resource_factory, self.health_checker)
        self.pools[config.resource_type] = pool

        logger.info(f"Created resource pool: {config.resource_type.value}")
        return pool

    async def get_resource(self, resource_type: ResourceType, timeout: Optional[float] = 10.0) -> Optional[Any]:
        """Get a resource from the specified pool."""
        if resource_type not in self.pools:
            logger.error(f"No pool configured for resource type: {resource_type.value}")
            return None

        pool = self.pools[resource_type]
        return await pool.acquire_resource(timeout)

    def register_resource_factory(self,
                                resource_type: ResourceType,
                                factory_function: Callable[[], Any],
                                cleanup_function: Optional[Callable[[Any], None]] = None):
        """Register factory and cleanup functions for a resource type."""
        self.resource_factory.register_factory(resource_type, factory_function, cleanup_function)

    def register_health_check(self,
                            resource_type: ResourceType,
                            check_function: Callable[[Any], bool]):
        """Register health check function for a resource type."""
        self.health_checker.register_health_check(resource_type, check_function)

    async def _global_optimization_loop(self):
        """Global optimization loop across all pools."""
        while self.running:
            try:
                await self._perform_global_optimization()
                await asyncio.sleep(60)  # Run optimization every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Global optimization error: {e}")
                await asyncio.sleep(120)

    async def _perform_global_optimization(self):
        """Perform global optimization across all pools."""
        # Analyze resource usage patterns
        total_utilization = 0.0
        pool_count = 0

        optimization_recommendations = []

        for resource_type, pool in self.pools.items():
            stats = pool.get_pool_stats()
            utilization = stats['current_utilization']
            total_utilization += utilization
            pool_count += 1

            # Generate recommendations
            if utilization > 0.9:
                optimization_recommendations.append({
                    'pool': resource_type.value,
                    'action': 'increase_max_size',
                    'reason': f'High utilization: {utilization:.2f}',
                    'priority': 'high'
                })
            elif utilization < 0.1 and stats['current_size'] > stats['min_size']:
                optimization_recommendations.append({
                    'pool': resource_type.value,
                    'action': 'decrease_size',
                    'reason': f'Low utilization: {utilization:.2f}',
                    'priority': 'low'
                })

        # Calculate system-wide metrics
        avg_utilization = total_utilization / max(pool_count, 1)

        # Record global metrics
        self.monitor.record_metric("resource_pool_global_utilization", avg_utilization, "gauge")
        self.monitor.record_metric("resource_pool_count", float(pool_count), "gauge")

        # Log optimization recommendations
        if optimization_recommendations:
            logger.info(f"Resource pool optimization recommendations: {optimization_recommendations}")

    def get_global_stats(self) -> Dict[str, Any]:
        """Get global statistics across all pools."""
        global_stats = {
            'total_pools': len(self.pools),
            'pools': {},
            'global_metrics': {}
        }

        # Collect stats from all pools
        total_resources = 0
        total_utilization = 0.0
        total_requests = 0

        for resource_type, pool in self.pools.items():
            pool_stats = pool.get_pool_stats()
            global_stats['pools'][resource_type.value] = pool_stats

            total_resources += pool_stats['current_size']
            total_utilization += pool_stats['current_utilization']
            total_requests += pool_stats['total_requests']

        # Calculate global metrics
        avg_utilization = total_utilization / max(len(self.pools), 1)

        global_stats['global_metrics'] = {
            'total_resources': total_resources,
            'average_utilization': avg_utilization,
            'total_requests': total_requests,
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024
        }

        return global_stats


# Global resource pool manager
_global_pool_manager: Optional[ResourcePoolManager] = None


def get_resource_pool_manager() -> ResourcePoolManager:
    """Get global resource pool manager instance."""
    global _global_pool_manager
    if _global_pool_manager is None:
        _global_pool_manager = ResourcePoolManager()
    return _global_pool_manager


# Example usage and factory functions
async def example_usage():
    """Example of using the resource pool optimization system."""

    # Get pool manager
    manager = get_resource_pool_manager()

    # Register a simple resource factory (database connections)
    def create_db_connection():
        # Simulate database connection creation
        return f"db_connection_{time.time()}"

    def cleanup_db_connection(conn):
        # Simulate connection cleanup
        print(f"Cleaning up {conn}")

    def check_db_connection(conn):
        # Simulate health check
        return True

    manager.register_resource_factory(
        ResourceType.DATABASE_CONNECTION,
        create_db_connection,
        cleanup_db_connection
    )

    manager.register_health_check(
        ResourceType.DATABASE_CONNECTION,
        check_db_connection
    )

    # Create pool configuration
    config = PoolConfiguration(
        resource_type=ResourceType.DATABASE_CONNECTION,
        min_size=5,
        max_size=20,
        initial_size=8,
        strategy=PoolStrategy.ADAPTIVE
    )

    # Create pool
    db_pool = manager.create_pool(config)

    # Start manager
    await manager.start()

    try:
        # Simulate resource usage
        for i in range(50):
            resource = await manager.get_resource(ResourceType.DATABASE_CONNECTION)
            if resource:
                async with resource as conn:
                    # Simulate work
                    await asyncio.sleep(0.1)
                    print(f"Used connection: {conn}")

            if i % 10 == 0:
                stats = manager.get_global_stats()
                pool_stats = stats['pools']['database_connection']
                print(f"Pool stats: size={pool_stats['current_size']}, "
                      f"utilization={pool_stats['current_utilization']:.2f}")

        # Get final statistics
        final_stats = manager.get_global_stats()
        print(f"Final global stats: {final_stats['global_metrics']}")

    finally:
        await manager.stop()


if __name__ == "__main__":
    asyncio.run(example_usage())
    print("Resource pool optimization example completed!")
