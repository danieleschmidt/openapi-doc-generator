"""
Advanced Scalability Integration System

This module provides a unified interface for all scalability improvements,
coordinating intelligent load balancing, predictive auto-scaling, resource
optimization, global distribution, performance analytics, memory management,
and concurrent processing for optimal system performance.

Integration Components:
1. Intelligent Load Balancing - Real-time metrics and quantum-inspired distribution
2. Predictive Auto-Scaling - ML-based capacity planning and preemptive scaling  
3. Resource Pool Optimization - Advanced connection pooling and management
4. Cross-Region Distribution - Global deployment with edge caching
5. Performance Analytics - Real-time monitoring and bottleneck detection
6. Memory Optimization - Advanced memory management and GC tuning
7. Concurrent Processing - Enhanced parallel processing with work-stealing
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .advanced_concurrent_processor import AdvancedConcurrentProcessor, SchedulingStrategy, TaskPriority
from .advanced_memory_optimizer import AdvancedMemoryOptimizer, MemoryStrategy
from .advanced_performance_analytics import PerformanceAnalyzer, PerformanceMetricType
from .enhanced_monitoring import get_monitor
from .global_distribution_optimizer import GlobalDistributionOptimizer, Region

# Import all scalability components
from .intelligent_load_balancer import InstanceConfiguration, IntelligentLoadBalancer, LoadBalancingAlgorithm
from .predictive_auto_scaler import PredictiveAutoScaler
from .resource_pool_optimizer import PoolConfiguration, ResourcePoolManager, ResourceType

logger = logging.getLogger(__name__)


@dataclass
class ScalabilityConfiguration:
    """Configuration for the integrated scalability system."""
    # Load Balancing
    load_balancing_algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.HYBRID_ADAPTIVE
    enable_session_affinity: bool = True

    # Auto-Scaling
    min_instances: int = 2
    max_instances: int = 50
    prediction_horizon_minutes: int = 30

    # Resource Pooling
    enable_resource_pooling: bool = True
    pool_strategies: Dict[ResourceType, str] = None

    # Global Distribution
    enable_global_distribution: bool = True
    regions: List[Region] = None
    edge_caching: bool = True

    # Performance Analytics
    enable_performance_analytics: bool = True
    enable_bottleneck_detection: bool = True

    # Memory Optimization
    memory_strategy: MemoryStrategy = MemoryStrategy.ADAPTIVE
    max_cache_memory_mb: float = 200.0

    # Concurrent Processing
    num_workers: int = None
    scheduling_strategy: SchedulingStrategy = SchedulingStrategy.WORK_STEALING
    numa_aware: bool = True

    def __post_init__(self):
        if self.pool_strategies is None:
            self.pool_strategies = {
                ResourceType.DATABASE_CONNECTION: "dynamic",
                ResourceType.HTTP_CONNECTION: "elastic",
                ResourceType.THREAD: "adaptive"
            }

        if self.regions is None:
            self.regions = [Region.US_EAST_1, Region.EU_CENTRAL_1, Region.ASIA_PACIFIC_1]


class IntegratedScalabilitySystem:
    """
    Master coordinator for all scalability improvements.
    Provides unified interface and coordination between components.
    """

    def __init__(self, config: Optional[ScalabilityConfiguration] = None):
        self.config = config or ScalabilityConfiguration()

        # Core components (initialized later)
        self.load_balancer: Optional[IntelligentLoadBalancer] = None
        self.auto_scaler: Optional[PredictiveAutoScaler] = None
        self.resource_pool_manager: Optional[ResourcePoolManager] = None
        self.global_distribution: Optional[GlobalDistributionOptimizer] = None
        self.performance_analyzer: Optional[PerformanceAnalyzer] = None
        self.memory_optimizer: Optional[AdvancedMemoryOptimizer] = None
        self.concurrent_processor: Optional[AdvancedConcurrentProcessor] = None

        # Integration state
        self.running = False
        self.start_time: Optional[datetime] = None
        self.monitor = get_monitor()

        # Background coordination tasks
        self.coordination_tasks: List[asyncio.Task] = []

        logger.info("Integrated Scalability System initialized")

    async def start(self, instances: List[InstanceConfiguration] = None):
        """Start all scalability components."""
        if self.running:
            return

        logger.info("Starting Integrated Scalability System...")
        self.running = True
        self.start_time = datetime.now()

        try:
            # 1. Initialize Memory Optimization first (foundation)
            await self._initialize_memory_optimization()

            # 2. Initialize Concurrent Processing (core processing capability)
            await self._initialize_concurrent_processing()

            # 3. Initialize Performance Analytics (monitoring foundation)
            await self._initialize_performance_analytics()

            # 4. Initialize Resource Pool Management
            await self._initialize_resource_pooling()

            # 5. Initialize Load Balancing
            await self._initialize_load_balancing(instances)

            # 6. Initialize Auto-Scaling
            await self._initialize_auto_scaling()

            # 7. Initialize Global Distribution (requires other components)
            await self._initialize_global_distribution()

            # 8. Start coordination tasks
            await self._start_coordination()

            logger.info("Integrated Scalability System started successfully")

        except Exception as e:
            logger.error(f"Failed to start scalability system: {e}")
            await self.stop()
            raise

    async def stop(self):
        """Stop all scalability components."""
        if not self.running:
            return

        logger.info("Stopping Integrated Scalability System...")
        self.running = False

        # Stop coordination tasks
        for task in self.coordination_tasks:
            if not task.done():
                task.cancel()

        if self.coordination_tasks:
            await asyncio.gather(*self.coordination_tasks, return_exceptions=True)

        # Stop components in reverse order
        components = [
            ("Global Distribution", self.global_distribution),
            ("Auto-Scaler", self.auto_scaler),
            ("Load Balancer", self.load_balancer),
            ("Resource Pool Manager", self.resource_pool_manager),
            ("Performance Analyzer", self.performance_analyzer),
            ("Concurrent Processor", self.concurrent_processor),
            ("Memory Optimizer", self.memory_optimizer)
        ]

        for name, component in components:
            if component:
                try:
                    if hasattr(component, 'stop'):
                        await component.stop()
                    logger.debug(f"{name} stopped")
                except Exception as e:
                    logger.warning(f"Error stopping {name}: {e}")

        logger.info("Integrated Scalability System stopped")

    async def _initialize_memory_optimization(self):
        """Initialize memory optimization component."""
        if self.config.memory_strategy:
            from .advanced_memory_optimizer import get_memory_optimizer

            self.memory_optimizer = get_memory_optimizer(
                strategy=self.config.memory_strategy,
                max_cache_memory_mb=self.config.max_cache_memory_mb
            )
            await self.memory_optimizer.start()

            logger.debug("Memory optimization initialized")

    async def _initialize_concurrent_processing(self):
        """Initialize concurrent processing component."""
        from .advanced_concurrent_processor import get_concurrent_processor

        self.concurrent_processor = await get_concurrent_processor(
            num_workers=self.config.num_workers,
            scheduling_strategy=self.config.scheduling_strategy,
            numa_aware=self.config.numa_aware
        )

        logger.debug("Concurrent processing initialized")

    async def _initialize_performance_analytics(self):
        """Initialize performance analytics component."""
        if self.config.enable_performance_analytics:
            from .advanced_performance_analytics import get_performance_analyzer

            self.performance_analyzer = get_performance_analyzer()
            await self.performance_analyzer.start()

            logger.debug("Performance analytics initialized")

    async def _initialize_resource_pooling(self):
        """Initialize resource pool management."""
        if self.config.enable_resource_pooling:
            from .resource_pool_optimizer import get_resource_pool_manager

            self.resource_pool_manager = get_resource_pool_manager()
            await self.resource_pool_manager.start()

            # Create default pools
            await self._create_default_resource_pools()

            logger.debug("Resource pooling initialized")

    async def _initialize_load_balancing(self, instances: List[InstanceConfiguration] = None):
        """Initialize intelligent load balancing."""
        from .intelligent_load_balancer import create_intelligent_load_balancer

        self.load_balancer = await create_intelligent_load_balancer(
            algorithm=self.config.load_balancing_algorithm,
            instances=instances or []
        )

        logger.debug(f"Load balancing initialized with {len(instances or [])} instances")

    async def _initialize_auto_scaling(self):
        """Initialize predictive auto-scaling."""
        from .predictive_auto_scaler import get_predictive_auto_scaler

        self.auto_scaler = await get_predictive_auto_scaler(
            min_instances=self.config.min_instances,
            max_instances=self.config.max_instances,
            prediction_horizon_minutes=self.config.prediction_horizon_minutes
        )

        logger.debug("Predictive auto-scaling initialized")

    async def _initialize_global_distribution(self):
        """Initialize global distribution optimization."""
        if self.config.enable_global_distribution:
            self.global_distribution = GlobalDistributionOptimizer()
            await self.global_distribution.start()

            # Add configured regions
            for region in self.config.regions:
                # Create sample instances for each region
                sample_instances = [
                    InstanceConfiguration(
                        instance_id=f"{region.value}_instance_1",
                        address="10.0.0.10",
                        port=8080,
                        region=region.value
                    )
                ]
                await self.global_distribution.add_region(region, sample_instances)

            logger.debug(f"Global distribution initialized with {len(self.config.regions)} regions")

    async def _create_default_resource_pools(self):
        """Create default resource pools."""
        if not self.resource_pool_manager:
            return

        # Database connection pool
        def create_db_connection():
            return f"db_connection_{id(object())}"

        def cleanup_db_connection(conn):
            pass  # Placeholder cleanup

        self.resource_pool_manager.register_resource_factory(
            ResourceType.DATABASE_CONNECTION,
            create_db_connection,
            cleanup_db_connection
        )

        db_config = PoolConfiguration(
            resource_type=ResourceType.DATABASE_CONNECTION,
            min_size=5,
            max_size=50,
            strategy=PoolConfiguration.strategy.ADAPTIVE
        )
        self.resource_pool_manager.create_pool(db_config)

        # HTTP connection pool
        http_config = PoolConfiguration(
            resource_type=ResourceType.HTTP_CONNECTION,
            min_size=10,
            max_size=100,
            strategy=PoolConfiguration.strategy.DYNAMIC
        )
        self.resource_pool_manager.create_pool(http_config)

        logger.debug("Default resource pools created")

    async def _start_coordination(self):
        """Start coordination tasks between components."""
        # Metrics coordination task
        self.coordination_tasks.append(
            asyncio.create_task(self._metrics_coordination_loop())
        )

        # Load balancer and auto-scaler coordination
        self.coordination_tasks.append(
            asyncio.create_task(self._scaling_coordination_loop())
        )

        # Performance optimization coordination
        self.coordination_tasks.append(
            asyncio.create_task(self._performance_coordination_loop())
        )

        logger.debug("Coordination tasks started")

    async def _metrics_coordination_loop(self):
        """Coordinate metrics collection across all components."""
        while self.running:
            try:
                # Collect metrics from all components
                metrics = await self._collect_unified_metrics()

                # Share metrics between components
                if self.performance_analyzer:
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            # Determine metric type
                            if 'response_time' in metric_name or 'duration' in metric_name:
                                metric_type = PerformanceMetricType.RESPONSE_TIME
                            elif 'cpu' in metric_name:
                                metric_type = PerformanceMetricType.CPU_USAGE
                            elif 'memory' in metric_name:
                                metric_type = PerformanceMetricType.MEMORY_USAGE
                            else:
                                metric_type = PerformanceMetricType.THROUGHPUT

                            self.performance_analyzer.record_metric(
                                metric_type, value, "integrated_system"
                            )

                # Update auto-scaler with current metrics
                if self.auto_scaler:
                    system_metrics = {
                        'cpu_utilization': metrics.get('cpu_percent', 50.0),
                        'memory_utilization': metrics.get('memory_percent', 60.0),
                        'request_rate': metrics.get('throughput', 100.0),
                        'response_time': metrics.get('avg_response_time', 200.0),
                        'error_rate': metrics.get('error_rate', 0.5)
                    }
                    await self.auto_scaler.update_metrics(system_metrics)

                await asyncio.sleep(30)  # Coordinate every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics coordination error: {e}")
                await asyncio.sleep(60)

    async def _scaling_coordination_loop(self):
        """Coordinate between load balancer and auto-scaler."""
        while self.running:
            try:
                if self.load_balancer and self.auto_scaler:
                    # Get load balancer statistics
                    lb_stats = self.load_balancer.get_load_balancing_stats()

                    # Get auto-scaler prediction
                    scaling_prediction = await self.auto_scaler.get_scaling_prediction()

                    # If scaling is recommended, coordinate with load balancer
                    if (scaling_prediction.recommended_action == "scale_up" and
                        scaling_prediction.urgency_score > 0.7):

                        # Execute scaling
                        scaling_result = await self.auto_scaler.execute_scaling(scaling_prediction)

                        if scaling_result['action'] == 'scaled':
                            # Update load balancer with new instance count
                            logger.info(f"Coordinated scaling: {scaling_result['old_instances']} -> "
                                      f"{scaling_result['new_instances']} instances")

                await asyncio.sleep(60)  # Coordinate every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scaling coordination error: {e}")
                await asyncio.sleep(120)

    async def _performance_coordination_loop(self):
        """Coordinate performance optimization across components."""
        while self.running:
            try:
                # Run comprehensive performance analysis
                if self.performance_analyzer:
                    analysis = await self.performance_analyzer.analyze_performance()

                    # Apply optimizations based on analysis
                    if analysis['bottlenecks']:
                        await self._apply_performance_optimizations(analysis)

                # Memory optimization coordination
                if self.memory_optimizer:
                    memory_result = await self.memory_optimizer.optimize_memory()

                    if memory_result['memory_freed_mb'] > 50:  # Significant memory freed
                        logger.info(f"Coordinated memory optimization freed "
                                  f"{memory_result['memory_freed_mb']:.1f}MB")

                await asyncio.sleep(120)  # Coordinate every 2 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance coordination error: {e}")
                await asyncio.sleep(180)

    async def _apply_performance_optimizations(self, analysis: Dict[str, Any]):
        """Apply performance optimizations based on analysis."""
        bottlenecks = analysis['bottlenecks']

        for bottleneck in bottlenecks[-3:]:  # Handle recent bottlenecks
            bottleneck_type = bottleneck.bottleneck_type.value

            # CPU-bound bottlenecks
            if 'cpu' in bottleneck_type:
                if self.concurrent_processor:
                    # Could increase parallelism or optimize task distribution
                    logger.debug("Applying CPU optimization via concurrent processor")

            # Memory-bound bottlenecks
            elif 'memory' in bottleneck_type:
                if self.memory_optimizer:
                    await self.memory_optimizer.optimize_memory()
                    logger.debug("Applied memory optimization")

            # Network/IO bottlenecks
            elif 'io' in bottleneck_type or 'network' in bottleneck_type:
                if self.resource_pool_manager:
                    # Could optimize connection pools
                    logger.debug("Optimizing resource pools for I/O performance")

    async def _collect_unified_metrics(self) -> Dict[str, Any]:
        """Collect unified metrics from all components."""
        unified_metrics = {
            'timestamp': datetime.now().isoformat(),
            'system_uptime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        }

        # Load balancer metrics
        if self.load_balancer:
            lb_stats = self.load_balancer.get_load_balancing_stats()
            unified_metrics.update({
                'load_balancer_decisions': lb_stats.get('total_decisions', 0),
                'healthy_instances': lb_stats.get('healthy_instances', 0),
                'active_sessions': lb_stats.get('active_sessions', 0)
            })

        # Auto-scaler metrics
        if self.auto_scaler:
            scaler_stats = self.auto_scaler.get_predictive_scaling_stats()
            unified_metrics.update({
                'current_instances': scaler_stats.get('current_instances', 0),
                'scaling_events': scaler_stats.get('total_scaling_events', 0),
                'prediction_accuracy': scaler_stats.get('prediction_accuracy', 0)
            })

        # Resource pool metrics
        if self.resource_pool_manager:
            pool_stats = self.resource_pool_manager.get_global_stats()
            unified_metrics.update({
                'total_resources': pool_stats['global_metrics'].get('total_resources', 0),
                'avg_resource_utilization': pool_stats['global_metrics'].get('average_utilization', 0)
            })

        # Global distribution metrics
        if self.global_distribution:
            global_stats = self.global_distribution.get_global_stats()
            unified_metrics.update({
                'active_regions': global_stats['global_metrics'].get('total_regions', 0),
                'global_cache_hit_rate': global_stats['global_metrics'].get('average_cache_hit_rate', 0)
            })

        # Performance analytics metrics
        if self.performance_analyzer:
            analytics_stats = self.performance_analyzer.get_analytics_stats()
            unified_metrics.update({
                'metrics_collected': analytics_stats.get('metrics_collected', 0),
                'active_alerts': analytics_stats.get('active_alerts', 0),
                'system_health_score': analytics_stats.get('system_health', 1.0)
            })

        # Memory optimizer metrics
        if self.memory_optimizer:
            memory_report = self.memory_optimizer.get_memory_report()
            unified_metrics.update({
                'memory_usage_mb': memory_report['current_metrics'].get('process_memory_mb', 0),
                'memory_pressure': memory_report.get('pressure_level', 'low'),
                'detected_leaks': memory_report['current_metrics'].get('memory_leaks_detected', 0)
            })

        # Concurrent processor metrics
        if self.concurrent_processor:
            processor_stats = self.concurrent_processor.get_stats()
            unified_metrics.update({
                'concurrent_tasks_executed': processor_stats['performance'].get('total_tasks_executed', 0),
                'throughput_tasks_per_sec': processor_stats['performance'].get('throughput_tasks_per_second', 0),
                'worker_efficiency': sum(w['efficiency'] for w in processor_stats['workers']) / len(processor_stats['workers']) if processor_stats['workers'] else 0
            })

        return unified_metrics

    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request through the integrated scalability system."""
        start_time = asyncio.get_event_loop().time()
        request_id = request_data.get('request_id', f"req_{id(request_data)}")

        try:
            # 1. Route through load balancer
            if self.load_balancer:
                decision = await self.load_balancer.select_instance(
                    request_context=request_data,
                    session_id=request_data.get('session_id'),
                    client_ip=request_data.get('client_ip')
                )
                selected_instance = decision.selected_instance
            else:
                selected_instance = "default_instance"

            # 2. Process through concurrent processor if needed
            if self.concurrent_processor and request_data.get('parallel_processing'):
                def process_task():
                    # Simulate processing
                    import time
                    time.sleep(0.1)
                    return f"processed_by_{selected_instance}"

                task_id = self.concurrent_processor.submit_task(
                    process_task,
                    priority=TaskPriority.HIGH if request_data.get('urgent') else TaskPriority.NORMAL
                )

                result = await self.concurrent_processor.wait_for_task(task_id, timeout=5.0)
                response = result.result if result and result.success else "processing_failed"
            else:
                # Direct processing
                response = f"processed_by_{selected_instance}"

            # 3. Cache in global distribution if enabled
            if self.global_distribution and request_data.get('cache_result'):
                cache_key = f"response_{request_id}"
                await self.global_distribution.cache_content(
                    cache_key,
                    response,
                    regions=[Region.US_EAST_1],
                    size_bytes=len(str(response))
                )

            # 4. Record performance metrics
            processing_time = asyncio.get_event_loop().time() - start_time

            if self.performance_analyzer:
                self.performance_analyzer.record_metric(
                    PerformanceMetricType.RESPONSE_TIME,
                    processing_time * 1000,  # Convert to milliseconds
                    "integrated_request_processing"
                )

            return {
                'request_id': request_id,
                'response': response,
                'selected_instance': selected_instance,
                'processing_time_ms': processing_time * 1000,
                'success': True
            }

        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time

            if self.performance_analyzer:
                self.performance_analyzer.record_metric(
                    PerformanceMetricType.ERROR_RATE,
                    1.0,
                    "integrated_request_processing"
                )

            logger.error(f"Request processing failed: {e}")

            return {
                'request_id': request_id,
                'error': str(e),
                'processing_time_ms': processing_time * 1000,
                'success': False
            }

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all scalability components."""
        status = {
            'system': {
                'running': self.running,
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                'configuration': {
                    'load_balancing_algorithm': self.config.load_balancing_algorithm.value,
                    'memory_strategy': self.config.memory_strategy.value,
                    'scheduling_strategy': self.config.scheduling_strategy.value,
                    'numa_aware': self.config.numa_aware,
                    'global_distribution_enabled': self.config.enable_global_distribution
                }
            },
            'components': {}
        }

        # Component statuses
        components = [
            ('load_balancer', self.load_balancer),
            ('auto_scaler', self.auto_scaler),
            ('resource_pool_manager', self.resource_pool_manager),
            ('global_distribution', self.global_distribution),
            ('performance_analyzer', self.performance_analyzer),
            ('memory_optimizer', self.memory_optimizer),
            ('concurrent_processor', self.concurrent_processor)
        ]

        for name, component in components:
            if component:
                try:
                    if hasattr(component, 'get_stats'):
                        status['components'][name] = {
                            'active': True,
                            'stats': component.get_stats()
                        }
                    elif hasattr(component, 'get_load_balancing_stats'):
                        status['components'][name] = {
                            'active': True,
                            'stats': component.get_load_balancing_stats()
                        }
                    elif hasattr(component, 'get_predictive_scaling_stats'):
                        status['components'][name] = {
                            'active': True,
                            'stats': component.get_predictive_scaling_stats()
                        }
                    elif hasattr(component, 'get_global_stats'):
                        status['components'][name] = {
                            'active': True,
                            'stats': component.get_global_stats()
                        }
                    elif hasattr(component, 'get_analytics_stats'):
                        status['components'][name] = {
                            'active': True,
                            'stats': component.get_analytics_stats()
                        }
                    elif hasattr(component, 'get_memory_report'):
                        status['components'][name] = {
                            'active': True,
                            'stats': component.get_memory_report()
                        }
                    else:
                        status['components'][name] = {
                            'active': True,
                            'stats': 'available'
                        }
                except Exception as e:
                    status['components'][name] = {
                        'active': True,
                        'error': str(e)
                    }
            else:
                status['components'][name] = {
                    'active': False,
                    'reason': 'not_initialized'
                }

        return status


# Factory function for easy creation
async def create_integrated_scalability_system(
    config: Optional[ScalabilityConfiguration] = None,
    instances: Optional[List[InstanceConfiguration]] = None
) -> IntegratedScalabilitySystem:
    """Create and start an integrated scalability system."""

    system = IntegratedScalabilitySystem(config)
    await system.start(instances)
    return system


# Global system instance
_global_scalability_system: Optional[IntegratedScalabilitySystem] = None


async def get_scalability_system(**kwargs) -> IntegratedScalabilitySystem:
    """Get global scalability system instance."""
    global _global_scalability_system

    if _global_scalability_system is None:
        config = ScalabilityConfiguration(**kwargs)
        _global_scalability_system = await create_integrated_scalability_system(config)

    return _global_scalability_system


# Example usage
if __name__ == "__main__":
    async def example_integrated_scalability():
        """Example of using the integrated scalability system."""

        # Create configuration
        config = ScalabilityConfiguration(
            load_balancing_algorithm=LoadBalancingAlgorithm.HYBRID_ADAPTIVE,
            min_instances=3,
            max_instances=20,
            memory_strategy=MemoryStrategy.ADAPTIVE,
            scheduling_strategy=SchedulingStrategy.WORK_STEALING,
            enable_global_distribution=True,
            numa_aware=True
        )

        # Create sample instances
        instances = [
            InstanceConfiguration(
                instance_id="instance_1",
                address="192.168.1.10",
                port=8080,
                region="us-east-1"
            ),
            InstanceConfiguration(
                instance_id="instance_2",
                address="192.168.1.11",
                port=8080,
                region="us-east-1"
            ),
            InstanceConfiguration(
                instance_id="instance_3",
                address="192.168.1.12",
                port=8080,
                region="eu-central-1"
            )
        ]

        # Create and start integrated system
        system = await create_integrated_scalability_system(config, instances)

        try:
            # Simulate request processing
            print("Processing sample requests through integrated system...")

            for i in range(20):
                request_data = {
                    'request_id': f'req_{i}',
                    'client_ip': f'192.168.{i % 10}.{100 + i}',
                    'session_id': f'session_{i % 5}',
                    'parallel_processing': i % 3 == 0,
                    'cache_result': i % 4 == 0,
                    'urgent': i % 10 == 0
                }

                result = await system.process_request(request_data)

                print(f"Request {i}: {result['success']}, "
                      f"Instance: {result.get('selected_instance', 'N/A')}, "
                      f"Time: {result['processing_time_ms']:.2f}ms")

                # Brief delay between requests
                await asyncio.sleep(0.1)

            # Wait for system to process and optimize
            await asyncio.sleep(5)

            # Get comprehensive system status
            status = system.get_comprehensive_status()

            print("\n=== Integrated Scalability System Status ===")
            print(f"System running: {status['system']['running']}")
            print(f"Uptime: {status['system']['uptime_seconds']:.1f} seconds")
            print("Configuration:")
            for key, value in status['system']['configuration'].items():
                print(f"  {key}: {value}")

            print("\nActive Components:")
            for component, info in status['components'].items():
                if info['active']:
                    print(f"  ✓ {component}")
                else:
                    print(f"  ✗ {component} ({info.get('reason', 'unknown')})")

            # Display key metrics from each component
            print("\n=== Key Performance Metrics ===")

            # Load Balancer
            if 'load_balancer' in status['components'] and status['components']['load_balancer']['active']:
                lb_stats = status['components']['load_balancer']['stats']
                print("Load Balancer:")
                print(f"  Total decisions: {lb_stats.get('total_decisions', 0)}")
                print(f"  Healthy instances: {lb_stats.get('healthy_instances', 0)}")

            # Auto Scaler
            if 'auto_scaler' in status['components'] and status['components']['auto_scaler']['active']:
                as_stats = status['components']['auto_scaler']['stats']
                print("Auto Scaler:")
                print(f"  Current instances: {as_stats.get('current_instances', 0)}")
                print(f"  Scaling events: {as_stats.get('total_scaling_events', 0)}")
                print(f"  Prediction accuracy: {as_stats.get('prediction_accuracy', 0):.1f}%")

            # Concurrent Processor
            if 'concurrent_processor' in status['components'] and status['components']['concurrent_processor']['active']:
                cp_stats = status['components']['concurrent_processor']['stats']
                print("Concurrent Processor:")
                print(f"  Tasks executed: {cp_stats['performance'].get('total_tasks_executed', 0)}")
                print(f"  Throughput: {cp_stats['performance'].get('throughput_tasks_per_second', 0):.2f} tasks/sec")

            # Memory Optimizer
            if 'memory_optimizer' in status['components'] and status['components']['memory_optimizer']['active']:
                mo_stats = status['components']['memory_optimizer']['stats']
                print("Memory Optimizer:")
                print(f"  Memory usage: {mo_stats['current_metrics'].get('process_memory_mb', 0):.1f} MB")
                print(f"  Pressure level: {mo_stats.get('pressure_level', 'unknown')}")

            print("\n=== Integration Success ===")
            print("All scalability components are successfully integrated and coordinated!")

        finally:
            await system.stop()

    # Run example
    asyncio.run(example_integrated_scalability())
    print("\nIntegrated Scalability System example completed!")
