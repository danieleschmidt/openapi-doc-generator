"""
Advanced Performance Analytics System

This module provides comprehensive real-time performance monitoring, bottleneck
detection, performance profiling, and advanced analytics for distributed systems
with machine learning-based anomaly detection and performance optimization.

Features:
- Real-time performance monitoring and metrics collection
- Advanced bottleneck detection and analysis
- Machine learning-based anomaly detection
- Performance profiling and code analysis
- Resource utilization analytics
- Predictive performance modeling
- Alert system for performance issues
- Performance optimization recommendations
"""

import asyncio
import cProfile
import gc
import io
import logging
import os
import pstats
import sys
import threading
import time
import tracemalloc
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import psutil
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .enhanced_monitoring import get_monitor
from .performance_optimizer import get_optimizer

logger = logging.getLogger(__name__)


class PerformanceMetricType(Enum):
    """Types of performance metrics."""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    QUEUE_DEPTH = "queue_depth"
    ERROR_RATE = "error_rate"
    CACHE_HIT_RATE = "cache_hit_rate"
    GC_TIME = "gc_time"
    THREAD_COUNT = "thread_count"
    CONNECTION_COUNT = "connection_count"


class BottleneckType(Enum):
    """Types of performance bottlenecks."""
    CPU_BOUND = "cpu_bound"
    MEMORY_BOUND = "memory_bound"
    IO_BOUND = "io_bound"
    NETWORK_BOUND = "network_bound"
    DATABASE_BOUND = "database_bound"
    LOCK_CONTENTION = "lock_contention"
    GC_PRESSURE = "gc_pressure"
    THREAD_POOL_EXHAUSTION = "thread_pool_exhaustion"
    MEMORY_LEAK = "memory_leak"
    INFINITE_LOOP = "infinite_loop"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""
    metric_type: PerformanceMetricType
    value: float
    timestamp: datetime
    component: str
    instance_id: str = "default"
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class BottleneckDetection:
    """Detected performance bottleneck."""
    bottleneck_type: BottleneckType
    severity: float  # 0.0 to 1.0
    affected_components: List[str]
    description: str
    recommendations: List[str]
    evidence: Dict[str, Any]
    detected_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None


@dataclass
class PerformanceAlert:
    """Performance-related alert."""
    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    affected_metrics: List[PerformanceMetricType]
    threshold_exceeded: Dict[str, float]
    recommendations: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class ProfilingResult:
    """Code profiling result."""
    function_name: str
    total_time: float
    cumulative_time: float
    call_count: int
    per_call_time: float
    percentage: float
    file_name: str
    line_number: int


class PerformanceProfiler:
    """Advanced performance profiler with real-time capabilities."""

    def __init__(self):
        self.profiler = None
        self.profiling_active = False
        self.profiling_results: Dict[str, List[ProfilingResult]] = {}
        self.memory_tracker = None

        # Enable tracemalloc for memory profiling
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        self.monitor = get_monitor()

    def start_profiling(self, components: Optional[List[str]] = None):
        """Start performance profiling."""
        if self.profiling_active:
            return

        self.profiler = cProfile.Profile()
        self.profiler.enable()
        self.profiling_active = True

        logger.info("Performance profiling started")

    def stop_profiling(self) -> Dict[str, List[ProfilingResult]]:
        """Stop profiling and return results."""
        if not self.profiling_active or not self.profiler:
            return {}

        self.profiler.disable()
        self.profiling_active = False

        # Process profiling results
        stats_buffer = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=stats_buffer)
        ps.sort_stats('cumulative')
        ps.print_stats()

        # Parse results
        results = self._parse_profiling_stats(ps)
        self.profiling_results[datetime.now().isoformat()] = results

        logger.info(f"Performance profiling stopped, analyzed {len(results)} functions")
        return {"current": results}

    def _parse_profiling_stats(self, stats: pstats.Stats) -> List[ProfilingResult]:
        """Parse profiling statistics into structured results."""
        results = []

        for func_key, (call_count, total_time, cumulative_time, callers) in stats.stats.items():
            file_name, line_number, function_name = func_key

            per_call_time = total_time / call_count if call_count > 0 else 0
            percentage = (cumulative_time / stats.total_tt * 100) if stats.total_tt > 0 else 0

            result = ProfilingResult(
                function_name=function_name,
                total_time=total_time,
                cumulative_time=cumulative_time,
                call_count=call_count,
                per_call_time=per_call_time,
                percentage=percentage,
                file_name=file_name,
                line_number=line_number
            )

            results.append(result)

        # Sort by cumulative time
        results.sort(key=lambda r: r.cumulative_time, reverse=True)
        return results[:50]  # Top 50 functions

    def get_memory_profile(self) -> Dict[str, Any]:
        """Get memory profiling information."""
        if not tracemalloc.is_tracing():
            return {}

        # Get current memory snapshot
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        memory_profile = {
            'total_memory_mb': sum(stat.size for stat in top_stats) / 1024 / 1024,
            'block_count': sum(stat.count for stat in top_stats),
            'top_allocations': []
        }

        # Top memory allocations
        for stat in top_stats[:20]:
            memory_profile['top_allocations'].append({
                'file': stat.traceback.format()[-1] if stat.traceback else 'unknown',
                'size_mb': stat.size / 1024 / 1024,
                'count': stat.count
            })

        return memory_profile

    def analyze_gc_performance(self) -> Dict[str, Any]:
        """Analyze garbage collection performance."""
        gc_stats = {
            'generation_0': gc.get_count()[0],
            'generation_1': gc.get_count()[1],
            'generation_2': gc.get_count()[2],
            'total_collections': sum(gc.get_stats()[i]['collections'] for i in range(3)),
            'total_collected': sum(gc.get_stats()[i]['collected'] for i in range(3)),
            'total_uncollectable': sum(gc.get_stats()[i]['uncollectable'] for i in range(3))
        }

        # Check for potential memory leaks
        if gc_stats['total_uncollectable'] > 100:
            gc_stats['memory_leak_risk'] = 'high'
        elif gc_stats['total_uncollectable'] > 50:
            gc_stats['memory_leak_risk'] = 'medium'
        else:
            gc_stats['memory_leak_risk'] = 'low'

        return gc_stats


class BottleneckDetector:
    """Advanced bottleneck detection system using machine learning."""

    def __init__(self):
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.bottlenecks: List[BottleneckDetection] = []

        # ML models for anomaly detection
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.model_trained = False

        # Thresholds for bottleneck detection
        self.thresholds = {
            PerformanceMetricType.CPU_USAGE: 85.0,
            PerformanceMetricType.MEMORY_USAGE: 90.0,
            PerformanceMetricType.RESPONSE_TIME: 2000.0,  # 2 seconds
            PerformanceMetricType.ERROR_RATE: 5.0,  # 5%
            PerformanceMetricType.QUEUE_DEPTH: 100,
            PerformanceMetricType.THREAD_COUNT: 500
        }

        logger.info("Bottleneck detector initialized")

    def add_metric(self, metric: PerformanceMetric):
        """Add a performance metric for analysis."""
        metric_key = f"{metric.component}_{metric.metric_type.value}"
        self.metric_history[metric_key].append(metric)

        # Retrain model periodically
        if len(self.metric_history[metric_key]) % 100 == 0:
            asyncio.create_task(self._retrain_models())

    async def detect_bottlenecks(self) -> List[BottleneckDetection]:
        """Detect performance bottlenecks using various techniques."""
        detected_bottlenecks = []

        # Rule-based detection
        rule_based_bottlenecks = await self._rule_based_detection()
        detected_bottlenecks.extend(rule_based_bottlenecks)

        # ML-based anomaly detection
        if self.model_trained:
            ml_bottlenecks = await self._ml_based_detection()
            detected_bottlenecks.extend(ml_bottlenecks)

        # Pattern-based detection
        pattern_bottlenecks = await self._pattern_based_detection()
        detected_bottlenecks.extend(pattern_bottlenecks)

        # Statistical analysis
        statistical_bottlenecks = await self._statistical_analysis()
        detected_bottlenecks.extend(statistical_bottlenecks)

        # Update bottleneck list
        self.bottlenecks.extend(detected_bottlenecks)

        return detected_bottlenecks

    async def _rule_based_detection(self) -> List[BottleneckDetection]:
        """Rule-based bottleneck detection."""
        bottlenecks = []

        for metric_key, metrics in self.metric_history.items():
            if not metrics:
                continue

            recent_metrics = list(metrics)[-10:]  # Last 10 metrics
            if len(recent_metrics) < 5:
                continue

            # Check for threshold violations
            for metric in recent_metrics:
                threshold = self.thresholds.get(metric.metric_type)
                if threshold and metric.value > threshold:

                    bottleneck_type = self._classify_bottleneck_type(metric)
                    severity = min(1.0, metric.value / threshold - 1.0)

                    bottleneck = BottleneckDetection(
                        bottleneck_type=bottleneck_type,
                        severity=severity,
                        affected_components=[metric.component],
                        description=f"{metric.metric_type.value} exceeded threshold: {metric.value:.2f} > {threshold}",
                        recommendations=self._get_bottleneck_recommendations(bottleneck_type),
                        evidence={
                            'metric_type': metric.metric_type.value,
                            'current_value': metric.value,
                            'threshold': threshold,
                            'component': metric.component
                        }
                    )

                    bottlenecks.append(bottleneck)

        return bottlenecks

    async def _ml_based_detection(self) -> List[BottleneckDetection]:
        """Machine learning-based anomaly detection."""
        bottlenecks = []

        try:
            # Prepare feature matrix
            features = []
            metric_info = []

            for metric_key, metrics in self.metric_history.items():
                if len(metrics) < 10:
                    continue

                recent_metrics = list(metrics)[-50:]  # Last 50 metrics

                # Extract statistical features
                values = [m.value for m in recent_metrics]
                feature_vector = [
                    np.mean(values),
                    np.std(values),
                    np.min(values),
                    np.max(values),
                    np.percentile(values, 95),
                    len(values)
                ]

                features.append(feature_vector)
                metric_info.append((metric_key, recent_metrics[-1]))

            if len(features) < 5:
                return bottlenecks

            # Detect anomalies
            features_array = np.array(features)
            features_scaled = self.scaler.transform(features_array)

            anomaly_scores = self.isolation_forest.decision_function(features_scaled)
            anomalies = self.isolation_forest.predict(features_scaled)

            # Process anomalies
            for i, (is_anomaly, score) in enumerate(zip(anomalies, anomaly_scores)):
                if is_anomaly == -1:  # Anomaly detected
                    metric_key, last_metric = metric_info[i]
                    severity = max(0.0, min(1.0, abs(score)))

                    bottleneck = BottleneckDetection(
                        bottleneck_type=BottleneckType.CPU_BOUND,  # Default, would need better classification
                        severity=severity,
                        affected_components=[last_metric.component],
                        description=f"ML anomaly detected in {metric_key} (score: {score:.3f})",
                        recommendations=["Monitor component closely", "Check for unusual load patterns"],
                        evidence={
                            'anomaly_score': score,
                            'metric_key': metric_key,
                            'detection_method': 'isolation_forest'
                        }
                    )

                    bottlenecks.append(bottleneck)

        except Exception as e:
            logger.warning(f"ML-based detection failed: {e}")

        return bottlenecks

    async def _pattern_based_detection(self) -> List[BottleneckDetection]:
        """Pattern-based bottleneck detection."""
        bottlenecks = []

        # Memory leak detection
        memory_metrics = []
        for metric_key, metrics in self.metric_history.items():
            if 'memory' in metric_key.lower():
                memory_metrics.extend(list(metrics))

        if len(memory_metrics) > 20:
            memory_values = [m.value for m in memory_metrics[-20:]]

            # Check for consistent increase (potential memory leak)
            slope, _, r_value, _, _ = stats.linregress(range(len(memory_values)), memory_values)

            if slope > 0.5 and r_value > 0.7:  # Consistent upward trend
                bottleneck = BottleneckDetection(
                    bottleneck_type=BottleneckType.MEMORY_LEAK,
                    severity=min(1.0, slope / 10.0),
                    affected_components=["memory_subsystem"],
                    description=f"Potential memory leak detected (slope: {slope:.3f})",
                    recommendations=[
                        "Check for unclosed resources",
                        "Review object lifecycle management",
                        "Monitor garbage collection metrics"
                    ],
                    evidence={
                        'trend_slope': slope,
                        'correlation': r_value,
                        'sample_size': len(memory_values)
                    }
                )

                bottlenecks.append(bottleneck)

        # Thread pool exhaustion detection
        thread_metrics = []
        for metric_key, metrics in self.metric_history.items():
            if 'thread' in metric_key.lower():
                thread_metrics.extend(list(metrics))

        if thread_metrics:
            recent_thread_count = thread_metrics[-1].value if thread_metrics else 0
            if recent_thread_count > 400:  # High thread count
                bottleneck = BottleneckDetection(
                    bottleneck_type=BottleneckType.THREAD_POOL_EXHAUSTION,
                    severity=min(1.0, recent_thread_count / 500.0),
                    affected_components=["thread_pool"],
                    description=f"High thread count detected: {recent_thread_count}",
                    recommendations=[
                        "Review thread pool configuration",
                        "Check for thread leaks",
                        "Optimize blocking operations"
                    ],
                    evidence={'thread_count': recent_thread_count}
                )

                bottlenecks.append(bottleneck)

        return bottlenecks

    async def _statistical_analysis(self) -> List[BottleneckDetection]:
        """Statistical analysis for bottleneck detection."""
        bottlenecks = []

        for metric_key, metrics in self.metric_history.items():
            if len(metrics) < 30:
                continue

            recent_values = [m.value for m in list(metrics)[-30:]]

            # Check for high variance (instability)
            mean_val = np.mean(recent_values)
            std_val = np.std(recent_values)

            if mean_val > 0 and (std_val / mean_val) > 0.5:  # Coefficient of variation > 50%

                # Determine bottleneck type from metric key
                if 'cpu' in metric_key.lower():
                    bottleneck_type = BottleneckType.CPU_BOUND
                elif 'memory' in metric_key.lower():
                    bottleneck_type = BottleneckType.MEMORY_BOUND
                elif 'io' in metric_key.lower():
                    bottleneck_type = BottleneckType.IO_BOUND
                else:
                    bottleneck_type = BottleneckType.CPU_BOUND  # Default

                bottleneck = BottleneckDetection(
                    bottleneck_type=bottleneck_type,
                    severity=min(1.0, std_val / mean_val),
                    affected_components=[metric_key.split('_')[0]],
                    description=f"High variance detected in {metric_key} (CV: {std_val/mean_val:.3f})",
                    recommendations=[
                        "Investigate cause of performance instability",
                        "Check for resource contention",
                        "Review load balancing"
                    ],
                    evidence={
                        'coefficient_of_variation': std_val / mean_val,
                        'mean': mean_val,
                        'std_dev': std_val
                    }
                )

                bottlenecks.append(bottleneck)

        return bottlenecks

    def _classify_bottleneck_type(self, metric: PerformanceMetric) -> BottleneckType:
        """Classify bottleneck type based on metric."""
        metric_type = metric.metric_type

        mapping = {
            PerformanceMetricType.CPU_USAGE: BottleneckType.CPU_BOUND,
            PerformanceMetricType.MEMORY_USAGE: BottleneckType.MEMORY_BOUND,
            PerformanceMetricType.DISK_IO: BottleneckType.IO_BOUND,
            PerformanceMetricType.NETWORK_IO: BottleneckType.NETWORK_BOUND,
            PerformanceMetricType.QUEUE_DEPTH: BottleneckType.THREAD_POOL_EXHAUSTION,
            PerformanceMetricType.GC_TIME: BottleneckType.GC_PRESSURE,
            PerformanceMetricType.THREAD_COUNT: BottleneckType.THREAD_POOL_EXHAUSTION,
        }

        return mapping.get(metric_type, BottleneckType.CPU_BOUND)

    def _get_bottleneck_recommendations(self, bottleneck_type: BottleneckType) -> List[str]:
        """Get recommendations for bottleneck type."""
        recommendations = {
            BottleneckType.CPU_BOUND: [
                "Scale horizontally by adding more instances",
                "Optimize CPU-intensive algorithms",
                "Consider using faster hardware",
                "Profile code to identify hot spots"
            ],
            BottleneckType.MEMORY_BOUND: [
                "Increase available memory",
                "Optimize memory usage patterns",
                "Implement memory pooling",
                "Check for memory leaks"
            ],
            BottleneckType.IO_BOUND: [
                "Use faster storage (SSD)",
                "Implement caching strategies",
                "Optimize database queries",
                "Use asynchronous I/O operations"
            ],
            BottleneckType.NETWORK_BOUND: [
                "Increase network bandwidth",
                "Implement data compression",
                "Use CDN for static content",
                "Optimize network protocols"
            ],
            BottleneckType.GC_PRESSURE: [
                "Tune garbage collection parameters",
                "Reduce object allocation rate",
                "Use object pooling",
                "Optimize data structures"
            ],
            BottleneckType.THREAD_POOL_EXHAUSTION: [
                "Increase thread pool size",
                "Optimize blocking operations",
                "Use asynchronous programming",
                "Implement backpressure mechanisms"
            ]
        }

        return recommendations.get(bottleneck_type, ["Monitor the situation closely"])

    async def _retrain_models(self):
        """Retrain ML models with recent data."""
        try:
            # Prepare training data
            features = []

            for metric_key, metrics in self.metric_history.items():
                if len(metrics) < 20:
                    continue

                # Create feature windows
                metric_list = list(metrics)
                for i in range(20, len(metric_list)):
                    window = metric_list[i-20:i]
                    values = [m.value for m in window]

                    feature_vector = [
                        np.mean(values),
                        np.std(values),
                        np.min(values),
                        np.max(values),
                        np.percentile(values, 95),
                        len(values)
                    ]

                    features.append(feature_vector)

            if len(features) > 50:
                features_array = np.array(features)

                # Fit scaler and model
                self.scaler.fit(features_array)
                features_scaled = self.scaler.transform(features_array)

                self.isolation_forest.fit(features_scaled)
                self.model_trained = True

                logger.debug(f"ML models retrained with {len(features)} samples")

        except Exception as e:
            logger.warning(f"Model retraining failed: {e}")


class PerformanceAnalyzer:
    """
    Advanced performance analytics system with real-time monitoring.
    """

    def __init__(self):
        # Core components
        self.profiler = PerformanceProfiler()
        self.bottleneck_detector = BottleneckDetector()

        # Metrics collection
        self.metrics_buffer: deque = deque(maxlen=10000)
        self.component_metrics: Dict[str, Dict[PerformanceMetricType, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=1000))
        )

        # Alerts and notifications
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []

        # System monitoring
        self.system_stats_history: deque = deque(maxlen=1000)

        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.analysis_task: Optional[asyncio.Task] = None
        self.running = False

        # External integrations
        self.monitor = get_monitor()
        self.optimizer = get_optimizer()

        logger.info("Performance Analyzer initialized")

    async def start(self):
        """Start performance analytics."""
        if self.running:
            return

        self.running = True

        # Start background tasks
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.analysis_task = asyncio.create_task(self._analysis_loop())

        logger.info("Performance analytics started")

    async def stop(self):
        """Stop performance analytics."""
        if not self.running:
            return

        self.running = False

        # Cancel background tasks
        tasks = [self.monitoring_task, self.analysis_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("Performance analytics stopped")

    def record_metric(self,
                     metric_type: PerformanceMetricType,
                     value: float,
                     component: str,
                     instance_id: str = "default",
                     tags: Optional[Dict[str, str]] = None):
        """Record a performance metric."""
        metric = PerformanceMetric(
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(),
            component=component,
            instance_id=instance_id,
            tags=tags or {}
        )

        self.metrics_buffer.append(metric)
        self.component_metrics[component][metric_type].append(metric)

        # Add to bottleneck detector
        self.bottleneck_detector.add_metric(metric)

        # Record in external monitor
        self.monitor.record_metric(
            f"perf_{component}_{metric_type.value}",
            value,
            "gauge"
        )

    def start_profiling(self, components: Optional[List[str]] = None):
        """Start performance profiling."""
        self.profiler.start_profiling(components)

    def stop_profiling(self) -> Dict[str, List[ProfilingResult]]:
        """Stop profiling and get results."""
        return self.profiler.stop_profiling()

    async def analyze_performance(self) -> Dict[str, Any]:
        """Perform comprehensive performance analysis."""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'system_overview': await self._get_system_overview(),
            'bottlenecks': await self.bottleneck_detector.detect_bottlenecks(),
            'component_analysis': self._analyze_components(),
            'memory_analysis': self.profiler.get_memory_profile(),
            'gc_analysis': self.profiler.analyze_gc_performance(),
            'recommendations': []
        }

        # Generate recommendations based on analysis
        analysis['recommendations'] = self._generate_recommendations(analysis)

        return analysis

    async def _get_system_overview(self) -> Dict[str, Any]:
        """Get system-wide performance overview."""
        # System resource usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Process-specific metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        process_cpu = process.cpu_percent()

        # Thread information
        thread_count = threading.active_count()

        overview = {
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': (disk.used / disk.total) * 100,
                'disk_free_gb': disk.free / (1024**3)
            },
            'process': {
                'cpu_percent': process_cpu,
                'memory_mb': process_memory.rss / (1024**2),
                'thread_count': thread_count,
                'pid': os.getpid()
            },
            'python': {
                'version': sys.version,
                'gc_enabled': gc.isenabled(),
                'gc_thresholds': gc.get_threshold()
            }
        }

        self.system_stats_history.append(overview)
        return overview

    def _analyze_components(self) -> Dict[str, Any]:
        """Analyze performance by component."""
        component_analysis = {}

        for component, metrics_by_type in self.component_metrics.items():
            component_stats = {}

            for metric_type, metric_history in metrics_by_type.items():
                if not metric_history:
                    continue

                values = [m.value for m in metric_history]

                stats = {
                    'current': values[-1] if values else 0,
                    'average': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'p95': np.percentile(values, 95) if len(values) > 1 else values[0],
                    'trend': self._calculate_trend(values),
                    'sample_count': len(values)
                }

                component_stats[metric_type.value] = stats

            component_analysis[component] = component_stats

        return component_analysis

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values."""
        if len(values) < 5:
            return "insufficient_data"

        # Use linear regression to determine trend
        x = list(range(len(values)))
        slope, _, r_value, _, _ = stats.linregress(x, values)

        # Classify trend based on slope and correlation
        if abs(r_value) < 0.3:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        # System-level recommendations
        system = analysis['system_overview']['system']

        if system['cpu_percent'] > 80:
            recommendations.append("High CPU usage detected. Consider scaling horizontally or optimizing CPU-bound operations.")

        if system['memory_percent'] > 85:
            recommendations.append("High memory usage detected. Consider increasing memory or optimizing memory usage patterns.")

        if system['disk_percent'] > 90:
            recommendations.append("Low disk space detected. Consider cleaning up or increasing storage capacity.")

        # GC recommendations
        gc_analysis = analysis['gc_analysis']
        if gc_analysis['memory_leak_risk'] == 'high':
            recommendations.append("High memory leak risk detected. Review object lifecycle management and check for unclosed resources.")

        # Bottleneck recommendations
        bottlenecks = analysis['bottlenecks']
        for bottleneck in bottlenecks[-5:]:  # Latest 5 bottlenecks
            recommendations.extend(bottleneck.recommendations)

        # Component-specific recommendations
        component_analysis = analysis['component_analysis']
        for component, metrics in component_analysis.items():
            for metric_name, stats in metrics.items():
                if stats['trend'] == 'increasing' and 'error' in metric_name:
                    recommendations.append(f"Increasing error rate in {component}. Investigate root cause.")

                if stats['current'] > stats['p95'] * 1.5:
                    recommendations.append(f"Unusual spike in {metric_name} for {component}. Monitor closely.")

        return list(set(recommendations))  # Remove duplicates

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add callback for performance alerts."""
        self.alert_callbacks.append(callback)

    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.running:
            try:
                # Collect system metrics
                await self._collect_system_metrics()

                # Check for alert conditions
                await self._check_alert_conditions()

                await asyncio.sleep(10)  # Monitor every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(30)

    async def _collect_system_metrics(self):
        """Collect system-wide performance metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.record_metric(PerformanceMetricType.CPU_USAGE, cpu_percent, "system")

        # Memory metrics
        memory = psutil.virtual_memory()
        self.record_metric(PerformanceMetricType.MEMORY_USAGE, memory.percent, "system")

        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info().rss / (1024**2)  # MB
        self.record_metric(PerformanceMetricType.MEMORY_USAGE, process_memory, "process")

        # Thread count
        thread_count = threading.active_count()
        self.record_metric(PerformanceMetricType.THREAD_COUNT, float(thread_count), "process")

        # GC metrics
        gc_stats = self.profiler.analyze_gc_performance()
        self.record_metric(PerformanceMetricType.GC_TIME, float(gc_stats['total_collections']), "gc")

    async def _check_alert_conditions(self):
        """Check for conditions that should trigger alerts."""
        current_time = datetime.now()

        # Check recent metrics for alert conditions
        for component, metrics_by_type in self.component_metrics.items():
            for metric_type, metric_history in metrics_by_type.items():
                if not metric_history:
                    continue

                recent_metric = metric_history[-1]

                # High CPU usage alert
                if (metric_type == PerformanceMetricType.CPU_USAGE and
                    recent_metric.value > 90):
                    await self._create_alert(
                        f"high_cpu_{component}",
                        AlertSeverity.HIGH,
                        f"High CPU usage in {component}",
                        f"CPU usage is {recent_metric.value:.1f}%",
                        [PerformanceMetricType.CPU_USAGE],
                        {"cpu_usage": recent_metric.value}
                    )

                # High memory usage alert
                elif (metric_type == PerformanceMetricType.MEMORY_USAGE and
                      recent_metric.value > 95):
                    await self._create_alert(
                        f"high_memory_{component}",
                        AlertSeverity.CRITICAL,
                        f"Critical memory usage in {component}",
                        f"Memory usage is {recent_metric.value:.1f}%",
                        [PerformanceMetricType.MEMORY_USAGE],
                        {"memory_usage": recent_metric.value}
                    )

                # High error rate alert
                elif (metric_type == PerformanceMetricType.ERROR_RATE and
                      recent_metric.value > 10):
                    await self._create_alert(
                        f"high_errors_{component}",
                        AlertSeverity.HIGH,
                        f"High error rate in {component}",
                        f"Error rate is {recent_metric.value:.2f}%",
                        [PerformanceMetricType.ERROR_RATE],
                        {"error_rate": recent_metric.value}
                    )

    async def _create_alert(self,
                          alert_id: str,
                          severity: AlertSeverity,
                          title: str,
                          description: str,
                          affected_metrics: List[PerformanceMetricType],
                          threshold_exceeded: Dict[str, float]):
        """Create and process a performance alert."""
        if alert_id in self.active_alerts:
            return  # Alert already active

        alert = PerformanceAlert(
            alert_id=alert_id,
            severity=severity,
            title=title,
            description=description,
            affected_metrics=affected_metrics,
            threshold_exceeded=threshold_exceeded,
            recommendations=[]
        )

        self.active_alerts[alert_id] = alert

        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        # Log alert
        log_level = {
            AlertSeverity.LOW: logging.INFO,
            AlertSeverity.MEDIUM: logging.WARNING,
            AlertSeverity.HIGH: logging.WARNING,
            AlertSeverity.CRITICAL: logging.ERROR
        }[severity]

        logger.log(log_level, f"PERFORMANCE ALERT [{severity.value.upper()}] {title}: {description}")

    async def _analysis_loop(self):
        """Background analysis loop."""
        while self.running:
            try:
                # Perform periodic analysis
                if len(self.metrics_buffer) > 100:
                    analysis = await self.analyze_performance()

                    # Store analysis results (could be saved to database)
                    logger.debug(f"Performance analysis completed: "
                               f"{len(analysis['bottlenecks'])} bottlenecks detected")

                await asyncio.sleep(60)  # Analyze every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Analysis loop error: {e}")
                await asyncio.sleep(120)

    def get_analytics_stats(self) -> Dict[str, Any]:
        """Get comprehensive analytics statistics."""
        return {
            'metrics_collected': len(self.metrics_buffer),
            'components_monitored': len(self.component_metrics),
            'active_alerts': len(self.active_alerts),
            'bottlenecks_detected': len(self.bottleneck_detector.bottlenecks),
            'profiling_active': self.profiler.profiling_active,
            'model_trained': self.bottleneck_detector.model_trained,
            'system_health': self._get_system_health_score()
        }

    def _get_system_health_score(self) -> float:
        """Calculate overall system health score."""
        if not self.system_stats_history:
            return 1.0

        latest_stats = self.system_stats_history[-1]

        # Weight different factors
        cpu_health = max(0, 1.0 - latest_stats['system']['cpu_percent'] / 100.0)
        memory_health = max(0, 1.0 - latest_stats['system']['memory_percent'] / 100.0)
        disk_health = max(0, 1.0 - latest_stats['system']['disk_percent'] / 100.0)

        # Alert penalty
        alert_penalty = len(self.active_alerts) * 0.1

        health_score = (cpu_health + memory_health + disk_health) / 3.0 - alert_penalty
        return max(0.0, min(1.0, health_score))


# Global performance analyzer
_global_analyzer: Optional[PerformanceAnalyzer] = None


def get_performance_analyzer() -> PerformanceAnalyzer:
    """Get global performance analyzer instance."""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = PerformanceAnalyzer()
    return _global_analyzer


# Decorator for automatic performance tracking
def track_performance(component: str, metric_type: PerformanceMetricType = PerformanceMetricType.RESPONSE_TIME):
    """Decorator to automatically track function performance."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            analyzer = get_performance_analyzer()
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                analyzer.record_metric(metric_type, duration * 1000, component)  # Convert to milliseconds
                return result
            except Exception:
                duration = time.time() - start_time
                analyzer.record_metric(PerformanceMetricType.ERROR_RATE, 1.0, component)
                raise

        def sync_wrapper(*args, **kwargs):
            analyzer = get_performance_analyzer()
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                analyzer.record_metric(metric_type, duration * 1000, component)
                return result
            except Exception:
                duration = time.time() - start_time
                analyzer.record_metric(PerformanceMetricType.ERROR_RATE, 1.0, component)
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Example usage
if __name__ == "__main__":
    async def example_performance_analytics():
        """Example of using performance analytics."""

        # Create analyzer
        analyzer = PerformanceAnalyzer()
        await analyzer.start()

        # Add alert callback
        def alert_callback(alert: PerformanceAlert):
            print(f"ALERT: {alert.title} - {alert.description}")

        analyzer.add_alert_callback(alert_callback)

        try:
            # Start profiling
            analyzer.start_profiling()

            # Simulate some work and record metrics
            for i in range(100):
                # Simulate varying performance
                response_time = np.random.uniform(50, 500)  # 50-500ms
                cpu_usage = np.random.uniform(30, 95)  # 30-95%
                memory_usage = np.random.uniform(40, 90)  # 40-90%
                error_rate = np.random.uniform(0, 2)  # 0-2%

                analyzer.record_metric(PerformanceMetricType.RESPONSE_TIME, response_time, "api_server")
                analyzer.record_metric(PerformanceMetricType.CPU_USAGE, cpu_usage, "api_server")
                analyzer.record_metric(PerformanceMetricType.MEMORY_USAGE, memory_usage, "api_server")
                analyzer.record_metric(PerformanceMetricType.ERROR_RATE, error_rate, "api_server")

                await asyncio.sleep(0.1)  # 100ms intervals

            # Stop profiling and get results
            profiling_results = analyzer.stop_profiling()
            print(f"Profiling results: {len(profiling_results.get('current', []))} functions analyzed")

            # Perform comprehensive analysis
            analysis = await analyzer.analyze_performance()

            print("Performance Analysis:")
            print(f"- System CPU: {analysis['system_overview']['system']['cpu_percent']:.1f}%")
            print(f"- System Memory: {analysis['system_overview']['system']['memory_percent']:.1f}%")
            print(f"- Bottlenecks detected: {len(analysis['bottlenecks'])}")
            print(f"- Recommendations: {len(analysis['recommendations'])}")

            for bottleneck in analysis['bottlenecks'][-3:]:  # Show last 3 bottlenecks
                print(f"  Bottleneck: {bottleneck.bottleneck_type.value} "
                      f"(severity: {bottleneck.severity:.2f}) - {bottleneck.description}")

            for recommendation in analysis['recommendations'][:5]:  # Show first 5 recommendations
                print(f"  Recommendation: {recommendation}")

            # Get analytics stats
            stats = analyzer.get_analytics_stats()
            print(f"Analytics Stats: {stats}")

            # Wait for monitoring cycles
            await asyncio.sleep(5)

        finally:
            await analyzer.stop()

    # Run example
    asyncio.run(example_performance_analytics())
    print("Performance analytics example completed!")
