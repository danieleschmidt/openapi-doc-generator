"""
Intelligent Auto-Scaling System

This module provides advanced auto-scaling capabilities that automatically
adjust resource allocation, processing strategies, and system configuration
based on workload characteristics and performance metrics.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
import psutil

from .enhanced_monitoring import get_monitor
from .performance_optimizer import get_optimizer, OptimizationConfig, ProcessingMode


class ScalingDirection(Enum):
    """Direction of scaling operations."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ScalingTrigger(Enum):
    """Triggers for scaling decisions."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"


@dataclass
class ScalingRule:
    """Configuration for scaling rules."""
    trigger: ScalingTrigger
    threshold_up: float
    threshold_down: float
    direction: ScalingDirection
    cooldown_seconds: float = 300.0  # 5 minutes
    evaluation_window: float = 60.0  # 1 minute
    min_data_points: int = 3


@dataclass
class ScalingAction:
    """Represents a scaling action to be performed."""
    action_type: str
    target: str
    old_value: Any
    new_value: Any
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ResourceLimits:
    """Resource limits for scaling operations."""
    min_workers: int = 1
    max_workers: int = 32
    min_cache_size: int = 100
    max_cache_size: int = 10000
    min_memory_threshold: float = 0.5
    max_memory_threshold: float = 0.95


class IntelligentAutoScaler:
    """Advanced auto-scaling system with intelligent decision making."""
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.logger = logging.getLogger(__name__)
        self.monitor = get_monitor()
        self.optimizer = get_optimizer()
        
        self.limits = limits or ResourceLimits()
        self.scaling_rules = self._create_default_rules()
        
        # State tracking
        self.last_scaling_actions: Dict[str, datetime] = {}
        self.metrics_history: Dict[str, List[Tuple[float, float]]] = {}  # (timestamp, value)
        self.scaling_history: List[ScalingAction] = []
        
        # Configuration
        self.scaling_enabled = True
        self.learning_enabled = True
        self.predictive_scaling = True
        
        # Async control
        self.scaling_task: Optional[asyncio.Task] = None
        self.stop_event = asyncio.Event()
        
        # Machine learning for predictive scaling
        self.workload_patterns: Dict[str, List[float]] = {}
        self.prediction_accuracy: float = 0.0
        
        self.logger.info("Intelligent auto-scaler initialized")
    
    def _create_default_rules(self) -> List[ScalingRule]:
        """Create default scaling rules."""
        return [
            # CPU utilization rules
            ScalingRule(
                trigger=ScalingTrigger.CPU_UTILIZATION,
                threshold_up=75.0,
                threshold_down=25.0,
                direction=ScalingDirection.UP,
                cooldown_seconds=180.0
            ),
            
            # Memory utilization rules
            ScalingRule(
                trigger=ScalingTrigger.MEMORY_UTILIZATION,
                threshold_up=80.0,
                threshold_down=40.0,
                direction=ScalingDirection.UP,
                cooldown_seconds=120.0
            ),
            
            # Response time rules
            ScalingRule(
                trigger=ScalingTrigger.RESPONSE_TIME,
                threshold_up=2.0,  # 2 seconds
                threshold_down=0.5,  # 0.5 seconds
                direction=ScalingDirection.UP,
                cooldown_seconds=240.0
            ),
            
            # Error rate rules
            ScalingRule(
                trigger=ScalingTrigger.ERROR_RATE,
                threshold_up=5.0,  # 5% error rate
                threshold_down=1.0,  # 1% error rate
                direction=ScalingDirection.UP,
                cooldown_seconds=300.0
            )
        ]
    
    async def start_auto_scaling(self):
        """Start the auto-scaling monitoring loop."""
        if self.scaling_task and not self.scaling_task.done():
            self.logger.warning("Auto-scaling already running")
            return
        
        self.stop_event.clear()
        self.scaling_task = asyncio.create_task(self._scaling_loop())
        self.logger.info("Auto-scaling started")
    
    async def stop_auto_scaling(self):
        """Stop the auto-scaling monitoring loop."""
        if self.scaling_task:
            self.stop_event.set()
            try:
                await asyncio.wait_for(self.scaling_task, timeout=10.0)
            except asyncio.TimeoutError:
                self.scaling_task.cancel()
        
        self.logger.info("Auto-scaling stopped")
    
    async def _scaling_loop(self):
        """Main auto-scaling monitoring loop."""
        while not self.stop_event.is_set():
            try:
                if self.scaling_enabled:
                    await self._evaluate_scaling_decisions()
                
                if self.learning_enabled:
                    await self._learn_workload_patterns()
                
                if self.predictive_scaling:
                    await self._predictive_scaling_check()
                
                # Sleep for evaluation interval
                await asyncio.sleep(30.0)  # Evaluate every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in scaling loop: {e}")
                await asyncio.sleep(60.0)  # Wait longer on error
    
    async def _evaluate_scaling_decisions(self):
        """Evaluate current metrics and make scaling decisions."""
        current_metrics = await self._collect_current_metrics()
        
        for rule in self.scaling_rules:
            try:
                await self._evaluate_rule(rule, current_metrics)
            except Exception as e:
                self.logger.error(f"Error evaluating rule {rule.trigger.value}: {e}")
    
    async def _collect_current_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1.0)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Process metrics
        process = psutil.Process()
        process_cpu = process.cpu_percent()
        process_memory = process.memory_info().rss / (1024**2)  # MB
        
        # Application metrics from monitor
        monitor_summary = self.monitor.get_metrics_summary()
        
        # Calculate derived metrics
        avg_response_time = self._calculate_average_response_time()
        error_rate = self._calculate_error_rate()
        throughput = self._calculate_throughput()
        
        metrics = {
            "cpu_utilization": cpu_percent,
            "memory_utilization": memory_percent,
            "process_cpu": process_cpu,
            "process_memory_mb": process_memory,
            "response_time": avg_response_time,
            "error_rate": error_rate,
            "throughput": throughput,
            "available_memory_gb": memory.available / (1024**3)
        }
        
        # Store metrics history
        current_time = time.time()
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
            
            self.metrics_history[metric_name].append((current_time, value))
            
            # Keep only recent history (last 24 hours)
            cutoff_time = current_time - 86400
            self.metrics_history[metric_name] = [
                (t, v) for t, v in self.metrics_history[metric_name]
                if t > cutoff_time
            ]
        
        return metrics
    
    async def _evaluate_rule(self, rule: ScalingRule, metrics: Dict[str, float]):
        """Evaluate a specific scaling rule."""
        metric_name = rule.trigger.value
        current_value = metrics.get(metric_name, 0.0)
        
        # Check if we're in cooldown
        last_action_time = self.last_scaling_actions.get(metric_name)
        if last_action_time:
            time_since_last = (datetime.now() - last_action_time).total_seconds()
            if time_since_last < rule.cooldown_seconds:
                return
        
        # Get recent values for evaluation window
        recent_values = self._get_recent_values(metric_name, rule.evaluation_window)
        
        if len(recent_values) < rule.min_data_points:
            return  # Not enough data
        
        # Calculate average over evaluation window
        avg_value = sum(recent_values) / len(recent_values)
        
        # Determine scaling direction
        scaling_needed = None
        
        if avg_value > rule.threshold_up:
            scaling_needed = ScalingDirection.UP
        elif avg_value < rule.threshold_down:
            scaling_needed = ScalingDirection.DOWN
        
        if scaling_needed:
            await self._execute_scaling_action(rule.trigger, scaling_needed, avg_value)
            self.last_scaling_actions[metric_name] = datetime.now()
    
    def _get_recent_values(self, metric_name: str, window_seconds: float) -> List[float]:
        """Get recent values for a metric within the specified window."""
        if metric_name not in self.metrics_history:
            return []
        
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        return [
            value for timestamp, value in self.metrics_history[metric_name]
            if timestamp > cutoff_time
        ]
    
    async def _execute_scaling_action(self, trigger: ScalingTrigger, 
                                    direction: ScalingDirection, metric_value: float):
        """Execute a scaling action based on the trigger and direction."""
        actions = []
        
        if trigger == ScalingTrigger.CPU_UTILIZATION:
            actions.extend(await self._scale_for_cpu(direction, metric_value))
        
        elif trigger == ScalingTrigger.MEMORY_UTILIZATION:
            actions.extend(await self._scale_for_memory(direction, metric_value))
        
        elif trigger == ScalingTrigger.RESPONSE_TIME:
            actions.extend(await self._scale_for_response_time(direction, metric_value))
        
        elif trigger == ScalingTrigger.ERROR_RATE:
            actions.extend(await self._scale_for_error_rate(direction, metric_value))
        
        # Record and log actions
        for action in actions:
            self.scaling_history.append(action)
            self.logger.info(
                f"Scaling action executed: {action.action_type} {action.target} "
                f"from {action.old_value} to {action.new_value} - {action.reason}"
            )
    
    async def _scale_for_cpu(self, direction: ScalingDirection, 
                           cpu_value: float) -> List[ScalingAction]:
        """Scale resources based on CPU utilization."""
        actions = []
        
        if direction == ScalingDirection.UP:
            # Increase parallelism
            current_workers = self.optimizer.parallel_processor.max_workers
            new_workers = min(current_workers + 2, self.limits.max_workers)
            
            if new_workers != current_workers:
                self.optimizer.parallel_processor.max_workers = new_workers
                actions.append(ScalingAction(
                    action_type="scale_workers",
                    target="parallel_processor",
                    old_value=current_workers,
                    new_value=new_workers,
                    reason=f"High CPU utilization: {cpu_value:.1f}%"
                ))
            
            # Switch to more efficient processing mode
            if self.optimizer.config.processing_mode != ProcessingMode.PROCESS_POOL:
                old_mode = self.optimizer.config.processing_mode
                self.optimizer.config.processing_mode = ProcessingMode.PROCESS_POOL
                actions.append(ScalingAction(
                    action_type="change_processing_mode",
                    target="optimization_config",
                    old_value=old_mode.value,
                    new_value=ProcessingMode.PROCESS_POOL.value,
                    reason=f"High CPU utilization: {cpu_value:.1f}%"
                ))
        
        elif direction == ScalingDirection.DOWN:
            # Decrease workers to save resources
            current_workers = self.optimizer.parallel_processor.max_workers
            new_workers = max(current_workers - 1, self.limits.min_workers)
            
            if new_workers != current_workers:
                self.optimizer.parallel_processor.max_workers = new_workers
                actions.append(ScalingAction(
                    action_type="scale_workers",
                    target="parallel_processor",
                    old_value=current_workers,
                    new_value=new_workers,
                    reason=f"Low CPU utilization: {cpu_value:.1f}%"
                ))
        
        return actions
    
    async def _scale_for_memory(self, direction: ScalingDirection, 
                              memory_value: float) -> List[ScalingAction]:
        """Scale resources based on memory utilization."""
        actions = []
        
        if direction == ScalingDirection.UP:
            # Reduce cache size to free memory
            if self.optimizer.cache:
                current_size = self.optimizer.cache.max_size
                new_size = max(current_size // 2, self.limits.min_cache_size)
                
                if new_size != current_size:
                    self.optimizer.cache.max_size = new_size
                    # Clear some cache entries
                    self.optimizer.cache.clear()
                    
                    actions.append(ScalingAction(
                        action_type="reduce_cache_size",
                        target="cache",
                        old_value=current_size,
                        new_value=new_size,
                        reason=f"High memory utilization: {memory_value:.1f}%"
                    ))
            
            # Enable aggressive memory optimization
            self.optimizer.config.memory_threshold = min(
                self.optimizer.config.memory_threshold - 0.1,
                self.limits.min_memory_threshold
            )
            
            # Force memory optimization
            self.optimizer.memory_optimizer.optimize_memory()
        
        elif direction == ScalingDirection.DOWN:
            # Increase cache size for better performance
            if self.optimizer.cache:
                current_size = self.optimizer.cache.max_size
                new_size = min(current_size * 2, self.limits.max_cache_size)
                
                if new_size != current_size:
                    self.optimizer.cache.max_size = new_size
                    actions.append(ScalingAction(
                        action_type="increase_cache_size",
                        target="cache",
                        old_value=current_size,
                        new_value=new_size,
                        reason=f"Low memory utilization: {memory_value:.1f}%"
                    ))
        
        return actions
    
    async def _scale_for_response_time(self, direction: ScalingDirection, 
                                     response_time: float) -> List[ScalingAction]:
        """Scale resources based on response time."""
        actions = []
        
        if direction == ScalingDirection.UP:
            # Increase parallelism for faster processing
            current_workers = self.optimizer.parallel_processor.max_workers
            new_workers = min(current_workers + 1, self.limits.max_workers)
            
            if new_workers != current_workers:
                self.optimizer.parallel_processor.max_workers = new_workers
                actions.append(ScalingAction(
                    action_type="scale_workers",
                    target="parallel_processor",
                    old_value=current_workers,
                    new_value=new_workers,
                    reason=f"High response time: {response_time:.2f}s"
                ))
            
            # Enable more aggressive caching
            if self.optimizer.cache:
                current_ttl = self.optimizer.cache.ttl
                new_ttl = min(current_ttl * 1.5, 7200)  # Max 2 hours
                
                self.optimizer.cache.ttl = new_ttl
                actions.append(ScalingAction(
                    action_type="increase_cache_ttl",
                    target="cache",
                    old_value=current_ttl,
                    new_value=new_ttl,
                    reason=f"High response time: {response_time:.2f}s"
                ))
        
        return actions
    
    async def _scale_for_error_rate(self, direction: ScalingDirection, 
                                  error_rate: float) -> List[ScalingAction]:
        """Scale resources based on error rate."""
        actions = []
        
        if direction == ScalingDirection.UP:
            # Reduce load by decreasing parallelism
            current_workers = self.optimizer.parallel_processor.max_workers
            new_workers = max(current_workers - 1, self.limits.min_workers)
            
            if new_workers != current_workers:
                self.optimizer.parallel_processor.max_workers = new_workers
                actions.append(ScalingAction(
                    action_type="scale_workers",
                    target="parallel_processor",
                    old_value=current_workers,
                    new_value=new_workers,
                    reason=f"High error rate: {error_rate:.1f}%"
                ))
            
            # Switch to more conservative processing mode
            if self.optimizer.config.processing_mode == ProcessingMode.PROCESS_POOL:
                old_mode = self.optimizer.config.processing_mode
                self.optimizer.config.processing_mode = ProcessingMode.THREADED
                actions.append(ScalingAction(
                    action_type="change_processing_mode",
                    target="optimization_config",
                    old_value=old_mode.value,
                    new_value=ProcessingMode.THREADED.value,
                    reason=f"High error rate: {error_rate:.1f}%"
                ))
        
        return actions
    
    def _calculate_average_response_time(self) -> float:
        """Calculate average response time from recent operations."""
        recent_metrics = list(self.optimizer.metrics_history)[-10:]
        if not recent_metrics:
            return 0.0
        
        return sum(m.duration for m in recent_metrics) / len(recent_metrics)
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate from monitor."""
        error_summary = self.monitor.error_handler.get_error_summary()
        
        if error_summary["error_count"] == 0:
            return 0.0
        
        # Calculate error rate over last hour
        total_operations = sum(
            len(times) for times in self.optimizer.operation_profiles.values()
        )
        
        if total_operations == 0:
            return 0.0
        
        return (error_summary["error_count"] / total_operations) * 100
    
    def _calculate_throughput(self) -> float:
        """Calculate operations per second."""
        recent_metrics = list(self.optimizer.metrics_history)[-60:]  # Last minute
        if len(recent_metrics) < 2:
            return 0.0
        
        time_span = recent_metrics[-1].timestamp - recent_metrics[0].timestamp
        if time_span == 0:
            return 0.0
        
        return len(recent_metrics) / time_span
    
    async def _learn_workload_patterns(self):
        """Learn workload patterns for predictive scaling."""
        current_hour = datetime.now().hour
        current_day = datetime.now().weekday()
        
        # Pattern key: "hour_day" (e.g., "14_1" for 2 PM on Tuesday)
        pattern_key = f"{current_hour}_{current_day}"
        
        if pattern_key not in self.workload_patterns:
            self.workload_patterns[pattern_key] = []
        
        # Record current throughput
        current_throughput = self._calculate_throughput()
        self.workload_patterns[pattern_key].append(current_throughput)
        
        # Keep only recent patterns (last 4 weeks)
        max_samples = 28  # 4 weeks of daily samples
        if len(self.workload_patterns[pattern_key]) > max_samples:
            self.workload_patterns[pattern_key] = \
                self.workload_patterns[pattern_key][-max_samples:]
    
    async def _predictive_scaling_check(self):
        """Check if predictive scaling should be triggered."""
        # Predict load for next hour
        next_hour = (datetime.now() + timedelta(hours=1)).hour
        current_day = datetime.now().weekday()
        pattern_key = f"{next_hour}_{current_day}"
        
        if pattern_key in self.workload_patterns:
            historical_loads = self.workload_patterns[pattern_key]
            if len(historical_loads) >= 3:  # Need at least 3 data points
                predicted_load = sum(historical_loads) / len(historical_loads)
                current_load = self._calculate_throughput()
                
                # If predicted load is significantly higher, pre-scale
                if predicted_load > current_load * 1.5:
                    await self._pre_scale_for_predicted_load(predicted_load)
    
    async def _pre_scale_for_predicted_load(self, predicted_load: float):
        """Pre-scale resources based on predicted load."""
        self.logger.info(f"Pre-scaling for predicted load: {predicted_load:.2f} ops/sec")
        
        # Gradually increase workers
        current_workers = self.optimizer.parallel_processor.max_workers
        recommended_workers = min(
            current_workers + 1,
            self.limits.max_workers
        )
        
        if recommended_workers != current_workers:
            self.optimizer.parallel_processor.max_workers = recommended_workers
            
            action = ScalingAction(
                action_type="predictive_scale_workers",
                target="parallel_processor",
                old_value=current_workers,
                new_value=recommended_workers,
                reason=f"Predicted load increase: {predicted_load:.2f} ops/sec"
            )
            
            self.scaling_history.append(action)
            self.logger.info(f"Predictive scaling: {action.reason}")
    
    def get_scaling_summary(self) -> Dict[str, Any]:
        """Get comprehensive scaling summary."""
        return {
            "enabled": self.scaling_enabled,
            "learning_enabled": self.learning_enabled,
            "predictive_scaling": self.predictive_scaling,
            "resource_limits": {
                "min_workers": self.limits.min_workers,
                "max_workers": self.limits.max_workers,
                "min_cache_size": self.limits.min_cache_size,
                "max_cache_size": self.limits.max_cache_size
            },
            "current_config": {
                "workers": self.optimizer.parallel_processor.max_workers,
                "cache_size": self.optimizer.cache.max_size if self.optimizer.cache else None,
                "processing_mode": self.optimizer.config.processing_mode.value,
                "memory_threshold": self.optimizer.config.memory_threshold
            },
            "recent_actions": [
                {
                    "action": f"{action.action_type} {action.target}",
                    "change": f"{action.old_value} → {action.new_value}",
                    "reason": action.reason,
                    "timestamp": action.timestamp.isoformat()
                }
                for action in self.scaling_history[-10:]  # Last 10 actions
            ],
            "workload_patterns": {
                pattern: {
                    "samples": len(loads),
                    "avg_load": sum(loads) / len(loads) if loads else 0,
                    "max_load": max(loads) if loads else 0
                }
                for pattern, loads in self.workload_patterns.items()
            }
        }


# Global auto-scaler instance
_global_scaler: Optional[IntelligentAutoScaler] = None


def get_auto_scaler(limits: Optional[ResourceLimits] = None) -> IntelligentAutoScaler:
    """Get global auto-scaler instance."""
    global _global_scaler
    if _global_scaler is None:
        _global_scaler = IntelligentAutoScaler(limits)
    return _global_scaler