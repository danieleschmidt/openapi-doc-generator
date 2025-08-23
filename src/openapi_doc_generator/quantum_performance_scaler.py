"""
Quantum-Inspired Performance Scaler

This module implements advanced auto-scaling and performance optimization
using quantum-inspired algorithms for optimal resource utilization.
"""

import asyncio
import logging
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
import psutil
from threading import Lock

from .resilient_circuit_breaker import circuit_breaker, CircuitBreakerConfig


class ScalingStrategy(Enum):
    """Scaling strategies for resource optimization."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    QUANTUM_ANNEALED = "quantum_annealed"
    ADAPTIVE = "adaptive"


class ResourceType(Enum):
    """Types of resources to scale."""
    CPU_THREADS = "cpu_threads"
    MEMORY_CACHE = "memory_cache"
    IO_WORKERS = "io_workers"
    NETWORK_CONNECTIONS = "network_connections"


@dataclass
class PerformanceMetrics:
    """Performance metrics for scaling decisions."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    request_rate: float = 0.0
    response_time: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    queue_size: int = 0
    active_workers: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingConfig:
    """Configuration for auto-scaling behavior."""
    min_workers: int = 2
    max_workers: int = multiprocessing.cpu_count() * 4
    target_cpu_usage: float = 70.0
    target_memory_usage: float = 80.0
    scale_up_threshold: float = 85.0
    scale_down_threshold: float = 30.0
    scale_up_cooldown: float = 60.0  # seconds
    scale_down_cooldown: float = 120.0  # seconds
    metrics_window_size: int = 60
    quantum_temperature: float = 1.0
    quantum_cooling_rate: float = 0.98


class QuantumPerformanceScaler:
    """Advanced auto-scaler with quantum-inspired optimization."""
    
    def __init__(self, name: str, config: Optional[ScalingConfig] = None):
        self.name = name
        self.config = config or ScalingConfig()
        self.current_workers = self.config.min_workers
        self.metrics_history: List[PerformanceMetrics] = []
        self.last_scale_up = 0.0
        self.last_scale_down = 0.0
        self.lock = Lock()
        self.logger = logging.getLogger(f"QuantumScaler.{name}")
        
        # Quantum-inspired parameters
        self.quantum_state = 1.0
        self.temperature = self.config.quantum_temperature
        self.energy_level = 0.0
        
        # Executors for different workload types
        self.thread_executor: Optional[ThreadPoolExecutor] = None
        self.process_executor: Optional[ProcessPoolExecutor] = None
        
        self._initialize_executors()
    
    def _initialize_executors(self):
        """Initialize thread and process executors."""
        self.thread_executor = ThreadPoolExecutor(
            max_workers=self.current_workers,
            thread_name_prefix=f"QuantumScaler-{self.name}"
        )
        self.process_executor = ProcessPoolExecutor(
            max_workers=max(2, self.current_workers // 2)
        )
    
    def submit_task(self, func: Callable, *args, use_processes=False, **kwargs):
        """Submit task for execution with auto-scaling."""
        executor = self.process_executor if use_processes else self.thread_executor
        
        # Record metrics before execution
        self._record_current_metrics()
        
        # Check if scaling is needed
        self._evaluate_scaling_decision()
        
        # Submit task with circuit breaker protection
        @circuit_breaker(f"scaler_{self.name}", CircuitBreakerConfig(timeout=300.0))
        def protected_task():
            return func(*args, **kwargs)
        
        future = executor.submit(protected_task)
        return future
    
    async def submit_async_batch(self, tasks: List[Callable], batch_size: Optional[int] = None) -> List[Any]:
        """Submit batch of async tasks with optimal scaling."""
        if not tasks:
            return []
        
        batch_size = batch_size or min(len(tasks), self.current_workers)
        results = []
        
        # Process in batches to avoid overwhelming the system
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            
            # Submit batch with concurrent execution
            batch_futures = [
                asyncio.create_task(self._execute_with_scaling(task))
                for task in batch
            ]
            
            # Wait for batch completion
            batch_results = await asyncio.gather(*batch_futures, return_exceptions=True)
            results.extend(batch_results)
            
            # Brief pause between batches to allow system adjustment
            if i + batch_size < len(tasks):
                await asyncio.sleep(0.1)
        
        return results
    
    async def _execute_with_scaling(self, task: Callable) -> Any:
        """Execute single task with scaling awareness."""
        start_time = time.time()
        
        try:
            # Execute task
            if asyncio.iscoroutinefunction(task):
                result = await task()
            else:
                result = task()
            
            # Record success metrics
            execution_time = time.time() - start_time
            self._update_performance_metrics(execution_time, success=True)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_performance_metrics(execution_time, success=False)
            raise
    
    def _record_current_metrics(self):
        """Record current system performance metrics."""
        try:
            metrics = PerformanceMetrics(
                cpu_usage=psutil.cpu_percent(interval=0.1),
                memory_usage=psutil.virtual_memory().percent,
                active_workers=self.current_workers,
                queue_size=getattr(self.thread_executor, '_work_queue', {}).qsize() or 0
            )
            
            with self.lock:
                self.metrics_history.append(metrics)
                
                # Keep only recent metrics
                if len(self.metrics_history) > self.config.metrics_window_size:
                    self.metrics_history = self.metrics_history[-self.config.metrics_window_size:]
        
        except Exception as e:
            self.logger.warning(f"Failed to record metrics: {e}")
    
    def _update_performance_metrics(self, execution_time: float, success: bool):
        """Update performance metrics after task execution."""
        with self.lock:
            if self.metrics_history:
                latest = self.metrics_history[-1]
                latest.response_time = execution_time
                latest.error_rate = 0.0 if success else 1.0
                latest.throughput += 1.0  # Tasks per second (simplified)
    
    def _evaluate_scaling_decision(self):
        """Evaluate if scaling up or down is needed using quantum-inspired algorithm."""
        if not self.metrics_history:
            return
        
        with self.lock:
            current_time = time.time()
            recent_metrics = self.metrics_history[-min(10, len(self.metrics_history)):]
            
            if not recent_metrics:
                return
            
            # Calculate average metrics
            avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
            avg_queue_size = sum(m.queue_size for m in recent_metrics) / len(recent_metrics)
            
            # Quantum-inspired decision making
            self._update_quantum_state(avg_cpu, avg_memory, avg_queue_size)
            
            # Scale up conditions
            should_scale_up = (
                (avg_cpu > self.config.scale_up_threshold or 
                 avg_memory > self.config.scale_up_threshold or
                 avg_queue_size > self.current_workers * 2) and
                current_time - self.last_scale_up > self.config.scale_up_cooldown and
                self.current_workers < self.config.max_workers and
                self.quantum_state > 0.7  # Quantum confidence threshold
            )
            
            # Scale down conditions
            should_scale_down = (
                avg_cpu < self.config.scale_down_threshold and
                avg_memory < self.config.scale_down_threshold and
                avg_queue_size < self.current_workers * 0.5 and
                current_time - self.last_scale_down > self.config.scale_down_cooldown and
                self.current_workers > self.config.min_workers and
                self.quantum_state < 0.3  # Low quantum activity
            )
            
            if should_scale_up:
                self._scale_up()
            elif should_scale_down:
                self._scale_down()
    
    def _update_quantum_state(self, cpu_usage: float, memory_usage: float, queue_size: float):
        """Update quantum state based on system metrics."""
        # Quantum-inspired energy calculation
        system_pressure = (cpu_usage + memory_usage) / 200.0  # Normalize to [0,1]
        queue_pressure = min(queue_size / (self.current_workers * 4), 1.0)
        
        # Calculate energy change (higher pressure = higher energy)
        energy_change = (system_pressure + queue_pressure) / 2.0
        self.energy_level = min(max(self.energy_level + energy_change - 0.1, 0.0), 2.0)
        
        # Quantum state evolution with cooling
        if self.energy_level > 1.0:
            self.quantum_state = min(self.quantum_state * 1.1, 1.0)  # Excitation
        else:
            self.quantum_state *= self.config.quantum_cooling_rate  # Cooling
        
        # Temperature cooling
        self.temperature *= self.config.quantum_cooling_rate
        self.temperature = max(self.temperature, 0.01)
    
    def _scale_up(self):
        """Scale up resources using quantum-inspired optimization."""
        old_workers = self.current_workers
        
        # Quantum-inspired scaling factor
        quantum_factor = 1.0 + (self.quantum_state * self.temperature)
        scale_factor = min(quantum_factor, 2.0)  # Limit aggressive scaling
        
        new_workers = min(
            int(self.current_workers * scale_factor),
            self.config.max_workers
        )
        
        if new_workers > self.current_workers:
            self.current_workers = new_workers
            self.last_scale_up = time.time()
            
            # Restart executors with new worker count
            self._restart_executors()
            
            self.logger.info(
                f"Scaled up {self.name}: {old_workers} → {new_workers} workers "
                f"(quantum_state={self.quantum_state:.3f}, temp={self.temperature:.3f})"
            )
    
    def _scale_down(self):
        """Scale down resources conservatively."""
        old_workers = self.current_workers
        
        # Conservative scaling down (reduce by 1 or small percentage)
        new_workers = max(
            self.current_workers - 1,
            int(self.current_workers * 0.8),
            self.config.min_workers
        )
        
        if new_workers < self.current_workers:
            self.current_workers = new_workers
            self.last_scale_down = time.time()
            
            # Restart executors with new worker count
            self._restart_executors()
            
            self.logger.info(
                f"Scaled down {self.name}: {old_workers} → {new_workers} workers"
            )
    
    def _restart_executors(self):
        """Restart executors with new worker configuration."""
        try:
            # Gracefully shutdown old executors
            if self.thread_executor:
                self.thread_executor.shutdown(wait=False)
            if self.process_executor:
                self.process_executor.shutdown(wait=False)
            
            # Create new executors
            self.thread_executor = ThreadPoolExecutor(
                max_workers=self.current_workers,
                thread_name_prefix=f"QuantumScaler-{self.name}"
            )
            self.process_executor = ProcessPoolExecutor(
                max_workers=max(2, self.current_workers // 2)
            )
        
        except Exception as e:
            self.logger.error(f"Failed to restart executors: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        with self.lock:
            if not self.metrics_history:
                return {"status": "no_metrics", "current_workers": self.current_workers}
            
            recent_metrics = self.metrics_history[-min(10, len(self.metrics_history)):]
            
            avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
            avg_response_time = sum(m.response_time for m in recent_metrics) / len(recent_metrics)
            avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics)
            
            return {
                "scaler_name": self.name,
                "current_workers": self.current_workers,
                "quantum_state": self.quantum_state,
                "temperature": self.temperature,
                "energy_level": self.energy_level,
                "performance": {
                    "avg_cpu_usage": avg_cpu,
                    "avg_memory_usage": avg_memory,
                    "avg_response_time": avg_response_time,
                    "avg_throughput": avg_throughput,
                },
                "scaling_history": {
                    "last_scale_up": self.last_scale_up,
                    "last_scale_down": self.last_scale_down,
                },
                "config": {
                    "min_workers": self.config.min_workers,
                    "max_workers": self.config.max_workers,
                    "target_cpu": self.config.target_cpu_usage,
                    "scale_up_threshold": self.config.scale_up_threshold,
                    "scale_down_threshold": self.config.scale_down_threshold,
                },
                "metrics_count": len(self.metrics_history),
                "timestamp": time.time()
            }
    
    def shutdown(self):
        """Gracefully shutdown the scaler."""
        try:
            if self.thread_executor:
                self.thread_executor.shutdown(wait=True)
            if self.process_executor:
                self.process_executor.shutdown(wait=True)
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


class QuantumScalerManager:
    """Manages multiple quantum performance scalers."""
    
    def __init__(self):
        self.scalers: Dict[str, QuantumPerformanceScaler] = {}
        self.lock = Lock()
    
    def get_scaler(self, name: str, config: Optional[ScalingConfig] = None) -> QuantumPerformanceScaler:
        """Get or create quantum scaler by name."""
        with self.lock:
            if name not in self.scalers:
                self.scalers[name] = QuantumPerformanceScaler(name, config)
            return self.scalers[name]
    
    def get_all_reports(self) -> Dict[str, Dict[str, Any]]:
        """Get performance reports for all scalers."""
        with self.lock:
            return {
                name: scaler.get_performance_report()
                for name, scaler in self.scalers.items()
            }
    
    def shutdown_all(self):
        """Shutdown all scalers gracefully."""
        with self.lock:
            for scaler in self.scalers.values():
                scaler.shutdown()


# Global scaler manager
_global_scaler_manager: Optional[QuantumScalerManager] = None


def get_quantum_scaler_manager() -> QuantumScalerManager:
    """Get global quantum scaler manager."""
    global _global_scaler_manager
    if _global_scaler_manager is None:
        _global_scaler_manager = QuantumScalerManager()
    return _global_scaler_manager


def quantum_scaled(name: str, config: Optional[ScalingConfig] = None):
    """Decorator for quantum-scaled function execution."""
    def decorator(func):
        scaler = get_quantum_scaler_manager().get_scaler(name, config)
        
        def wrapper(*args, **kwargs):
            future = scaler.submit_task(func, *args, **kwargs)
            return future.result()
        
        wrapper.__name__ = f"quantum_scaled_{func.__name__}"
        return wrapper
    return decorator