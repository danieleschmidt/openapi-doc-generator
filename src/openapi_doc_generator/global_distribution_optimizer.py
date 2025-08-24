"""
Global Distribution Optimization System

This module implements advanced cross-region distribution capabilities with
edge caching, geographic optimization, CDN integration, and global deployment
strategies for optimal performance across worldwide deployments.

Features:
- Multi-region deployment coordination
- Edge caching with intelligent cache placement
- Geographic latency optimization
- CDN integration and edge computing
- Global load balancing and failover
- Regional auto-scaling coordination
- Cross-region data synchronization
- Disaster recovery and resilience
"""

import asyncio
import hashlib
import json
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from .intelligent_load_balancer import (
    IntelligentLoadBalancer, InstanceConfiguration, 
    LoadBalancingAlgorithm, LoadBalancingMetrics
)
from .predictive_auto_scaler import PredictiveAutoScaler
from .resource_pool_optimizer import ResourcePoolManager
from .enhanced_monitoring import get_monitor

logger = logging.getLogger(__name__)


class Region(Enum):
    """Global regions for deployment."""
    US_EAST_1 = "us-east-1"
    US_WEST_1 = "us-west-1"
    EU_CENTRAL_1 = "eu-central-1"
    EU_WEST_1 = "eu-west-1"
    ASIA_PACIFIC_1 = "asia-pacific-1"
    ASIA_PACIFIC_2 = "asia-pacific-2"
    AUSTRALIA_1 = "australia-1"
    SOUTH_AMERICA_1 = "south-america-1"
    AFRICA_1 = "africa-1"
    MIDDLE_EAST_1 = "middle-east-1"


class EdgeLocation(Enum):
    """Edge computing locations."""
    CDN_EDGE = "cdn_edge"
    COMPUTE_EDGE = "compute_edge"
    CACHE_EDGE = "cache_edge"
    HYBRID_EDGE = "hybrid_edge"


class CacheStrategy(Enum):
    """Edge caching strategies."""
    LEAST_RECENTLY_USED = "lru"
    MOST_FREQUENTLY_USED = "mfu"
    GEOGRAPHIC_PROXIMITY = "geographic"
    PREDICTIVE_CACHING = "predictive"
    HYBRID_INTELLIGENT = "hybrid_intelligent"


class SyncStrategy(Enum):
    """Data synchronization strategies."""
    EVENTUAL_CONSISTENCY = "eventual_consistency"
    STRONG_CONSISTENCY = "strong_consistency"
    CAUSAL_CONSISTENCY = "causal_consistency"
    SESSION_CONSISTENCY = "session_consistency"
    QUANTUM_CONSISTENCY = "quantum_consistency"


@dataclass
class GeographicCoordinates:
    """Geographic coordinates for latency calculations."""
    latitude: float
    longitude: float
    
    def distance_to(self, other: 'GeographicCoordinates') -> float:
        """Calculate great circle distance in kilometers."""
        # Haversine formula
        R = 6371  # Earth's radius in km
        
        lat1_rad = math.radians(self.latitude)
        lat2_rad = math.radians(other.latitude)
        dlat_rad = math.radians(other.latitude - self.latitude)
        dlon_rad = math.radians(other.longitude - self.longitude)
        
        a = (math.sin(dlat_rad/2) * math.sin(dlat_rad/2) +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(dlon_rad/2) * math.sin(dlon_rad/2))
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        return distance


@dataclass
class RegionInfo:
    """Information about a deployment region."""
    region: Region
    name: str
    coordinates: GeographicCoordinates
    availability_zones: List[str]
    cost_factor: float = 1.0  # Relative cost multiplier
    compliance_zones: List[str] = field(default_factory=list)
    max_latency_ms: float = 200.0
    preferred_regions: List[Region] = field(default_factory=list)


@dataclass
class EdgeNode:
    """Edge computing node configuration."""
    node_id: str
    location: EdgeLocation
    region: Region
    coordinates: GeographicCoordinates
    capacity_cpu: float
    capacity_memory_gb: float
    capacity_storage_gb: float
    current_load: float = 0.0
    health_status: str = "healthy"
    supported_operations: List[str] = field(default_factory=list)
    cache_hit_ratio: float = 0.0
    average_response_time_ms: float = 0.0


@dataclass
class CacheItem:
    """Cached item with geographic and temporal metadata."""
    key: str
    value: Any
    size_bytes: int
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    geographic_relevance: Dict[Region, float] = field(default_factory=dict)
    popularity_score: float = 0.0
    ttl_seconds: Optional[int] = None
    
    def is_expired(self) -> bool:
        """Check if cache item has expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds


@dataclass
class GlobalRequest:
    """Request with global context."""
    request_id: str
    client_region: Region
    client_coordinates: Optional[GeographicCoordinates]
    operation: str
    data_size_bytes: int
    latency_requirement_ms: float
    consistency_requirement: SyncStrategy
    cache_preference: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Global routing decision."""
    request_id: str
    selected_region: Region
    selected_edge_nodes: List[str]
    routing_reasons: List[str]
    expected_latency_ms: float
    cache_strategy: CacheStrategy
    fallback_regions: List[Region]
    cost_estimate: float
    timestamp: datetime = field(default_factory=datetime.now)


class GlobalRegionManager:
    """Manages global region information and routing decisions."""
    
    def __init__(self):
        self.regions: Dict[Region, RegionInfo] = {}
        self.region_latencies: Dict[Tuple[Region, Region], float] = {}
        self._initialize_regions()
        self._calculate_inter_region_latencies()
        
        logger.info("Global Region Manager initialized")
    
    def _initialize_regions(self):
        """Initialize global region information."""
        self.regions = {
            Region.US_EAST_1: RegionInfo(
                region=Region.US_EAST_1,
                name="US East (N. Virginia)",
                coordinates=GeographicCoordinates(38.13, -78.45),
                availability_zones=["us-east-1a", "us-east-1b", "us-east-1c"],
                cost_factor=1.0,
                compliance_zones=["US", "GDPR"]
            ),
            Region.US_WEST_1: RegionInfo(
                region=Region.US_WEST_1,
                name="US West (N. California)",
                coordinates=GeographicCoordinates(37.35, -121.96),
                availability_zones=["us-west-1a", "us-west-1b"],
                cost_factor=1.1,
                compliance_zones=["US", "CCPA"]
            ),
            Region.EU_CENTRAL_1: RegionInfo(
                region=Region.EU_CENTRAL_1,
                name="Europe (Frankfurt)",
                coordinates=GeographicCoordinates(50.11, 8.68),
                availability_zones=["eu-central-1a", "eu-central-1b", "eu-central-1c"],
                cost_factor=1.05,
                compliance_zones=["EU", "GDPR"]
            ),
            Region.EU_WEST_1: RegionInfo(
                region=Region.EU_WEST_1,
                name="Europe (Ireland)",
                coordinates=GeographicCoordinates(53.33, -6.25),
                availability_zones=["eu-west-1a", "eu-west-1b", "eu-west-1c"],
                cost_factor=1.03,
                compliance_zones=["EU", "GDPR"]
            ),
            Region.ASIA_PACIFIC_1: RegionInfo(
                region=Region.ASIA_PACIFIC_1,
                name="Asia Pacific (Singapore)",
                coordinates=GeographicCoordinates(1.37, 103.8),
                availability_zones=["ap-1a", "ap-1b"],
                cost_factor=1.15,
                compliance_zones=["APAC"]
            ),
            Region.ASIA_PACIFIC_2: RegionInfo(
                region=Region.ASIA_PACIFIC_2,
                name="Asia Pacific (Tokyo)",
                coordinates=GeographicCoordinates(35.68, 139.69),
                availability_zones=["ap-2a", "ap-2b", "ap-2c"],
                cost_factor=1.2,
                compliance_zones=["APAC", "Japan"]
            ),
            Region.AUSTRALIA_1: RegionInfo(
                region=Region.AUSTRALIA_1,
                name="Australia (Sydney)",
                coordinates=GeographicCoordinates(-33.87, 151.21),
                availability_zones=["au-1a", "au-1b"],
                cost_factor=1.18,
                compliance_zones=["APAC", "Australia"]
            ),
            Region.SOUTH_AMERICA_1: RegionInfo(
                region=Region.SOUTH_AMERICA_1,
                name="South America (São Paulo)",
                coordinates=GeographicCoordinates(-23.55, -46.64),
                availability_zones=["sa-1a", "sa-1b"],
                cost_factor=1.12,
                compliance_zones=["SA", "Brazil"]
            )
        }
    
    def _calculate_inter_region_latencies(self):
        """Calculate expected latencies between regions."""
        for region1, info1 in self.regions.items():
            for region2, info2 in self.regions.items():
                if region1 == region2:
                    latency = 5.0  # Intra-region latency
                else:
                    # Calculate based on distance (simplified model)
                    distance_km = info1.coordinates.distance_to(info2.coordinates)
                    # Base latency: ~1ms per 100km + processing overhead
                    latency = max(20.0, distance_km / 100.0 + 20.0)
                
                self.region_latencies[(region1, region2)] = latency
    
    def get_region_latency(self, from_region: Region, to_region: Region) -> float:
        """Get expected latency between regions."""
        return self.region_latencies.get((from_region, to_region), 200.0)
    
    def find_closest_regions(self, 
                           client_coordinates: GeographicCoordinates,
                           max_regions: int = 3) -> List[Tuple[Region, float]]:
        """Find closest regions to client coordinates."""
        region_distances = []
        
        for region, info in self.regions.items():
            distance = client_coordinates.distance_to(info.coordinates)
            region_distances.append((region, distance))
        
        # Sort by distance and return top regions
        region_distances.sort(key=lambda x: x[1])
        return region_distances[:max_regions]
    
    def get_compliance_compatible_regions(self, 
                                        compliance_requirements: List[str]) -> List[Region]:
        """Get regions compatible with compliance requirements."""
        compatible_regions = []
        
        for region, info in self.regions.items():
            if any(req in info.compliance_zones for req in compliance_requirements):
                compatible_regions.append(region)
        
        return compatible_regions


class IntelligentEdgeCache:
    """Intelligent edge caching system with global optimization."""
    
    def __init__(self, 
                 max_size_gb: float = 10.0,
                 strategy: CacheStrategy = CacheStrategy.HYBRID_INTELLIGENT):
        
        self.max_size_gb = max_size_gb
        self.strategy = strategy
        self.current_size_bytes = 0
        
        # Cache storage
        self.cache: Dict[str, CacheItem] = {}
        self.access_history: deque = deque(maxlen=10000)
        
        # Geographic optimization
        self.regional_popularity: Dict[Region, Dict[str, float]] = defaultdict(dict)
        self.geographic_access_patterns: Dict[str, Dict[Region, int]] = defaultdict(lambda: defaultdict(int))
        
        # Predictive caching
        self.access_predictions: Dict[str, float] = {}
        self.prefetch_queue: deque = deque()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'prefetches': 0,
            'bytes_served': 0
        }
        
        logger.info(f"Edge cache initialized: {max_size_gb}GB, strategy={strategy.value}")
    
    async def get(self, key: str, client_region: Region) -> Optional[Any]:
        """Get item from cache with geographic context."""
        if key in self.cache:
            cache_item = self.cache[key]
            
            # Check expiration
            if cache_item.is_expired():
                await self._evict_item(key)
                self.stats['misses'] += 1
                return None
            
            # Update access patterns
            cache_item.last_accessed = datetime.now()
            cache_item.access_count += 1
            self.geographic_access_patterns[key][client_region] += 1
            
            # Update popularity score
            self._update_popularity_score(cache_item, client_region)
            
            # Record access
            self.access_history.append({
                'timestamp': datetime.now(),
                'key': key,
                'region': client_region,
                'operation': 'hit'
            })
            
            self.stats['hits'] += 1
            self.stats['bytes_served'] += cache_item.size_bytes
            
            return cache_item.value
        
        else:
            self.stats['misses'] += 1
            self.access_history.append({
                'timestamp': datetime.now(),
                'key': key,
                'region': client_region,
                'operation': 'miss'
            })
            
            return None
    
    async def put(self, 
                  key: str, 
                  value: Any, 
                  client_region: Region,
                  size_bytes: int,
                  ttl_seconds: Optional[int] = None):
        """Put item in cache with geographic optimization."""
        
        # Check if we need to evict items
        while (self.current_size_bytes + size_bytes) > (self.max_size_gb * 1024 * 1024 * 1024):
            evicted = await self._evict_items(size_bytes)
            if not evicted:
                # Cannot make space, reject cache
                logger.warning(f"Cannot cache item {key}: insufficient space")
                return False
        
        # Create cache item
        cache_item = CacheItem(
            key=key,
            value=value,
            size_bytes=size_bytes,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            geographic_relevance={client_region: 1.0},
            ttl_seconds=ttl_seconds
        )
        
        # Store item
        self.cache[key] = cache_item
        self.current_size_bytes += size_bytes
        
        # Initialize geographic patterns
        self.geographic_access_patterns[key][client_region] = 1
        
        logger.debug(f"Cached item {key} ({size_bytes} bytes) for region {client_region.value}")
        return True
    
    async def invalidate(self, key: str):
        """Invalidate cached item."""
        if key in self.cache:
            await self._evict_item(key)
            logger.debug(f"Invalidated cache item: {key}")
    
    def _update_popularity_score(self, cache_item: CacheItem, client_region: Region):
        """Update popularity score based on access patterns."""
        # Time-based decay
        age_hours = (datetime.now() - cache_item.created_at).total_seconds() / 3600.0
        age_factor = math.exp(-age_hours / 24.0)  # Decay over 24 hours
        
        # Access frequency factor
        frequency_factor = cache_item.access_count / max(1, age_hours)
        
        # Geographic relevance factor
        geographic_factor = cache_item.geographic_relevance.get(client_region, 0.1)
        
        # Combined popularity score
        cache_item.popularity_score = (
            age_factor * 0.3 +
            frequency_factor * 0.5 +
            geographic_factor * 0.2
        )
        
        # Update geographic relevance
        cache_item.geographic_relevance[client_region] = min(1.0, 
            cache_item.geographic_relevance.get(client_region, 0.0) + 0.1
        )
    
    async def _evict_items(self, bytes_needed: int) -> bool:
        """Evict items to make space."""
        if not self.cache:
            return False
        
        if self.strategy == CacheStrategy.LEAST_RECENTLY_USED:
            items_to_evict = sorted(
                self.cache.values(),
                key=lambda x: x.last_accessed
            )
        
        elif self.strategy == CacheStrategy.MOST_FREQUENTLY_USED:
            # Evict least frequently used (inverse of MFU)
            items_to_evict = sorted(
                self.cache.values(),
                key=lambda x: x.access_count
            )
        
        elif self.strategy == CacheStrategy.GEOGRAPHIC_PROXIMITY:
            # Evict items with low geographic relevance
            items_to_evict = sorted(
                self.cache.values(),
                key=lambda x: sum(x.geographic_relevance.values())
            )
        
        elif self.strategy == CacheStrategy.PREDICTIVE_CACHING:
            # Evict items with low predicted future access
            items_to_evict = sorted(
                self.cache.values(),
                key=lambda x: self.access_predictions.get(x.key, 0.0)
            )
        
        else:  # HYBRID_INTELLIGENT
            # Multi-factor scoring
            items_to_evict = sorted(
                self.cache.values(),
                key=lambda x: x.popularity_score
            )
        
        # Evict items until we have enough space
        bytes_freed = 0
        evicted_items = []
        
        for item in items_to_evict:
            if bytes_freed >= bytes_needed:
                break
            
            bytes_freed += item.size_bytes
            evicted_items.append(item.key)
        
        # Actually evict the items
        for key in evicted_items:
            await self._evict_item(key)
        
        return bytes_freed >= bytes_needed
    
    async def _evict_item(self, key: str):
        """Evict a specific item."""
        if key in self.cache:
            cache_item = self.cache[key]
            self.current_size_bytes -= cache_item.size_bytes
            del self.cache[key]
            self.stats['evictions'] += 1
            
            # Clean up tracking data
            if key in self.geographic_access_patterns:
                del self.geographic_access_patterns[key]
            if key in self.access_predictions:
                del self.access_predictions[key]
    
    async def predict_access_patterns(self):
        """Predict future access patterns for proactive caching."""
        # Analyze access history
        recent_accesses = list(self.access_history)[-1000:]  # Last 1000 accesses
        
        # Group by key and analyze patterns
        key_patterns = defaultdict(list)
        for access in recent_accesses:
            key_patterns[access['key']].append(access['timestamp'])
        
        # Predict future access likelihood
        current_time = datetime.now()
        
        for key, access_times in key_patterns.items():
            if len(access_times) < 2:
                continue
            
            # Calculate access intervals
            intervals = []
            for i in range(1, len(access_times)):
                interval = (access_times[i] - access_times[i-1]).total_seconds()
                intervals.append(interval)
            
            if intervals:
                # Predict next access based on average interval
                avg_interval = sum(intervals) / len(intervals)
                last_access = access_times[-1]
                expected_next_access = last_access + timedelta(seconds=avg_interval)
                
                # Calculate likelihood based on time until expected access
                time_until_access = (expected_next_access - current_time).total_seconds()
                likelihood = max(0.0, 1.0 - abs(time_until_access) / 3600.0)  # Decay over 1 hour
                
                self.access_predictions[key] = likelihood
    
    async def prefetch_content(self, predictions: Dict[str, Any]):
        """Prefetch content based on predictions."""
        for key, predicted_content in predictions.items():
            if key not in self.cache:
                # Add to prefetch queue
                self.prefetch_queue.append({
                    'key': key,
                    'content': predicted_content,
                    'priority': self.access_predictions.get(key, 0.0)
                })
        
        # Process prefetch queue
        while self.prefetch_queue:
            prefetch_item = self.prefetch_queue.popleft()
            
            # Estimate region based on predictions (simplified)
            client_region = Region.US_EAST_1  # Default region
            
            size_bytes = len(str(prefetch_item['content']))
            success = await self.put(
                prefetch_item['key'],
                prefetch_item['content'],
                client_region,
                size_bytes
            )
            
            if success:
                self.stats['prefetches'] += 1
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        hit_rate = self.stats['hits'] / max(self.stats['hits'] + self.stats['misses'], 1) * 100
        
        return {
            'strategy': self.strategy.value,
            'max_size_gb': self.max_size_gb,
            'current_size_gb': self.current_size_bytes / (1024 * 1024 * 1024),
            'utilization': (self.current_size_bytes / (self.max_size_gb * 1024 * 1024 * 1024)) * 100,
            'item_count': len(self.cache),
            'hit_rate': hit_rate,
            'total_hits': self.stats['hits'],
            'total_misses': self.stats['misses'],
            'total_evictions': self.stats['evictions'],
            'total_prefetches': self.stats['prefetches'],
            'bytes_served': self.stats['bytes_served'],
            'geographic_distribution': {
                region.value: count for region, patterns in self.regional_popularity.items()
                for pattern, count in patterns.items()
            }
        }


class GlobalDistributionOptimizer:
    """
    Main orchestrator for global distribution optimization.
    """
    
    def __init__(self):
        # Core components
        self.region_manager = GlobalRegionManager()
        self.edge_caches: Dict[Region, IntelligentEdgeCache] = {}
        self.load_balancers: Dict[Region, IntelligentLoadBalancer] = {}
        self.auto_scalers: Dict[Region, PredictiveAutoScaler] = {}
        
        # Edge nodes
        self.edge_nodes: Dict[str, EdgeNode] = {}
        
        # Global coordination
        self.global_request_router = GlobalRequestRouter(self.region_manager)
        
        # Monitoring and optimization
        self.monitor = get_monitor()
        
        # Performance tracking
        self.routing_history: deque = deque(maxlen=10000)
        self.performance_metrics: Dict[Region, Dict[str, float]] = defaultdict(dict)
        
        # Background tasks
        self.optimization_task: Optional[asyncio.Task] = None
        self.sync_task: Optional[asyncio.Task] = None
        self.running = False
        
        logger.info("Global Distribution Optimizer initialized")
    
    async def start(self):
        """Start the global distribution optimizer."""
        if self.running:
            return
        
        self.running = True
        
        # Initialize edge caches for all regions
        for region in self.region_manager.regions.keys():
            self.edge_caches[region] = IntelligentEdgeCache(
                max_size_gb=5.0,
                strategy=CacheStrategy.HYBRID_INTELLIGENT
            )
        
        # Start background tasks
        self.optimization_task = asyncio.create_task(self._global_optimization_loop())
        self.sync_task = asyncio.create_task(self._synchronization_loop())
        
        logger.info("Global Distribution Optimizer started")
    
    async def stop(self):
        """Stop the global distribution optimizer."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel background tasks
        tasks = [self.optimization_task, self.sync_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Stop regional components
        for scaler in self.auto_scalers.values():
            await scaler.stop()
        
        for lb in self.load_balancers.values():
            await lb.stop()
        
        logger.info("Global Distribution Optimizer stopped")
    
    async def route_request(self, request: GlobalRequest) -> RoutingDecision:
        """Route a global request to the optimal region and edge nodes."""
        return await self.global_request_router.route_request(
            request, self.edge_caches, self.edge_nodes
        )
    
    async def add_region(self, region: Region, instances: List[InstanceConfiguration]):
        """Add a new region with instances."""
        if region not in self.load_balancers:
            # Create load balancer for region
            from .intelligent_load_balancer import create_intelligent_load_balancer
            
            lb = await create_intelligent_load_balancer(
                algorithm=LoadBalancingAlgorithm.HYBRID_ADAPTIVE,
                instances=instances
            )
            self.load_balancers[region] = lb
            
            # Create auto-scaler for region
            scaler = PredictiveAutoScaler(
                min_instances=len(instances),
                max_instances=len(instances) * 5
            )
            await scaler.start()
            self.auto_scalers[region] = scaler
            
            logger.info(f"Added region {region.value} with {len(instances)} instances")
    
    async def add_edge_node(self, edge_node: EdgeNode):
        """Add an edge computing node."""
        self.edge_nodes[edge_node.node_id] = edge_node
        logger.info(f"Added edge node {edge_node.node_id} in {edge_node.region.value}")
    
    async def cache_content(self, 
                          key: str, 
                          content: Any, 
                          regions: List[Region],
                          size_bytes: int,
                          ttl_seconds: Optional[int] = None):
        """Cache content across multiple regions."""
        success_count = 0
        
        for region in regions:
            if region in self.edge_caches:
                cache = self.edge_caches[region]
                success = await cache.put(
                    key, content, region, size_bytes, ttl_seconds
                )
                if success:
                    success_count += 1
        
        return success_count
    
    async def get_cached_content(self, key: str, client_region: Region) -> Optional[Any]:
        """Get cached content from the optimal region."""
        # Try local region first
        if client_region in self.edge_caches:
            content = await self.edge_caches[client_region].get(key, client_region)
            if content is not None:
                return content
        
        # Try nearby regions
        closest_regions = self.region_manager.find_closest_regions(
            self.region_manager.regions[client_region].coordinates, max_regions=3
        )
        
        for region, _ in closest_regions:
            if region != client_region and region in self.edge_caches:
                content = await self.edge_caches[region].get(key, client_region)
                if content is not None:
                    # Cache in client region for future access
                    await self.edge_caches[client_region].put(
                        key, content, client_region, len(str(content))
                    )
                    return content
        
        return None
    
    async def _global_optimization_loop(self):
        """Global optimization and coordination loop."""
        while self.running:
            try:
                await self._perform_global_optimization()
                await asyncio.sleep(60)  # Optimize every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Global optimization error: {e}")
                await asyncio.sleep(120)
    
    async def _perform_global_optimization(self):
        """Perform global optimization across all regions."""
        # Collect performance metrics from all regions
        global_metrics = {}
        
        for region, lb in self.load_balancers.items():
            lb_stats = lb.get_load_balancing_stats()
            cache_stats = self.edge_caches[region].get_cache_stats()
            
            global_metrics[region.value] = {
                'load_balancer': lb_stats,
                'edge_cache': cache_stats,
                'avg_response_time': lb_stats.get('recent_decisions', [{}])[-1:][0].get('confidence', 0) * 100,
            }
        
        # Optimize cache distribution
        await self._optimize_cache_distribution(global_metrics)
        
        # Optimize edge node placement
        await self._optimize_edge_nodes(global_metrics)
        
        # Record global metrics
        self.monitor.record_metric("global_regions_active", float(len(global_metrics)), "gauge")
        
        total_cache_hit_rate = sum(
            metrics['edge_cache']['hit_rate'] for metrics in global_metrics.values()
        ) / len(global_metrics)
        
        self.monitor.record_metric("global_cache_hit_rate", total_cache_hit_rate, "gauge")
    
    async def _optimize_cache_distribution(self, global_metrics: Dict[str, Any]):
        """Optimize cache content distribution across regions."""
        # Analyze access patterns across regions
        access_patterns = {}
        
        for region, cache in self.edge_caches.items():
            cache_stats = cache.get_cache_stats()
            access_patterns[region] = {
                'hit_rate': cache_stats['hit_rate'],
                'utilization': cache_stats['utilization'],
                'item_count': cache_stats['item_count']
            }
        
        # Predict future access patterns
        for region, cache in self.edge_caches.items():
            await cache.predict_access_patterns()
        
        # Consider content prefetching between regions
        # This is a simplified implementation
        for region, cache in self.edge_caches.items():
            if access_patterns[region]['hit_rate'] < 70:  # Low hit rate
                # Consider prefetching from high-performing regions
                for other_region, other_cache in self.edge_caches.items():
                    if (other_region != region and 
                        access_patterns[other_region]['hit_rate'] > 85):
                        # Implement inter-region prefetching logic
                        pass
    
    async def _optimize_edge_nodes(self, global_metrics: Dict[str, Any]):
        """Optimize edge node placement and load distribution."""
        # Analyze edge node performance
        for node_id, edge_node in self.edge_nodes.items():
            # Update node metrics (simplified)
            edge_node.current_load = np.random.uniform(0.3, 0.8)  # Simulate load
            edge_node.cache_hit_ratio = np.random.uniform(0.6, 0.95)  # Simulate cache performance
            edge_node.average_response_time_ms = np.random.uniform(20, 100)  # Simulate response time
        
        # Identify underperforming nodes
        underperforming_nodes = [
            node for node in self.edge_nodes.values()
            if node.current_load > 0.9 or node.average_response_time_ms > 150
        ]
        
        if underperforming_nodes:
            logger.warning(f"Found {len(underperforming_nodes)} underperforming edge nodes")
    
    async def _synchronization_loop(self):
        """Cross-region data synchronization loop."""
        while self.running:
            try:
                await self._perform_cross_region_sync()
                await asyncio.sleep(30)  # Sync every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Synchronization error: {e}")
                await asyncio.sleep(60)
    
    async def _perform_cross_region_sync(self):
        """Perform cross-region data synchronization."""
        # This is a simplified synchronization implementation
        # In production, would implement proper consistency protocols
        
        # Collect cache invalidations that need to be propagated
        invalidations = []
        
        # Propagate invalidations across regions
        for invalidation in invalidations:
            for region, cache in self.edge_caches.items():
                await cache.invalidate(invalidation['key'])
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get comprehensive global distribution statistics."""
        stats = {
            'regions': {},
            'edge_nodes': {},
            'global_metrics': {},
            'routing_performance': {}
        }
        
        # Regional statistics
        for region, cache in self.edge_caches.items():
            cache_stats = cache.get_cache_stats()
            
            lb_stats = {}
            if region in self.load_balancers:
                lb_stats = self.load_balancers[region].get_load_balancing_stats()
            
            scaler_stats = {}
            if region in self.auto_scalers:
                scaler_stats = self.auto_scalers[region].get_predictive_scaling_stats()
            
            stats['regions'][region.value] = {
                'cache': cache_stats,
                'load_balancer': lb_stats,
                'auto_scaler': scaler_stats
            }
        
        # Edge node statistics
        for node_id, edge_node in self.edge_nodes.items():
            stats['edge_nodes'][node_id] = {
                'region': edge_node.region.value,
                'location': edge_node.location.value,
                'current_load': edge_node.current_load,
                'health_status': edge_node.health_status,
                'cache_hit_ratio': edge_node.cache_hit_ratio,
                'average_response_time_ms': edge_node.average_response_time_ms
            }
        
        # Global metrics
        total_regions = len(self.edge_caches)
        total_cache_size_gb = sum(cache.current_size_bytes for cache in self.edge_caches.values()) / (1024**3)
        avg_hit_rate = sum(cache.get_cache_stats()['hit_rate'] for cache in self.edge_caches.values()) / max(total_regions, 1)
        
        stats['global_metrics'] = {
            'total_regions': total_regions,
            'total_edge_nodes': len(self.edge_nodes),
            'total_cache_size_gb': total_cache_size_gb,
            'average_cache_hit_rate': avg_hit_rate,
            'total_routing_decisions': len(self.routing_history)
        }
        
        # Routing performance
        if self.routing_history:
            recent_decisions = list(self.routing_history)[-100:]  # Last 100 decisions
            avg_expected_latency = sum(d.expected_latency_ms for d in recent_decisions) / len(recent_decisions)
            avg_cost = sum(d.cost_estimate for d in recent_decisions) / len(recent_decisions)
            
            stats['routing_performance'] = {
                'average_expected_latency_ms': avg_expected_latency,
                'average_cost_estimate': avg_cost,
                'recent_decision_count': len(recent_decisions)
            }
        
        return stats


class GlobalRequestRouter:
    """Routes requests to optimal regions and edge nodes."""
    
    def __init__(self, region_manager: GlobalRegionManager):
        self.region_manager = region_manager
        
    async def route_request(self,
                          request: GlobalRequest,
                          edge_caches: Dict[Region, IntelligentEdgeCache],
                          edge_nodes: Dict[str, EdgeNode]) -> RoutingDecision:
        """Route a request to the optimal region and edge nodes."""
        
        routing_reasons = []
        
        # Find candidate regions
        candidate_regions = await self._find_candidate_regions(request, routing_reasons)
        
        # Score regions based on multiple factors
        region_scores = {}
        
        for region in candidate_regions:
            score = await self._score_region(
                region, request, edge_caches, edge_nodes, routing_reasons
            )
            region_scores[region] = score
        
        # Select best region
        if not region_scores:
            # Fallback to closest region
            if request.client_coordinates:
                closest_regions = self.region_manager.find_closest_regions(
                    request.client_coordinates, max_regions=1
                )
                selected_region = closest_regions[0][0] if closest_regions else Region.US_EAST_1
            else:
                selected_region = request.client_region
            
            routing_reasons.append("Fallback to default region")
        else:
            selected_region = max(region_scores.keys(), key=lambda r: region_scores[r])
        
        # Select edge nodes in the region
        selected_edge_nodes = await self._select_edge_nodes(
            selected_region, request, edge_nodes
        )
        
        # Calculate expected latency
        expected_latency = self.region_manager.get_region_latency(
            request.client_region, selected_region
        )
        
        # Determine cache strategy
        cache_strategy = self._determine_cache_strategy(request, selected_region)
        
        # Calculate cost estimate
        region_info = self.region_manager.regions[selected_region]
        cost_estimate = region_info.cost_factor * (request.data_size_bytes / 1024 / 1024) * 0.001  # $0.001 per MB
        
        # Create fallback regions
        fallback_regions = [
            region for region, score in sorted(region_scores.items(), key=lambda x: x[1], reverse=True)
            if region != selected_region
        ][:2]  # Top 2 alternatives
        
        decision = RoutingDecision(
            request_id=request.request_id,
            selected_region=selected_region,
            selected_edge_nodes=selected_edge_nodes,
            routing_reasons=routing_reasons,
            expected_latency_ms=expected_latency,
            cache_strategy=cache_strategy,
            fallback_regions=fallback_regions,
            cost_estimate=cost_estimate
        )
        
        return decision
    
    async def _find_candidate_regions(self, 
                                    request: GlobalRequest,
                                    routing_reasons: List[str]) -> List[Region]:
        """Find candidate regions for request routing."""
        candidate_regions = []
        
        # Start with all regions
        all_regions = list(self.region_manager.regions.keys())
        
        # Filter by latency requirements
        if request.latency_requirement_ms < float('inf'):
            for region in all_regions:
                expected_latency = self.region_manager.get_region_latency(
                    request.client_region, region
                )
                if expected_latency <= request.latency_requirement_ms:
                    candidate_regions.append(region)
            
            if candidate_regions:
                routing_reasons.append(f"Filtered by latency requirement: {request.latency_requirement_ms}ms")
            else:
                candidate_regions = all_regions
                routing_reasons.append("No regions meet latency requirement, using all")
        else:
            candidate_regions = all_regions
        
        # Filter by compliance requirements if specified
        compliance_requirements = request.metadata.get('compliance_requirements', [])
        if compliance_requirements:
            compliant_regions = self.region_manager.get_compliance_compatible_regions(
                compliance_requirements
            )
            candidate_regions = [r for r in candidate_regions if r in compliant_regions]
            
            if candidate_regions:
                routing_reasons.append(f"Filtered by compliance: {compliance_requirements}")
            else:
                # No compliant regions, this is an error condition
                routing_reasons.append("No regions meet compliance requirements")
                candidate_regions = []
        
        return candidate_regions
    
    async def _score_region(self,
                          region: Region,
                          request: GlobalRequest,
                          edge_caches: Dict[Region, IntelligentEdgeCache],
                          edge_nodes: Dict[str, EdgeNode],
                          routing_reasons: List[str]) -> float:
        """Score a region for request routing."""
        score = 0.0
        
        # Latency factor (lower latency = higher score)
        expected_latency = self.region_manager.get_region_latency(
            request.client_region, region
        )
        latency_score = max(0, 1.0 - (expected_latency / 500.0))  # Normalize to 500ms max
        score += latency_score * 0.3
        
        # Cache hit probability
        if region in edge_caches:
            cache = edge_caches[region]
            cache_stats = cache.get_cache_stats()
            cache_hit_score = cache_stats['hit_rate'] / 100.0
            score += cache_hit_score * 0.2
        
        # Edge node availability and performance
        region_edge_nodes = [
            node for node in edge_nodes.values()
            if node.region == region and node.health_status == "healthy"
        ]
        
        if region_edge_nodes:
            avg_load = sum(node.current_load for node in region_edge_nodes) / len(region_edge_nodes)
            load_score = max(0, 1.0 - avg_load)  # Lower load = higher score
            score += load_score * 0.2
            
            avg_response_time = sum(node.average_response_time_ms for node in region_edge_nodes) / len(region_edge_nodes)
            response_time_score = max(0, 1.0 - (avg_response_time / 200.0))  # Normalize to 200ms
            score += response_time_score * 0.15
        
        # Cost factor (lower cost = higher score)
        region_info = self.region_manager.regions[region]
        cost_score = max(0, 2.0 - region_info.cost_factor)  # Normalize around 1.0 cost factor
        score += cost_score * 0.15
        
        return score
    
    async def _select_edge_nodes(self,
                               region: Region,
                               request: GlobalRequest,
                               edge_nodes: Dict[str, EdgeNode]) -> List[str]:
        """Select optimal edge nodes in a region."""
        # Find healthy edge nodes in the region
        region_nodes = [
            node for node in edge_nodes.values()
            if node.region == region and node.health_status == "healthy"
        ]
        
        if not region_nodes:
            return []
        
        # Score nodes based on load and capabilities
        node_scores = {}
        
        for node in region_nodes:
            score = 0.0
            
            # Load factor (lower load = higher score)
            load_score = max(0, 1.0 - node.current_load)
            score += load_score * 0.4
            
            # Cache performance
            cache_score = node.cache_hit_ratio
            score += cache_score * 0.3
            
            # Response time (lower = better)
            response_score = max(0, 1.0 - (node.average_response_time_ms / 200.0))
            score += response_score * 0.3
            
            node_scores[node.node_id] = score
        
        # Select top nodes (limit to 3 for redundancy)
        selected_nodes = sorted(
            node_scores.keys(),
            key=lambda n: node_scores[n],
            reverse=True
        )[:3]
        
        return selected_nodes
    
    def _determine_cache_strategy(self, 
                                request: GlobalRequest, 
                                selected_region: Region) -> CacheStrategy:
        """Determine optimal cache strategy for the request."""
        # Simple strategy selection based on request characteristics
        
        if request.cache_preference:
            if request.latency_requirement_ms < 100:
                return CacheStrategy.PREDICTIVE_CACHING
            elif request.data_size_bytes > 1024 * 1024:  # > 1MB
                return CacheStrategy.GEOGRAPHIC_PROXIMITY
            else:
                return CacheStrategy.HYBRID_INTELLIGENT
        else:
            return CacheStrategy.LEAST_RECENTLY_USED


# Example usage and testing
async def example_global_distribution():
    """Example of using the global distribution optimizer."""
    
    # Create optimizer
    optimizer = GlobalDistributionOptimizer()
    await optimizer.start()
    
    try:
        # Add regions with instances
        regions_to_add = [
            (Region.US_EAST_1, [
                InstanceConfiguration(
                    instance_id="us-east-1a-1",
                    address="10.0.1.10",
                    port=8080,
                    region="us-east-1",
                    availability_zone="us-east-1a"
                ),
                InstanceConfiguration(
                    instance_id="us-east-1b-1",
                    address="10.0.1.20",
                    port=8080,
                    region="us-east-1",
                    availability_zone="us-east-1b"
                )
            ]),
            (Region.EU_CENTRAL_1, [
                InstanceConfiguration(
                    instance_id="eu-central-1a-1",
                    address="10.1.1.10",
                    port=8080,
                    region="eu-central-1",
                    availability_zone="eu-central-1a"
                )
            ])
        ]
        
        for region, instances in regions_to_add:
            await optimizer.add_region(region, instances)
        
        # Add edge nodes
        edge_nodes = [
            EdgeNode(
                node_id="edge-us-east-1",
                location=EdgeLocation.CDN_EDGE,
                region=Region.US_EAST_1,
                coordinates=GeographicCoordinates(38.13, -78.45),
                capacity_cpu=8.0,
                capacity_memory_gb=16.0,
                capacity_storage_gb=100.0,
                supported_operations=["cache", "compute"]
            ),
            EdgeNode(
                node_id="edge-eu-central-1",
                location=EdgeLocation.HYBRID_EDGE,
                region=Region.EU_CENTRAL_1,
                coordinates=GeographicCoordinates(50.11, 8.68),
                capacity_cpu=4.0,
                capacity_memory_gb=8.0,
                capacity_storage_gb=50.0,
                supported_operations=["cache"]
            )
        ]
        
        for edge_node in edge_nodes:
            await optimizer.add_edge_node(edge_node)
        
        # Cache some content
        await optimizer.cache_content(
            key="api_docs_v1",
            content="OpenAPI documentation content...",
            regions=[Region.US_EAST_1, Region.EU_CENTRAL_1],
            size_bytes=1024 * 50,  # 50KB
            ttl_seconds=3600
        )
        
        # Simulate global requests
        requests = [
            GlobalRequest(
                request_id=f"req_{i}",
                client_region=Region.US_EAST_1 if i % 2 == 0 else Region.EU_CENTRAL_1,
                client_coordinates=GeographicCoordinates(40.7, -74.0) if i % 2 == 0 else GeographicCoordinates(52.5, 13.4),
                operation="get_api_docs",
                data_size_bytes=1024,
                latency_requirement_ms=150.0,
                consistency_requirement=SyncStrategy.EVENTUAL_CONSISTENCY
            )
            for i in range(10)
        ]
        
        # Route requests
        for request in requests:
            decision = await optimizer.route_request(request)
            print(f"Request {request.request_id}: routed to {decision.selected_region.value}, "
                  f"expected latency: {decision.expected_latency_ms:.1f}ms, "
                  f"cost: ${decision.cost_estimate:.4f}")
        
        # Wait for optimization cycles
        await asyncio.sleep(5)
        
        # Get global statistics
        stats = optimizer.get_global_stats()
        print(f"\nGlobal Distribution Statistics:")
        print(f"Active regions: {stats['global_metrics']['total_regions']}")
        print(f"Edge nodes: {stats['global_metrics']['total_edge_nodes']}")
        print(f"Average cache hit rate: {stats['global_metrics']['average_cache_hit_rate']:.1f}%")
        
        # Test cached content retrieval
        cached_content = await optimizer.get_cached_content("api_docs_v1", Region.US_EAST_1)
        print(f"Cache retrieval successful: {cached_content is not None}")
        
    finally:
        await optimizer.stop()


if __name__ == "__main__":
    asyncio.run(example_global_distribution())
    print("Global distribution optimization example completed!")