"""High level API documentation orchestrator."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List

from .config import config
from .discovery import RouteDiscoverer, RouteInfo
from .markdown import MarkdownGenerator
from .schema import SchemaInferer, SchemaInfo
from .spec import OpenAPISpecGenerator
from .utils import PerformanceMetrics, get_performance_tracker, get_processing_pool
from .i18n import get_i18n_manager


@dataclass
class DocumentationResult:
    routes: List[RouteInfo]
    schemas: List[SchemaInfo]

    def generate_openapi_spec(
        self,
        title: str = config.DEFAULT_API_TITLE,
        version: str = config.DEFAULT_API_VERSION,
        localize: bool = True,
    ) -> dict:
        """Return OpenAPI specification for the analyzed app."""
        generator = OpenAPISpecGenerator(
            self.routes, self.schemas, title=title, version=version
        )
        spec = generator.generate()
        
        # Apply localization if enabled
        if localize:
            i18n_manager = get_i18n_manager()
            spec = i18n_manager.localize_documentation(spec)
        
        return spec

    def generate_markdown(
        self,
        title: str = config.DEFAULT_API_TITLE,
        version: str = config.DEFAULT_API_VERSION,
    ) -> str:
        """Return markdown documentation for the analyzed app."""
        spec = self.generate_openapi_spec(title=title, version=version)
        return MarkdownGenerator().generate(spec)


class APIDocumentator:
    """Analyze an application to generate documentation artifacts with scaling optimization."""

    def __init__(self, enable_concurrent_processing: bool = True) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self._enable_concurrent = enable_concurrent_processing
        self._performance_tracker = get_performance_tracker()

    def analyze_app(self, app_path: str) -> DocumentationResult:
        """Analyze application with performance optimizations and concurrent processing."""
        start_time = time.time()
        self._logger.info("Discovering routes from %s", app_path)

        if self._enable_concurrent:
            return self._analyze_app_concurrent(app_path, start_time)
        else:
            return self._analyze_app_sequential(app_path, start_time)

    def _analyze_app_sequential(self, app_path: str, start_time: float) -> DocumentationResult:
        """Sequential analysis (original method)."""
        routes = RouteDiscoverer(app_path).discover()

        self._logger.info("Inferring schemas from %s", app_path)
        schemas = SchemaInferer(app_path).infer()

        # Record performance metrics
        duration_ms = (time.time() - start_time) * 1000
        metric = PerformanceMetrics(
            operation_name="analyze_app_sequential",
            duration_ms=duration_ms,
            memory_usage_mb=0.0,  # Would need memory tracking
            cpu_percent=0.0,      # Would need CPU tracking
            timestamp=time.time(),
            processing_rate=len(routes) / (duration_ms / 1000) if duration_ms > 0 else 0
        )
        self._performance_tracker.record_metric(metric)

        return DocumentationResult(routes, schemas)

    def _analyze_app_concurrent(self, app_path: str, start_time: float) -> DocumentationResult:
        """Concurrent analysis for improved performance."""
        processing_pool = get_processing_pool()

        # Submit concurrent tasks
        route_future = processing_pool.submit_task(self._discover_routes, app_path)
        schema_future = processing_pool.submit_task(self._infer_schemas, app_path)

        # Wait for completion and collect results
        routes = route_future.result()
        schemas = schema_future.result()

        # Record performance metrics
        duration_ms = (time.time() - start_time) * 1000
        metric = PerformanceMetrics(
            operation_name="analyze_app_concurrent",
            duration_ms=duration_ms,
            memory_usage_mb=0.0,  # Would need memory tracking
            cpu_percent=0.0,      # Would need CPU tracking
            timestamp=time.time(),
            thread_count=2,  # Two concurrent tasks
            processing_rate=len(routes) / (duration_ms / 1000) if duration_ms > 0 else 0
        )
        self._performance_tracker.record_metric(metric)

        return DocumentationResult(routes, schemas)

    def _discover_routes(self, app_path: str) -> List[RouteInfo]:
        """Thread-safe route discovery."""
        return RouteDiscoverer(app_path).discover()

    def _infer_schemas(self, app_path: str) -> List[SchemaInfo]:
        """Thread-safe schema inference."""
        self._logger.info("Inferring schemas from %s", app_path)
        try:
            return SchemaInferer(app_path).infer()
        except FileNotFoundError:
            self._logger.info("No models found in %s", app_path)
            return []
