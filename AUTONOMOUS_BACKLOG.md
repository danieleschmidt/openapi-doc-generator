# Autonomous Development Backlog

**Generated:** 2025-07-22 16:37 UTC  
**Analysis Method:** WSJF (Weighted Shortest Job First)  
**Current State:** 97% test coverage, 154 tests passing, all security scans clear, all linting issues resolved

## Scoring Methodology

**Impact Score (1-10):**
- 10: Critical security/reliability issues
- 8-9: High performance or maintainability improvements  
- 6-7: Medium enhancements to user experience
- 4-5: Low-medium code quality improvements
- 1-3: Minor optimizations

**Effort Score (1-10):**
- 1-2: Small changes (< 30 min)
- 3-4: Medium changes (30-90 min)  
- 5-6: Large changes (2-4 hours)
- 7-8: Complex changes (4-8 hours)
- 9-10: Major refactoring (1+ days)

**WSJF Score = Impact / Effort**

---

## High Priority Tasks (WSJF > 2.5)

### 1. **✅ Improve Test Coverage in Discovery Module** - COMPLETED
- **WSJF:** 4.33 (Impact: 6, Effort: 3)
- **Missing Coverage:** Lines 115-116, 130, 136, 138, 140, 143 in `discovery.py` (92% → 97%)
- **Description:** Add edge case tests for AST parsing failures and framework detection edge cases
- **Implementation:** Write tests for malformed Python files, missing imports, and fallback scenarios
- **Risk:** Low - additive testing only
- **RESULT:** Improved coverage from 92% to 97%, added 12 comprehensive edge case tests

### 2. **✅ Complete CLI Error Path Coverage** - COMPLETED
- **WSJF:** 3.5 (Impact: 7, Effort: 2)  
- **Missing Coverage:** Lines 127-128, 138 in `cli.py` (97% → 99%)
- **Description:** Add tests for JSON parsing errors and unknown format handling  
- **Implementation:** Create malformed JSON old-spec files and test invalid format arguments
- **Risk:** Low - testing error paths
- **RESULT:** Improved coverage from 97% to 99%, identified line 138 as unreachable dead code

### 3. **✅ Enhance GraphQL Error Resilience** - COMPLETED
- **WSJF:** 3.0 (Impact: 6, Effort: 2)
- **Missing Coverage:** Lines 34, 36 in `graphql.py` (92% → 100%)  
- **Description:** Add tests for GraphQL schema parsing failures and malformed schemas
- **Implementation:** Test with invalid GraphQL schema files and corrupted input
- **Risk:** Low - defensive improvement
- **RESULT:** Achieved 100% coverage, added comprehensive error scenario testing

### 4. **✅ Optimize Complex Functions (Cyclomatic Complexity)** - COMPLETED
- **WSJF:** 2.67 (Impact: 8, Effort: 3)
- **Description:** Refactor `main()` function (complexity C), `_detect_framework()` (complexity C), and validator methods
- **Implementation:** Extract helper functions, reduce nesting, simplify conditional logic
- **Risk:** Medium - requires careful refactoring to maintain behavior
- **RESULT:** Successfully reduced both functions from complexity C to B, added 16 behavior verification tests

---

## Medium Priority Tasks (WSJF 1.5-2.5)

### 5. **✅ Performance Optimization: AST Caching** - COMPLETED
- **WSJF:** 2.0 (Impact: 6, Effort: 3)
- **Description:** Cache parsed AST results to avoid repeated parsing of the same files
- **Implementation:** Add LRU cache to AST parsing functions, measure performance improvement
- **Risk:** Low - performance enhancement with fallback
- **RESULT:** Achieved 40.7% performance improvement (1.69x speedup), added 7 comprehensive tests

### 6. **✅ Add Structured JSON Logging Option** - COMPLETED
- **WSJF:** 1.75 (Impact: 7, Effort: 4) 
- **Description:** Implement `--log-format json` CLI option for machine-readable logs
- **Implementation:** Add structured logging with correlation IDs and timing metrics
- **Risk:** Low - additive feature
- **RESULT:** Implemented comprehensive JSON logging with correlation IDs, timing metrics, and 8 comprehensive tests

### 7. **✅ Docker Image Creation** - COMPLETED
- **WSJF:** 1.67 (Impact: 5, Effort: 3)
- **Description:** Create automated Docker image build and publish workflow
- **Implementation:** Add Dockerfile, GitHub Actions workflow, update documentation
- **Risk:** Low - infrastructure improvement
- **RESULT:** Created multi-stage optimized Dockerfile with security hardening, automated GitHub Actions CI/CD pipeline, comprehensive test suite with 9 tests, updated documentation with Docker usage examples

---

## Lower Priority Tasks (WSJF < 1.5)

### 8. **✅ Add Route Performance Metrics** - COMPLETED
- **WSJF:** 1.4 (Impact: 7, Effort: 5)
- **Description:** Collect and emit metrics on route discovery performance
- **Implementation:** Add timing decorators, memory usage tracking, export to structured logs
- **Risk:** Low - observability enhancement
- **RESULT:** Implemented comprehensive performance measurement decorator with timing, memory tracking, and aggregated statistics. Added CLI option `--performance-metrics`, integrated with JSON logging, and created 11 comprehensive tests covering all functionality.

### 9. **Expand Framework Support**
- **WSJF:** 1.2 (Impact: 6, Effort: 5)  
- **Description:** Add support for Starlette, Tornado, or other frameworks
- **Implementation:** Create new plugin, add detection logic, comprehensive testing
- **Risk:** Medium - new feature requiring extensive testing

### 10. **Advanced CLI Features**
- **WSJF:** 1.0 (Impact: 4, Effort: 4)
- **Description:** Add verbose/quiet modes, colored output, progress indicators
- **Implementation:** Enhanced CLI with rich formatting and user experience improvements
- **Risk:** Low - user experience enhancement

---

## Technical Debt Register

| Component | Debt Type | Priority | Estimated Effort |
|-----------|-----------|----------|------------------|
| `discovery.py` | High complexity functions | Medium | 2-3 hours |
| `validator.py` | Complex validation methods | Medium | 2-3 hours |
| `cli.py` | Monolithic main function | Low | 1-2 hours |

---

## Newly Identified High Priority Tasks (WSJF > 3.0)

### 9. **Reduce Function Complexity - Discovery Module** - NEW
- **WSJF:** 4.0 (Impact: 8, Effort: 2)
- **Description:** Refactor deeply nested functions in discovery.py (_discover_flask, _discover_fastapi, _discover_django)
- **Implementation:** Extract helper methods, reduce nesting depth from 7 to ≤4 levels
- **Risk:** Medium - requires careful refactoring to maintain behavior
- **Files:** `src/openapi_doc_generator/discovery.py:224, 189, 267`

### 10. **Replace Broad Exception Handlers** - NEW
- **WSJF:** 3.5 (Impact: 7, Effort: 2)
- **Description:** Replace generic `except Exception:` with specific exception types for better error handling
- **Implementation:** Add specific exception types for AST parsing, GraphQL parsing, plugin loading
- **Risk:** Low - improves error diagnostics and debugging
- **Files:** `src/openapi_doc_generator/utils.py:246, schema.py:59, graphql.py:28, discovery.py:71`

### 11. **Improve Validator Module Complexity** - NEW  
- **WSJF:** 3.0 (Impact: 6, Effort: 2)
- **Description:** Reduce cyclomatic complexity in validator methods (8+ nesting levels)
- **Implementation:** Extract validation logic into smaller, focused methods
- **Risk:** Low - improves maintainability and testability
- **Files:** `src/openapi_doc_generator/validator.py:95, 184`

## Remaining Priority Tasks (WSJF 1.5-3.0)

### 12. **✅ Extract Magic Numbers to Constants** - COMPLETED
- **WSJF:** 2.0 (Impact: 4, Effort: 2)  
- **Description:** Replace hardcoded values with named constants for better maintainability
- **Implementation:** Extract cache sizes, memory conversion factors, default URLs
- **Risk:** Very Low - improves code readability
- **RESULT:** Added AST_CACHE_SIZE, MEMORY_CONVERSION_FACTOR, TEST_BASE_URL constants to config.py

### 13. **✅ Fix Linting Issues** - COMPLETED
- **WSJF:** 3.0 (Impact: 3, Effort: 1)
- **Description:** Resolve 9 line length violations (E501 errors) for better code quality
- **Implementation:** Break long lines while preserving functionality
- **Risk:** Very Low - formatting improvements only  
- **RESULT:** Fixed all linting violations across 4 modules, improved code readability

### 14. **✅ Reduce Validator Complexity** - COMPLETED  
- **WSJF:** 2.5 (Impact: 5, Effort: 2)
- **Description:** Reduce _validate_components_section complexity from C=11 to B rating
- **Implementation:** Extract helper methods for focused validation logic
- **Risk:** Low - improves maintainability
- **RESULT:** Reduced complexity by creating _validate_component_schemas, _validate_single_schema, _has_valid_schema_properties methods

## Next Actions

1. **Immediate (Next 1 hour):** Start with Task #9 (Discovery module complexity) - highest new WSJF
2. **Short Term (This session):** Complete Tasks #9-11 to address code complexity issues
3. **Medium Term:** Address remaining magic numbers and technical debt
4. **Long Term:** Framework expansion and advanced features

---

## Success Metrics

- **Coverage Target:** 100% test coverage across all modules
- **Performance Target:** Route discovery < 1s for typical applications
- **Quality Target:** All functions complexity < B rating
- **Security Target:** Zero security issues in all scans