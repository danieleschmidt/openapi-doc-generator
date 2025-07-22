## v0.1.5
 - **REFACTORING**: Reduce function complexity and improve code maintainability
   - Refactored discovery module: extracted helper methods, reduced nesting depth from 7 to 4 levels
   - Enhanced exception handling: replaced generic exceptions with specific types
   - Simplified validator complexity: broke down complex methods into focused helpers
 - **QUALITY**: Extract magic numbers to named constants for better configuration management
   - Added AST_CACHE_SIZE, MEMORY_CONVERSION_FACTOR, and TEST_BASE_URL constants
   - Centralized configuration for improved maintainability and testability
 - **LINTING**: Fixed all code style violations for improved readability
   - Resolved 9 line length violations (E501 errors)
   - Improved code formatting across 4 modules
 - **TESTING**: Maintained 97% test coverage with 154 passing tests
 - **COMPLEXITY**: Reduced cyclomatic complexity from C-rating to B-rating functions

## v0.1.4
 - **QUALITY**: Achieve 100% test coverage in schema module (up from 82%)
 - **ENHANCEMENT**: Add comprehensive schema file I/O error handling tests (9 new tests)
 - **ROBUSTNESS**: Test handling of unreadable files, encoding errors, and syntax errors
 - **RELIABILITY**: Ensure graceful degradation when schema processing fails
 - **QUALITY**: Improve overall test coverage to 96% with 86 total tests

## v0.1.3
 - **SECURITY**: Add comprehensive path traversal attack test coverage (16 new tests)
 - **ENHANCEMENT**: Improve GraphQL error handling with robust exception management  
 - **ENHANCEMENT**: Add empty path validation to prevent directory resolution exploits
 - **QUALITY**: Achieve 95% test coverage with 77 tests (up from 52)
 - **QUALITY**: Improve CLI and GraphQL module coverage by 7-8% each

## v0.1.2
 - **CRITICAL FIX**: Resolve XSS vulnerability in JavaScript JSON serialization for playground generation  
 - **ENHANCEMENT**: Improve CLI path validation to balance security with test compatibility
 - **QUALITY**: Achieve 93% test coverage with all 52 tests passing
 - **QUALITY**: Pass all linting checks and security scans (ruff, bandit)

## v0.1.1
 - **SECURITY**: Fix XSS vulnerability in playground HTML generation by properly escaping user input
 - **SECURITY**: Add path validation to prevent directory traversal attacks in CLI
 - **ENHANCEMENT**: Improve type safety with comprehensive type annotations across modules
 - **ENHANCEMENT**: Add robust error handling for file I/O and AST parsing operations
 - **ENHANCEMENT**: Add security tests for XSS prevention in playground generation
 - **QUALITY**: Add specific exception handling in schema inference and discovery modules

## v0.1.0
 - Introduce plugin interface for custom route discovery
 - Provide built-in aiohttp plugin
 - Document extension guide and update README
 - Bump package version to 0.1.0

## v0.0.18
 - Add CONTRIBUTING guide and CODEOWNERS
 - Validate CLI `--old-spec` input and log errors
 - Introduce basic logging across modules
 - Bump package version to 0.0.18

## v0.0.17
 - Document completed roadmap and sort changelog entries
 - Bump package version to 0.0.17

## v0.0.16
- Allow customizing API title and version via CLI options
- Bump package version to 0.0.16

## v0.0.15
- Generate API deprecation and migration guides
- CLI supports `--old-spec` and `--format guide`
- Bump package version to 0.0.15

## v0.0.14
- Automatically publish documentation via GitHub Pages
- Bump package version to 0.0.14

## v0.0.13
- Generate pytest suites from discovered routes
- CLI can write generated tests via --tests option
- Bump package version to 0.0.13

## v0.0.12
- Add GraphQL schema introspection support
- CLI can output GraphQL introspection JSON via --format graphql
- Bump package version to 0.0.12

## v0.0.11
- Add --version flag to CLI
- Bump package version to 0.0.11

## v0.0.10
- Add CLI option to choose output format (markdown, openapi, html)
- Bump package version to 0.0.10

## v0.0.9
- Add OpenAPI spec validation with improvement suggestions
- Bump package version to 0.0.9

## v0.0.8
- Implement interactive API playground generation
- Bump package version to 0.0.8

## v0.0.7
- Add schema inference for dataclasses and Pydantic models
- Generate basic OpenAPI specification
- Bump package version to 0.0.7

## v0.0.6
- Implement Django and Express route discovery
- Bump package version to 0.0.6

## v0.0.5
- Add Flask route discovery
- Document CLI usage in README
- Bump package version to 0.0.5

## v0.0.4
- Mark foundational testing and packaging tasks complete
- Add packaging metadata test
- Bump package version to 0.0.4

## v0.0.3
- Add GitHub Actions workflow for linting, security checks, and tests
- Mark CI setup task completed in docs
- Bump package version to 0.0.3

## v0.0.2
- Applying previous commit.
- Merge pull request #1 from danieleschmidt/codex/generate/update-strategic-development-plan
- docs(review): add code and product review
- Update README.md
- Initial commit
