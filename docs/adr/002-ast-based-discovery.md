# ADR-002: AST-Based Route Discovery

## Status
Accepted

## Context
The route discovery system needs to analyze Python and JavaScript source code to extract:
- Route definitions and HTTP methods
- Function signatures and parameter types
- Docstrings and inline documentation
- Type annotations and schema information

We evaluated several approaches for code analysis:
- Regular expressions
- String parsing
- AST (Abstract Syntax Tree) parsing
- Static analysis tools

## Decision
Use AST-based parsing for route discovery with the following implementation:

1. **Python AST**: Use Python's built-in `ast` module for Python frameworks
2. **JavaScript AST**: Use appropriate JavaScript AST parsers for Node.js frameworks
3. **Caching**: Cache parsed AST trees to improve performance
4. **Type Inference**: Extract type information from annotations and docstrings

## Consequences

### Positive
- **Accuracy**: More reliable than regex-based parsing
- **Completeness**: Can extract complex code patterns and metadata
- **Type Safety**: Better type inference from annotations
- **Maintainability**: Easier to extend for new language features
- **Performance**: Caching provides good performance characteristics

### Negative
- **Complexity**: AST parsing is more complex than string-based approaches
- **Dependencies**: Requires language-specific AST parsers
- **Memory Usage**: AST trees consume more memory than simple parsing
- **Edge Cases**: Complex code patterns may still be challenging to parse

## Alternatives Considered

### Regular Expression Parsing
Simple regex patterns to match route definitions.
- **Rejected**: Too fragile and prone to false positives/negatives

### String-Based Parsing
Line-by-line string analysis with pattern matching.
- **Rejected**: Cannot handle complex nested structures reliably

### External Static Analysis Tools
Use tools like `jedi` for Python or `typescript-eslint` for JavaScript.
- **Rejected**: Additional dependencies and complexity without significant benefits

## Implementation Details

```python
import ast
from typing import List, Dict, Any

class ASTAnalyzer:
    def __init__(self):
        self._cache: Dict[str, ast.AST] = {}
    
    def parse_file(self, file_path: str) -> ast.AST:
        if file_path in self._cache:
            return self._cache[file_path]
        
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())
            self._cache[file_path] = tree
            return tree
    
    def extract_routes(self, tree: ast.AST) -> List[Dict[str, Any]]:
        # Implementation for route extraction
        pass
```

## Performance Considerations
- AST parsing cached per file to avoid re-parsing
- Memory usage monitored for large codebases
- Parallel processing for multiple files when beneficial

## Date
2024-07-20