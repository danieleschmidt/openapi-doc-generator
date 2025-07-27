# Architecture Decision Records (ADR)

This directory contains Architecture Decision Records (ADRs) for the OpenAPI Doc Generator project.

## What is an ADR?

An Architecture Decision Record (ADR) is a document that captures an important architectural decision made along with its context and consequences.

## ADR Template

For consistency, please use the following template when creating new ADRs:

```markdown
# ADR-XXXX: [Title]

## Status
[Proposed | Accepted | Deprecated | Superseded by ADR-YYYY]

## Context
[Describe the forces at play, including technological, political, social, and project local]

## Decision
[State the architecture decision and full justification]

## Consequences
[Describe the resulting context, after applying the decision. All consequences should be listed here, not just the "positive" ones]

## Alternatives Considered
[List the options that were evaluated and why they were not chosen]

## Date
[When the decision was made]
```

## Current ADRs

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [001](001-plugin-architecture.md) | Plugin Architecture for Framework Support | Accepted | 2024-07-15 |
| [002](002-ast-based-discovery.md) | AST-Based Route Discovery | Accepted | 2024-07-20 |
| [003](003-performance-monitoring.md) | Integrated Performance Monitoring | Accepted | 2024-12-01 |
| [004](004-container-security.md) | Container Security Strategy | Accepted | 2025-01-15 |

## Guidelines

1. **Numbering**: Use sequential numbering (001, 002, 003, etc.)
2. **Naming**: Use kebab-case for filenames
3. **Status**: Keep status updated as decisions evolve
4. **Context**: Provide sufficient context for future maintainers
5. **Review**: All ADRs should be reviewed by the core team before acceptance