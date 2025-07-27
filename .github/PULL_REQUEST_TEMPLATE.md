# Pull Request

## Description

Brief description of the changes in this PR.

## Type of Change

Please check the type of change your PR introduces:

- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] 🚀 New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🎨 Code style/formatting
- [ ] ♻️ Code refactoring (no functional changes)
- [ ] ⚡ Performance improvement
- [ ] ✅ Test addition or improvement
- [ ] 🔧 Build/CI changes
- [ ] 🔒 Security fix

## Related Issues

- Fixes #(issue number)
- Closes #(issue number)
- Related to #(issue number)

## Changes Made

### Summary
- Change 1: Brief description
- Change 2: Brief description
- Change 3: Brief description

### Technical Details
Provide more detailed technical information about the implementation:

- Architecture changes
- New dependencies added
- Database schema changes (if any)
- API changes (if any)

## Testing

### Test Coverage
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] End-to-end tests added/updated
- [ ] Performance tests added/updated
- [ ] Security tests added/updated

### Manual Testing
Describe the manual testing you performed:

```bash
# Commands used for testing
openapi-doc-generator --app examples/app.py --format openapi
```

**Test Results:**
- ✅ Feature works as expected
- ✅ No regressions detected
- ✅ Performance is acceptable

### Test Environment
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.9.0]
- Framework tested: [e.g., Flask 2.0.1]

## Documentation

- [ ] Code is self-documenting with clear variable names and functions
- [ ] Docstrings added/updated for new functions and classes
- [ ] README updated (if applicable)
- [ ] Documentation added to `docs/` (if applicable)
- [ ] API documentation updated (if applicable)
- [ ] Examples updated (if applicable)

## Security Considerations

- [ ] No sensitive information (credentials, tokens, etc.) is included
- [ ] Input validation is implemented where needed
- [ ] Security implications have been considered
- [ ] Dependencies are secure and up-to-date

## Performance Impact

- [ ] No significant performance regression
- [ ] Performance improvements documented
- [ ] Memory usage considered
- [ ] Benchmarks run (if applicable)

## Backward Compatibility

- [ ] Changes are backward compatible
- [ ] Breaking changes are documented and justified
- [ ] Migration guide provided (if needed)
- [ ] Deprecation warnings added (if applicable)

## Checklist

### Code Quality
- [ ] Code follows the project's style guidelines
- [ ] Self-review of code completed
- [ ] Code is DRY (Don't Repeat Yourself)
- [ ] Error handling is appropriate
- [ ] Logging is appropriate

### Testing & CI
- [ ] All tests pass locally
- [ ] Pre-commit hooks pass
- [ ] CI/CD pipeline passes
- [ ] Code coverage maintained or improved
- [ ] No new security vulnerabilities introduced

### Documentation & Communication
- [ ] Changes are documented
- [ ] Complex algorithms are explained
- [ ] Public API changes are documented
- [ ] Breaking changes are clearly marked

## Deployment Notes

Any special deployment considerations:

- [ ] Requires database migration
- [ ] Requires configuration changes
- [ ] Requires environment variable updates
- [ ] Requires dependency updates

## Screenshots (if applicable)

Include screenshots for UI changes or CLI output changes.

## Additional Notes

Any additional information that reviewers should know:

- Known limitations
- Future improvement suggestions
- Dependencies on other PRs
- Special review focus areas

---

## For Reviewers

### Review Focus Areas
Please pay special attention to:

- [ ] Logic correctness
- [ ] Error handling
- [ ] Performance implications
- [ ] Security considerations
- [ ] Documentation completeness

### Testing Checklist for Reviewers
- [ ] Checkout the branch and test locally
- [ ] Verify tests pass
- [ ] Check that examples work
- [ ] Validate documentation accuracy

---

**By submitting this PR, I confirm that:**

- [ ] I have read and followed the [contributing guidelines](CONTRIBUTING.md)
- [ ] I have tested my changes thoroughly
- [ ] I have considered the impact on existing users
- [ ] I am willing to address feedback and make necessary changes