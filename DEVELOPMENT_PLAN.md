# 🧭 Project Vision

> A robust tool that automatically generates OpenAPI specs, markdown docs, and testing artifacts for multiple web frameworks. Developers can quickly document and validate APIs without manual effort.

---

# 📅 12-Week Roadmap

## I1 – Security & Refactoring
- **Themes**: Security, Code Health
- **Goals / Epics**
  - Harden CLI input validation and error handling
  - Introduce static analysis and secret scanning in CI
  - Refactor discovery modules for extensibility
- **Definition of Done**
  - All inputs validated with unit tests
  - `bandit` and secret-scan hooks run in CI
  - Plugins load via entry points with documented interface

## I2 – Performance & Developer UX
- **Themes**: Performance, Developer Experience
- **Goals / Epics**
  - Optimize route and schema discovery to avoid repeated AST parsing
  - Improve CLI feedback and logging
  - Ship pre-built Docker image for easy adoption
- **Definition of Done**
  - Discovery completes in <2s for example app
  - CLI offers verbose and quiet modes
  - Docker image published with automated build

## I3 – Observability & Community
- **Themes**: Observability, Documentation, Community
- **Goals / Epics**
  - Emit structured logs and usage metrics
  - Expand README with advanced examples and FAQs
  - Establish contribution guidelines and triage process
- **Definition of Done**
  - Logs consumable by standard aggregators
  - Docs cover all output formats with screenshots
  - CONTRIBUTING updated with issue/PR workflow

---

# ✅ Epic & Task Checklist

### 🔐 Increment 1: Security & Refactoring
- [ ] **EPIC** Harden CLI input handling
  - [ ] Validate file paths and formats
  - [ ] Add error codes and docs
- [ ] **EPIC** Add security scanners to CI
  - [ ] Integrate `bandit` rules
  - [ ] Add secret scanning pre-commit
- [ ] **EPIC** Modularize discovery plugins
  - [ ] Load plugins via entry points
  - [ ] Document plugin API

### 🚀 Increment 2: Performance & Developer UX
- [ ] **EPIC** Speed up discovery
  - [ ] Cache parsed ASTs
  - [ ] Profile large apps
- [ ] **EPIC** Improve CLI usability
  - [ ] Verbose/quiet flags
  - [ ] Clearer error messages
- [ ] **EPIC** Publish Docker image
  - [ ] GitHub Actions build & push
  - [ ] Provide usage docs

### 📈 Increment 3: Observability & Community
- [ ] **EPIC** Structured logging
  - [ ] JSON log option
  - [ ] Add request/route metrics
- [ ] **EPIC** Expand documentation
  - [ ] Advanced guides
  - [ ] FAQ section
- [ ] **EPIC** Community processes
  - [ ] Issue templates
  - [ ] Release checklist

---

# ⚠️ Risks & Mitigation
- **Framework drift** – track upstream releases and update plugins regularly.
- **Large spec complexity** – add profiling and optimize data structures.
- **CI instability** – run workflows on matrix of Python versions and keep caches minimal.
- **Security false positives** – tune scanners and allow suppressions via config.

---

# 📊 KPIs & Metrics
- [ ] >85% test coverage
- [ ] <15 min CI pipeline time
- [ ] <5% error rate on core CLI commands
- [ ] 100% secrets loaded from env or vault

---

# 👥 Ownership & Roles
- **DevOps** – CI/CD, Docker image, observability
- **Backend** – Core discovery and generation logic
- **Documentation** – Examples, guides, community support
