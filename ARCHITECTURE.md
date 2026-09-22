# RateMyCollezzz — Locked Architecture

## 1. Purpose

RateMyCollezzz is a student-first college discovery and review platform.

The project is intentionally divided into five architectural layers.

The purpose of this structure is to keep responsibilities clear, keep the
repository understandable, prevent duplicate logic, and stop implementation
tools from expanding the architecture without approval.

This document is the architectural source of truth.

Normal implementation work must NOT modify this file.

---

# 2. Five-Layer Architecture

## Layer 1 — Experience

Folder:

experience/

Responsibility:

Everything directly related to the user-facing experience.

Examples:

- College discovery
- Search
- Filters
- 2D map
- College details
- Review presentation
- Community presentation
- Profiles
- Compare
- Notifications UI
- Responsive interface

Layer 1 does NOT own:

- Database business logic
- Review intelligence
- Trust decisions
- Authentication internals
- Persistent data storage

Layer 1 communicates with Layer 2.

---

## Layer 2 — Platform Core

Folder:

platform/

Primary technology:

Django + Python

Responsibility:

Application and business logic.

Examples:

- Accounts
- Profiles
- Colleges
- Locations
- Reviews
- Ratings
- Communities
- Confessions
- Discussions
- Chat
- Notifications
- Reports
- Moderation records
- Platform APIs
- Permissions
- Business rules

Layer 2 owns the platform domain.

Layer 2 does NOT perform ML review classification.

Layer 2 does NOT independently perform final trust decisions.

---

## Layer 3 — Review Intelligence / RRM

Folder:

rrm/

Responsibility:

Understand and analyze review content.

Core pipeline:

Review
  ->
Validation
  ->
Duplicate / similarity checks
  ->
RMC tokenizer
  ->
Transformer Encoder
      +
Character CNN
  ->
Feature Fusion
  ->
Multi-task Prediction
  ->
Risk Signals

Intended outputs include:

- spam risk
- deception risk
- toxicity risk
- advertising risk
- off-topic risk
- PII risk
- similarity indicators
- review-quality signals
- optional experimental AI-like signal

Core principle:

RRM UNDERSTANDS.

RRM does NOT make the final moderation decision.

RRM must never directly delete a review.

---

## Layer 4 — Trust & Safety

Folder:

trust/

Responsibility:

Combine evidence and determine platform-level trust actions.

Possible inputs:

- RRM outputs
- duplicate / similarity evidence
- account behaviour
- timing behaviour
- rating patterns
- coordination evidence
- platform rules
- reports

Possible decisions:

- Publish
- Moderate
- Hold
- Remove from public visibility

Removal from public visibility does not imply permanent destruction of
internal evidence.

Core principle:

TRUST DECIDES.

Layer 4 must remain separate from Layer 3.

---

## Layer 5 — Data / Security / Operations

Folder:

ops/

Responsibility:

Cross-cutting infrastructure required by the platform.

Examples:

- PostgreSQL
- Redis
- WebSockets infrastructure
- Authentication infrastructure
- Privacy controls
- Audit
- Rate limiting
- Backups
- Monitoring
- Model storage
- Deployment
- Production configuration
- Performance operations

Layer 5 supports the other layers.

It must not absorb their business logic.

---

# 3. Critical Data Separation

Reviews may affect college ratings.

Community content does not affect college ratings.

Therefore:

Review
  ->
may contribute to college rating

Confession
  ->
community content only
  ->
must not contribute to college rating

Discussion
  ->
community content only
  ->
must not contribute to college rating

Chat
  ->
conversation only
  ->
must not contribute to college rating

---

# 4. Review Intelligence Boundary

The critical system boundary is:

Layer 3 = intelligence

Layer 4 = trust decision

Expected flow:

Review
  ->
RRM
  ->
Risk Signals
  ->
Trust Engine
  ->
Risk Resolver
  ->
Platform Decision

The RRM must not become an autonomous moderation engine.

---

# 5. Repository Architecture Rule

Only these five architectural root folders are permitted:

experience/
platform/
rrm/
trust/
ops/

New top-level architectural folders require explicit architectural approval.

Implementation agents must not create alternative structures such as:

backend/
frontend/
core/
shared/
common/
services/
helpers/
engines/

at repository root unless this architecture is deliberately revised first.

---

# 6. Development Rule

Only one architectural layer is considered ACTIVE at a time.

Within an active layer, work is divided into small controlled components.

A future layer may be studied for compatibility, but must not be implemented
early.

---

# 7. Legacy Code

The previous implementation is preserved separately outside this active
repository.

Legacy code is reference material.

It must never be copied blindly into the new architecture.

Migration process:

Legacy component
  ->
Inspect
  ->
Understand
  ->
Validate usefulness
  ->
Adapt or rebuild
  ->
Place inside correct locked layer

---

# 8. Architecture Change Policy

Architecture changes require deliberate approval.

An implementation tool must never make architecture changes autonomously.

If a requested implementation appears to require an architectural change,
the implementation must stop and report the requirement instead.
