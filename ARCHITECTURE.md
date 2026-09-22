# RateMyCollezzz — Locked Architecture

## 1. Purpose

RateMyCollezzz is a student-first college discovery, review, community,
and trust platform.

The project uses five architectural layers.

The architecture is intentionally compact so that:

- responsibilities remain obvious
- files remain easy to locate
- model research remains isolated from platform logic
- implementation agents cannot expand the project arbitrarily
- working functionality is preserved during refactoring
- future development remains understandable

This document is the architectural source of truth.

Normal implementation work must NOT modify this file unless an architecture
change is explicitly approved.

---

# 2. Layer 1 — Experience

Folder:

`experience/`

Responsibility:

Everything directly related to the user-facing interface.

Includes:

- Discover interface
- college search
- filters
- college list
- college details
- reviews UI
- profiles
- compare
- saved colleges
- notifications UI
- gallery UI
- community UI
- map interaction
- responsive design

Layer 1 owns presentation and interaction.

Layer 1 does NOT own:

- database business logic
- RRM model logic
- trust decisions
- persistence infrastructure
- security infrastructure

---

## 2.1 Map Requirement

The map system supports two presentation modes.

### 2D Mode

Supports:

- standard top-down map
- zoom
- pan
- college selection
- college pins
- search/navigation
- location exploration

### 3D Perspective Mode

Supports:

- pitched camera
- bearing/orientation changes
- improved spatial navigation
- better map perspective while exploring colleges and surrounding areas

Both modes must use the same real geographic data.

3D perspective MUST NOT fabricate:

- campus buildings
- college infrastructure
- fictional terrain
- synthetic campuses
- fictional geographic information

The 3D mode exists for map navigation and perspective.

It is NOT synthetic campus reconstruction.

---

## 2.2 Map Visual Direction

Future Layer-1 development should preserve the established UI direction:

- map-centric Discover experience
- high map clarity
- readable roads and area labels
- dark map visual hierarchy where appropriate
- strong college-location pins
- college logos inside pins
- clear hover state
- clearly stronger selected-pin state
- smooth zoom
- smooth pitch
- smooth bearing transitions
- 2D / 3D mode toggle

Map visual improvements must not change verified geographic information.

---

## 2.3 UI Migration Policy

Existing working Layer-1 functionality is preserved by default.

Refactoring means:

- reorganizing files
- clarifying responsibilities
- reducing unnecessary duplication
- improving maintainability
- improving visual design
- improving code readability

Refactoring does NOT automatically authorize feature removal.

A working UI feature may only be removed when:

1. it is demonstrated to be obsolete or truly duplicate, and
2. removal is explicitly approved.

The legacy frontend is therefore migration material and reference
implementation.

It is not disposable code.

---

# 3. Layer 2 — Platform Core

Folder:

`platform/`

Primary technology:

Django + Python

Responsibility:

Application and business logic.

Includes:

- accounts
- profiles
- colleges
- college locations
- reviews
- ratings
- helpful votes
- saved colleges
- compare support
- communities
- confessions
- discussions
- chat
- notifications
- reports
- moderation records
- platform APIs
- permissions
- authorization rules
- business rules

Layer 2 owns the platform domain.

Layer 2 must NOT implement the RRM neural architecture.

Layer 2 must NOT replace Layer-4 trust decisions.

---

# 4. Layer 3 — Review Intelligence / RRM

Folder:

`rrm/`

Responsibility:

Understand and analyze review content.

The RRM must remain independently testable and must expose clean contracts
for later integration with Layer 2 and Layer 4.

---

## 4.1 RRM Pipeline

Conceptual pipeline:

Review
↓
Input Validation
↓
Exact Duplicate / Similarity Signals
↓
RMC Tokenizer
↓
Semantic Branch + Character Branch
↓
Feature Fusion
↓
Multi-task Prediction
↓
Risk Signals

Semantic branch:

SentencePiece tokens
↓
Transformer Encoder
↓
Contextual review representation

Character branch:

Raw characters
↓
Character CNN
↓
Character-level representation

Then:

Semantic representation
+
Character representation
↓
Feature Fusion
↓
Multi-task Heads

---

## 4.2 Intended RRM Signals

Primary intended signals:

- spam
- deception
- toxicity
- advertising
- off-topic
- PII
- similarity indicators
- review-quality signals

Optional experimental signal:

- AI-like writing indicator

The AI-like signal must never be treated as strong proof of authorship.

---

## 4.3 Model Direction

Current research direction includes:

- custom SentencePiece tokenizer
- English reviews
- Hinglish reviews
- Roman Hindi
- Indian English
- college terminology
- compact Transformer encoder
- Character CNN
- feature fusion
- multi-task prediction heads

Current model-size design target:

approximately 22–30M trainable parameters

This is a DESIGN TARGET.

It is NOT:

- a measured result
- a proven optimum
- a mandatory final parameter count

Final values for:

- model dimensions
- vocabulary size
- sequence length
- dropout
- pooling
- loss weighting
- learning rate
- thresholds
- calibration
- parameter count

must be justified through actual experiments.

---

## 4.4 Research Principle

Do not jump directly to the final custom model.

Research progression must include:

Simple Baselines
↓
Strong Pretrained Baselines
↓
Custom RRM
↓
Ablations
↓
Robustness Evaluation
↓
Efficiency Evaluation
↓
Error Analysis

The custom architecture must earn its complexity through evidence.

---

## 4.5 RRM / Trust Boundary

Core rule:

**RRM UNDERSTANDS.**

**TRUST DECIDES.**

The RRM must NOT directly:

- delete reviews
- ban users
- remove users
- approve moderation cases
- permanently reject reviews
- make final platform moderation decisions

The RRM produces evidence and risk signals.

Layer 4 consumes those signals.

---

# 5. Layer 4 — Trust & Safety

Folder:

`trust/`

Responsibility:

Combine evidence and determine platform-level trust actions.

Possible inputs include:

- RRM outputs
- similarity evidence
- account behaviour
- timing behaviour
- rating patterns
- coordination evidence
- user reports
- platform rules

Possible platform decisions include:

- Publish
- Moderate
- Hold
- Remove from public visibility

Expected flow:

Review
↓
RRM
↓
Risk Signals
↓
Trust Engine
↓
Risk Resolver
↓
Platform Decision

Layer 4 is architecturally separate from Layer 3.

Layer-4 algorithms must not be implemented inside the RRM subsystem.

Implementation timing is controlled by `IMPLEMENTATION_PLAN.md`.

Future Layer-4 research may include:

- behavioural risk
- temporal bursts
- account patterns
- coordinated manipulation
- campaign detection
- reputation signals
- collective opinion spam

Future requirements may influence interface design.

Future algorithms must not be implemented early without approval.

---

# 6. Layer 5 — Data / Security / Operations

Folder:

`ops/`

Responsibility:

Cross-cutting infrastructure required by the platform.

Includes:

- PostgreSQL
- Redis
- WebSocket infrastructure
- authentication infrastructure
- privacy controls
- security configuration
- audit
- backups
- monitoring
- model storage
- deployment
- performance operations
- rate limiting

Layer 5 supports the other layers.

It must not absorb their application logic or ML responsibilities.

---

## 6.1 Authentication Boundary

Layer 2 owns:

- user identity behaviour
- roles
- permissions
- authorization rules
- account-related business logic

Layer 5 owns:

- secure authentication infrastructure
- credential protection
- session/security configuration
- secret handling
- security middleware
- rate limiting
- production security controls

This separation prevents duplicated authentication logic.

---

# 7. Critical Data Separation

Reviews may contribute to college ratings.

Community content does not.

Therefore:

Review
↓
may affect college rating

Confession
↓
community content only

Discussion
↓
community content only

Chat
↓
conversation only

Community popularity must never directly alter college ratings.

---

# 8. Development Order vs Architecture Order

Architectural layer numbers describe RESPONSIBILITY.

They do NOT require implementation to occur strictly in numerical order.

The currently active development layer and component are defined in:

`IMPLEMENTATION_PLAN.md`

A layer may be developed independently when:

- its boundaries remain intact
- its contracts are documented
- fake implementations of other layers are not created
- its interfaces remain testable
- future integration requirements are considered without implementing
  future algorithms prematurely

---

# 9. Repository Structure

Only these five architectural root folders are permitted:

`experience/`

`platform/`

`rrm/`

`trust/`

`ops/`

Do not create alternate architectural roots such as:

`backend/`

`frontend/`

`core/`

`shared/`

`common/`

`services/`

`engines/`

unless an architecture revision is explicitly approved.

Nested folders may be created only when they solve a demonstrated
organizational problem.

Folder creation is not a substitute for clear design.

---

# 10. Legacy Project

The previous implementation is preserved outside the active repository.

It contains valuable working UI and platform code.

Legacy migration process:

Legacy Implementation
↓
Inspect
↓
Understand
↓
Classify Ownership
↓
Preserve Useful Behaviour
↓
Simplify Where Justified
↓
Migrate Into Correct Layer

Legacy code must not be copied blindly.

Legacy functionality must not be discarded blindly either.

---

# 11. Architecture Change Policy

Implementation tools do not have permission to modify architecture.

If implementation appears to require:

- a new architectural layer
- a new root folder
- cross-layer responsibility movement
- major model redesign
- removal of working functionality
- an architectural dependency not currently defined

implementation must STOP and report the requirement first.

Architecture changes require explicit approval.