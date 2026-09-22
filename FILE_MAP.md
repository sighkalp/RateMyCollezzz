# RateMyCollezzz — File Map

This document is the navigation index for the repository.

Its purpose is simple:

**If someone asks where something lives, this file should help them find it.**

Update this file whenever meaningful project files are:

- created
- moved
- renamed
- removed

Do not fill this document with speculative future files.

Only document files that actually exist or have been explicitly approved.

---

## Root Control Files

### `ARCHITECTURE.md`

**Layer:** Project-wide control

**Purpose:** Defines the stable five-layer architecture.

Defines:

- layer ownership
- layer boundaries
- RRM / Trust separation
- 2D + 3D map requirement
- UI migration policy
- repository structure
- legacy migration policy
- architecture change policy

**Must NOT:** Be modified by normal implementation tasks.

---

### `IMPLEMENTATION_PLAN.md`

**Layer:** Project-wide control

**Purpose:** Defines what is currently being built.

Contains:

- active layer
- active component
- RRM build sequence
- allowed work
- forbidden work
- experiment requirements
- completion rules
- learning/teaching gate

This is where implementation priority changes.

---

### `CLAUDE.md`

**Layer:** Project-wide control

**Purpose:** Defines mandatory implementation-agent behaviour.

Prevents:

- architecture expansion
- scope expansion
- unapproved files
- unapproved folders
- unnecessary dependencies
- unrelated refactoring
- fabricated research results
- uncontrolled feature removal
- uncontrolled Git operations

---

### `FILE_MAP.md`

**Layer:** Project-wide control

**Purpose:** Repository navigation and file ownership map.

This file should remain concise and accurate.

---

### `README.md`

**Layer:** Project-wide overview

**Purpose:** Human-facing overview of RateMyCollezzz.

Explains:

- what the project is
- five architectural layers
- current development focus
- current RRM direction
- where to start reading

---

### `.gitignore`

**Layer:** Project-wide repository hygiene

**Purpose:** Prevent local/generated content from entering Git.

Examples:

- secrets
- virtual environments
- Python caches
- Node modules
- build output
- local databases
- model checkpoints
- temporary artifacts
- logs

---

## Layer 1 — `experience/`

**Ownership:** User-facing experience.

**Current state:** Placeholder only unless `IMPLEMENTATION_PLAN.md` activates Layer 1.

**Currently existing tracked placeholder:** `experience/.gitkeep`

Known responsibilities:

- Discover UI
- search
- filters
- map
- 2D navigation
- 3D perspective navigation
- college pins
- college details
- reviews UI
- gallery UI
- compare
- saved colleges
- profile UI
- notifications UI
- community UI

Legacy frontend exists externally and will later be migrated deliberately.

Do not pre-create speculative Layer-1 files while RRM work is active.

---

## Layer 2 — `platform/`

**Ownership:** Django Platform Core.

**Current state:** Placeholder only unless activated.

**Currently existing tracked placeholder:** `platform/.gitkeep`

Known responsibilities:

- accounts
- profiles
- colleges
- locations
- reviews
- ratings
- helpful votes
- saved colleges
- communities
- confessions
- discussions
- chat
- notifications
- reports
- moderation records
- platform APIs
- permissions
- business logic

Do not pre-create speculative Layer-2 files while RRM work is active.

---

## Layer 3 — `rrm/`

**Ownership:** Review Intelligence / Review Risk Model.

**Current state:** ACTIVE

**Current active component:** RRM 3.1 — Research Contract

**Currently existing tracked placeholder:** `rrm/.gitkeep`

Expected responsibilities will eventually include:

- research configuration
- dataset handling
- annotation support
- deterministic pre-checks
- duplicate/similarity evidence
- baseline models
- tokenizer
- Transformer encoder
- Character CNN
- feature fusion
- multi-task heads
- training
- evaluation
- robustness testing
- inference

Exact files must be added incrementally.

Do **not** generate the entire RRM folder tree in advance.

Every new file must have:

- one clear responsibility
- an approved reason to exist
- an entry in this file

Remove `rrm/.gitkeep` only when the first approved real RRM file is added.

---

## Layer 4 — `trust/`

**Ownership:** Trust & Safety.

**Current state:** Placeholder only unless activated.

**Currently existing tracked placeholder:** `trust/.gitkeep`

Future responsibilities may include:

- individual risk aggregation
- behavioural analysis
- temporal analysis
- coordination analysis
- risk resolution
- moderation decisions
- campaign detection
- reputation signals

Layer 4 consumes RRM signals.

It does not belong inside `rrm/`.

Do not implement Trust algorithms while Layer 3 is the active scope unless explicitly authorized.

---

## Layer 5 — `ops/`

**Ownership:** Data, Security, and Operations.

**Current state:** Placeholder only unless activated.

**Currently existing tracked placeholder:** `ops/.gitkeep`

Future responsibilities include:

- PostgreSQL
- Redis
- security configuration
- authentication infrastructure
- WebSocket infrastructure
- audit
- backups
- monitoring
- model storage
- deployment
- operational performance

Do not pre-create operational infrastructure during RRM research unless explicitly required.

---

## Legacy Implementation

The legacy implementation is intentionally outside the active repository.

Known local backup created during migration:

`C:\Projects\RateMyCollezzz_LEGACY_20260922_115705`

A ZIP backup also exists locally.

This path is local recovery information, not a portable repository dependency.

Legacy code is reference and migration material.

Migration rule:

```text
Inspect
  ↓
Understand
  ↓
Classify
  ↓
Preserve useful behaviour
  ↓
Simplify where justified
  ↓
Migrate deliberately
```

Legacy code must not be blindly copied.

Legacy functionality must not be blindly discarded.

---

## File Registration Format

Whenever a meaningful new file is approved, document it using:

```md
### `path/to/file.py`

**Layer:** Layer X

**Purpose:** One concise description.

**Used by:** Relevant components.

**Must NOT:** Responsibility that belongs elsewhere.
```

This format keeps ownership obvious.
