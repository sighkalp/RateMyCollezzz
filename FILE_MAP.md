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

# Root Control Files

## `ARCHITECTURE.md`

Purpose:

Defines the permanent five-layer architecture.

Defines:

- layer ownership
- layer boundaries
- RRM / Trust separation
- 2D + 3D map requirement
- UI migration policy
- repository structure
- legacy migration policy
- architecture change policy

Normal implementation tasks must not modify it.

---

## `IMPLEMENTATION_PLAN.md`

Purpose:

Defines what is currently being built.

Contains:

- active layer
- active objectives
- RRM build sequence
- allowed work
- forbidden work
- experiment requirements
- completion rules

This is where implementation priority changes.

---

## `CLAUDE.md`

Purpose:

Defines mandatory implementation-agent behaviour.

Prevents:

- architecture expansion
- scope expansion
- unapproved files
- unapproved folders
- unnecessary dependencies
- unrelated refactoring
- fabricated research results
- uncontrolled feature removal

---

## `FILE_MAP.md`

Purpose:

Repository navigation.

This file should remain concise and accurate.

---

## `README.md`

Purpose:

Human-facing overview of RateMyCollezzz.

Explains:

- what the project is
- five architectural layers
- current model direction
- where to start reading

---

## `.gitignore`

Purpose:

Prevent local/generated content from entering Git.

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

# Layer 1 — `experience/`

Ownership:

User-facing experience.

Current implementation priority:

Defined by `IMPLEMENTATION_PLAN.md`.

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

# Layer 2 — `platform/`

Ownership:

Django platform core.

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

# Layer 3 — `rrm/`

Ownership:

Review Intelligence / Review Risk Model.

Current development priority:

See `IMPLEMENTATION_PLAN.md`.

Expected responsibilities will eventually include:

- research configuration
- dataset handling
- annotation support
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

Do NOT generate the entire RRM folder tree in advance.

Every new file must have:

- one clear responsibility
- an approved reason to exist
- an entry in this file

---

# Layer 4 — `trust/`

Ownership:

Trust & Safety.

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

Do not implement Trust algorithms while Layer 3 is the active scope unless
explicitly authorized.

---

# Layer 5 — `ops/`

Ownership:

Data, Security, and Operations.

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

Do not pre-create operational infrastructure during RRM research unless
explicitly required.

---

# Legacy Implementation

Preserved external project:

`C:\Projects\RateMyCollezzz_LEGACY_20260922_115705`

A ZIP backup also exists.

Legacy code is reference and migration material.

Migration rule:

Inspect
→ Understand
→ Classify
→ Preserve useful behaviour
→ Simplify where justified
→ Migrate deliberately

Legacy code must not be blindly copied.

Legacy functionality must not be blindly discarded.

---

# File Registration Format

Whenever a meaningful new file is approved, document it using:

## `path/to/file.py`

Layer:

Layer X

Purpose:

One concise description.

Used by:

Relevant components.

Must NOT:

Responsibility that belongs elsewhere.

This format keeps ownership obvious.