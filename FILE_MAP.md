# RateMyCollezzz — File Map

This document explains where important project files live and what each one
does.

It must be updated whenever meaningful project files are created, moved,
renamed, or removed.

---

# Root Control Files

## ARCHITECTURE.md

Purpose:

Defines the locked five-layer architecture and layer boundaries.

Owner:

Project architecture.

Normal implementation work must not modify this file.

---

## IMPLEMENTATION_PLAN.md

Purpose:

Defines the currently active work and exactly what implementation is allowed.

Owner:

Project planning.

Updated when approved implementation scope changes.

---

## FILE_MAP.md

Purpose:

Acts as the navigation map for the repository.

Every important project file should eventually be discoverable through this
document.

---

## CLAUDE.md

Purpose:

Defines mandatory behaviour for Claude Code or other implementation agents.

It prevents:

- architecture expansion
- uncontrolled file creation
- scope expansion
- unrelated refactoring
- duplicate logic
- future-feature implementation

---

## README.md

Purpose:

Human-facing project overview and navigation entry point.

---

## .gitignore

Purpose:

Prevents generated, local, secret, temporary, and large runtime artifacts from
being committed.

---

# Architectural Folders

## experience/

Layer:

Layer 1 — Experience

Current state:

EMPTY / NOT STARTED

Current file:

.gitkeep

---

## platform/

Layer:

Layer 2 — Platform Core

Current state:

EMPTY / NOT STARTED

Current file:

.gitkeep

---

## rrm/

Layer:

Layer 3 — Review Intelligence / RRM

Current state:

EMPTY / NOT STARTED

Current file:

.gitkeep

---

## trust/

Layer:

Layer 4 — Trust & Safety

Current state:

EMPTY / NOT STARTED

Current file:

.gitkeep

---

## ops/

Layer:

Layer 5 — Data / Security / Operations

Current state:

EMPTY / NOT STARTED

Current file:

.gitkeep

---

# Legacy Project

Previous project implementation is intentionally stored outside this
repository.

It is reference material only.

No legacy file should be migrated without:

1. inspection
2. architectural classification
3. usefulness review
4. approval
5. controlled migration or reconstruction
