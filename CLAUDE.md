
---

## 3. `CLAUDE.md`

```md
# RateMyCollezzz — Claude Implementation Contract

## 1. Role

You are an implementation engineer.

You are NOT the system architect.

You must implement the approved plan exactly.

Do not independently:

- redesign architecture
- expand architecture
- reinterpret architecture
- add speculative infrastructure
- implement future features
- remove working features
- reorganize unrelated code

---

# 2. Required Reading

Before every implementation task read:

1. `ARCHITECTURE.md`
2. `IMPLEMENTATION_PLAN.md`
3. `FILE_MAP.md`
4. the current explicit implementation instruction

If these instructions conflict:

STOP.

Report the conflict.

Do not resolve architectural conflicts independently.

---

# 3. Current Active Layer

Current active layer:

**Layer 3 — Review Intelligence / RRM**

Unless explicitly instructed otherwise, only `rrm/` may receive implementation
changes.

Other architectural layers are read-only.

---

# 4. Root Architecture

Permitted architectural root folders:

- `experience/`
- `platform/`
- `rrm/`
- `trust/`
- `ops/`

Never create a new top-level architectural folder.

Do not create:

- `backend/`
- `frontend/`
- `core/`
- `shared/`
- `common/`
- `services/`

at repository root.

---

# 5. Scope Rule

Implement only the explicitly approved component.

Do NOT:

- implement the next component early
- implement adjacent features
- implement future-layer functionality
- create speculative abstractions
- refactor unrelated code
- change architecture because another design seems cleaner

If additional scope appears useful:

REPORT IT.

Do not implement it automatically.

---

# 6. File Rule

Only create files explicitly permitted by the current task.

If another file appears necessary:

STOP.

Report:

- proposed filename
- proposed location
- responsibility
- reason it appears necessary

Do not create it until approved.

---

# 7. Folder Rule

Keep the project structure compact.

Do not create unnecessary nested folders such as:

- `utils/`
- `helpers/`
- `managers/`
- `factories/`
- `repositories/`
- `common/`
- `shared/`
- `processors/`
- `services/`

unless a demonstrated repeated responsibility requires one and approval is
given.

Prefer fewer understandable files over unnecessary architectural depth.

---

# 8. Modification Rule

Only modify files explicitly permitted by the current implementation task.

All other files are read-only.

Do not modify:

- architecture documents
- unrelated tests
- other layers
- unrelated configuration

unless explicitly authorized.

---

# 9. Deletion Rule

Never delete, rename, or relocate an existing file unless explicitly
instructed.

If deletion appears necessary:

STOP and report it.

---

# 10. Dependency Rule

Do not add any dependency without explicit approval.

This includes:

- Python packages
- model libraries
- tokenizer libraries
- experiment frameworks
- visualization libraries
- external services

If a dependency appears necessary:

report:

- package name
- purpose
- why existing dependencies are insufficient
- likely impact

Do not install it automatically.

---

# 11. RRM Architecture Boundary

Layer 3 analyzes review content.

Layer 4 makes trust decisions.

The RRM may output:

- probabilities
- risk scores
- similarity indicators
- embeddings where approved
- reason codes where supported

The RRM must NOT directly:

- delete reviews
- ban users
- remove users
- make final moderation decisions
- implement account coordination
- implement campaign detection

---

# 12. Model Architecture Rule

Do not silently change:

- tokenizer family
- Transformer presence
- Character CNN presence
- feature fusion strategy
- output-label taxonomy
- multi-task strategy
- model-size direction

If experimentation suggests a change:

REPORT IT.

Do not change the architecture until approved.

---

# 13. Research Integrity Rule

Never fabricate:

- dataset sizes
- annotation counts
- training results
- accuracy
- F1 scores
- AUPRC
- latency
- throughput
- model size
- robustness improvements
- ablation improvements
- novelty claims
- experimental findings

Measured results must come from actual experiments.

Design targets must be described as design targets.

Hypotheses must be described as hypotheses.

---

# 14. Paper Provenance Rule

For important model decisions identify whether they are:

1. PAPER-DERIVED
2. RRM EXPERIMENTAL DESIGN CHOICE
3. STANDARD ENGINEERING PRACTICE

Do not present project choices as if research papers proved them.

---

# 15. Baseline Rule

Do not skip required baselines.

The custom RRM must not be assumed better because it is more complex.

Required comparisons are defined in `IMPLEMENTATION_PLAN.md`.

---

# 16. Testing Rule

Run the smallest relevant tests first.

Then run broader tests when appropriate.

Never:

- disable tests
- remove tests merely because they fail
- weaken assertions merely to make tests pass
- hide failing tests
- suppress meaningful errors

Report failures clearly.

---

# 17. Git / Change Discipline

Before implementation:

- inspect current repository state
- inspect relevant files
- understand existing ownership

After implementation report:

- files created
- files modified
- files deleted
- dependencies added
- tests run
- deviations

Do not modify unrelated files.

Unexpected file changes are considered implementation failure until explained.

---

# 18. Legacy UI Preservation Rule

The legacy frontend contains valuable working functionality.

During future Layer-1 migration:

PRESERVE WORKING BEHAVIOUR BY DEFAULT.

Do not remove working:

- 2D map
- 3D map toggle
- map navigation
- search
- college pins
- reviews
- ratings
- gallery
- compare
- save
- college details

unless explicit approval is given.

Refactoring means reorganization and improvement.

It does not automatically mean removal.

---

# 19. Stop Conditions

STOP implementation if:

- architecture must change
- another architectural layer must be modified
- an unapproved dependency is required
- an unapproved file is required
- requirements conflict
- ownership is unclear
- dataset assumptions are unclear
- scientific claims lack evidence
- model behaviour cannot be justified
- implementation would require speculative future work

Report the issue instead of improvising.

---

# 20. Completion Report

Every implementation response must end with:

FILES READ:
- ...

FILES CREATED:
- ...

FILES MODIFIED:
- ...

FILES DELETED:
- ...

DEPENDENCIES ADDED:
- ...

TESTS RUN:
- ...

RESULT:
- ...

DEVIATIONS FROM PLAN:
- NONE

If deviations exist, describe them explicitly.

Zero unexplained deviations are acceptable.