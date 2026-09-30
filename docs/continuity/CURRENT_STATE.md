# RateMyCollezzz — Current State

**Last continuity update:** 2026-09-29

This file answers one question:

> Where exactly is the project right now?

Update it after every major phase, commit, push, or material blocker.

---

# 1. CURRENT RRM STATUS

```text
RRM 3.1   LOCKED
RRM 3.2   LOCKED
RRM 3.3   LOCKED
RRM 3.4   LOCKED
RRM 3.5   LOCKED
RRM 3.6   LOCKED
RRM 3.7   LOCKED
RRM 3.8   LOCKED

RRM 3.9   LOCKED + COMMITTED + PUSHED
          commit: 81cf7ce

RRM 3.10  SCIENTIFIC PROTOCOL LOCKED
          INFRASTRUCTURE IMPLEMENTATION IN PROGRESS
          NOT COMMIT-READY

RRM 3.11  NOT STARTED
```

Do not activate 3.11 yet.

---

# 2. LAST FULLY COMPLETED/PUSHED PHASE

## RRM 3.9 — Multi-task Heads + Training Infrastructure

Commit:

```text
81cf7ce
```

Message:

```text
feat(rrm): add multitask heads and training infrastructure
```

Locked regression baseline before RRM 3.10:

```text
853 passed
2 known pre-existing RoBERTa scheduler warnings
```

---

# 3. ACTIVE PHASE — RRM 3.10

## Scientific protocol

Status:

```text
LOCKED
```

Key frozen rules:

- six labels remain canonical
- UNKNOWN = -1
- primary metric = macro-AUPRC
- checkpoint selection = validation masked BCE
- one-class task => discriminative metrics None
- no arbitrary 5/5 support cutoff
- fixed threshold = 0.5
- decision uses `>=`
- validation-only threshold tuning
- initial neural comparison = 3 seeds
- bootstrap:
  - seed 42
  - 1000 reps
  - 95% percentile CI
- required ablation:
  - semantic-only vs full semantic+char
- production scientific evaluation remains blocked

---

# 4. PARTIAL RRM 3.10 IMPLEMENTATION

At the last verified handoff, these partial/untracked files existed:

```text
rrm/scientific_evaluation.py
rrm/experiment_results.py
rrm/tests/test_scientific_evaluation.py
```

The following had not yet been completed:

```text
rrm/tests/test_experiment_results.py
```

`FILE_MAP.md` had not yet been finalized for RRM 3.10.

No RRM 3.10 commit has been made.

---

# 5. LAST VERIFIED FOCUSED TEST RESULT

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest rrm/tests/test_scientific_evaluation.py -q --cache-clear
```

Last verified result:

```text
114 passed
1 failed
```

Failure:

```text
TestLineageLeakage.test_deterministic_ordering
```

Observed:

```text
expected record_id_a == "r1"
actual   record_id_a == "r3"
```

The coding agent diagnosed insertion/input-order dependence in lineage leakage reporting.

After this point, additional edits were attempted, but the coding-agent quota ended before a clean rerun.

Therefore:

> **Inspect the true current working tree before doing anything else.**

---

# 6. KNOWN RRM 3.10 ISSUES TO AUDIT

## A. Deterministic lineage ordering

`check_lineage_leakage()` must be independent of input order.

Do not weaken the test.

## B. Macro type mismatch

RRM 3.10 owns a scientific macro result with:

```text
macro_f1
macro_f1_task_count
macro_f1_task_names

macro_auprc
macro_auprc_task_count
macro_auprc_task_names
```

Partial `experiment_results.py` was observed using old RRM 3.9 `MacroMetrics` while rendering task-name fields that old type does not own.

Preferred fix:

```text
ScientificResult -> RRM 3.10 ScientificMacroResult
```

Do not modify locked `rrm/evaluation.py` merely to add name fields.

## C. Private/nonexistent helper imports

Partial `experiment_results.py` was observed trying to import:

```text
_task_metrics_to_dict
_macro_metrics_to_dict
_provenance_to_dict
```

from `scientific_evaluation.py`.

Avoid unnecessary private cross-module coupling.

Serialization helpers belong locally in `experiment_results.py` or should use explicit `asdict`.

## D. Dependency direction

Preferred:

```text
scientific_evaluation.py
        ↓
experiment_results.py
```

`experiment_results.py` may import public scientific structures.

`scientific_evaluation.py` should not import `experiment_results.py`.

No circular import.

## E. Scientific status ownership

Use one canonical owner for:

```text
NON_SCIENTIFIC_SYNTHETIC_SMOKE
```

and related statuses.

Do not duplicate independent definitions across both modules.

## F. Smoke macro path

Audit smoke result construction.

Use the RRM 3.10 scientific macro result path.

## G. Near-duplicate reporting

Audit whether implementation reports only one best match per query.

The leakage contract should not silently hide relevant unresolved cross-split candidates.

Reuse public similarity primitives.

---

# 7. EXACT NEXT STEPS

Start with:

```powershell
cd C:\Projects\RateMyCollezzz

git status --short
git diff -- FILE_MAP.md rrm/
```

Read actual current:

```text
rrm/scientific_evaluation.py
rrm/experiment_results.py
rrm/tests/test_scientific_evaluation.py
```

Check whether:

```text
rrm/tests/test_experiment_results.py
```

exists.

## Focused test 1

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest rrm/tests/test_scientific_evaluation.py -q --cache-clear
```

Fix genuine current failures.

## Focused test 2

Create/finish:

```text
rrm/tests/test_experiment_results.py
```

Then run:

```powershell
.\.venv\Scripts\python.exe -m pytest rrm/tests/test_experiment_results.py -q --cache-clear
```

## FILE_MAP

Register once:

```text
rrm/scientific_evaluation.py
rrm/experiment_results.py
rrm/tests/test_scientific_evaluation.py
rrm/tests/test_experiment_results.py
```

Active component:

```text
RRM 3.10 - Scientific Evaluation Infrastructure
```

Do not activate RRM 3.11.

## Full regression

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest rrm/tests/ -q --cache-clear
```

Required:

- all previous 853 tests still pass
- all new 3.10 tests pass
- only the same 2 known RoBERTa warnings remain
- do not predict final count

## Runtime / hygiene

Run:

```powershell
.\.venv\Scripts\python.exe rrm/runtime_check.py

git diff -- rrm/requirements.txt

git diff --check

git status --short
```

Artifact scan must exclude:

```text
.venv/
.git/
__pycache__/
.pytest_cache/
```

and inspect task-generated:

```text
*.pt
*.pth
*.bin
*.safetensors
*.model
*.vocab
tokenizer_manifest.json
_tmp*.py
```

Do not leave generated evaluation JSON/Markdown in repo unless explicitly intended.

---

# 8. COMMIT POLICY

Do not commit/push RRM 3.10 until:

1. focused scientific-evaluation tests pass
2. experiment-results tests pass
3. full RRM regression passes
4. runtime check passes
5. requirements unchanged
6. diff check clean
7. artifact scan clean
8. final status limited to intended files
9. ChatGPT/user audits final report

Possible eventual commit message:

```text
feat(rrm): add scientific evaluation infrastructure
```

Do not commit until approved.

---

# 9. PRODUCTION MODEL STATUS

Still true:

```text
production corpus:
NOT AVAILABLE

production tokenizer:
NOT TRAINED

RRM 3.6 production pretraining:
NOT EXECUTED

production semantic checkpoint:
NOT AVAILABLE

real labeled production dataset:
NOT AVAILABLE

production supervised RRM:
NOT TRAINED

real final test evaluation:
NOT EXECUTED

scientific production performance claims:
NONE
```

---

# 10. NEXT PHASE AFTER RRM 3.10

Only after 3.10 is:

```text
reviewed
tested
committed
pushed
LOCKED
```

move to:

```text
RRM 3.11 — Packaging + Layer Contract
```

3.11 should package stable RRM outputs/contracts.

It must not silently start production model training.

---

# 11. BROADER REMAINING PROJECT WORK

Even after RRM 3.11:

## Model/data

- real RMC dataset
- annotation
- frozen splits
- production tokenizer
- semantic pretraining
- supervised training
- 3-seed experiments
- semantic-only ablation
- fair baselines
- final scientific evaluation
- model freeze

## Trust

- evidence-to-policy integration
- moderation logic
- appeals
- auditability

## Backend

- full Django platform
- auth/permissions
- college data
- review lifecycle
- helpful votes
- reports
- moderation
- notifications
- analytics
- production DB/security

## UI

- full multi-page product
- not only Discover
- landing
- search
- college detail
- review flows
- profile
- admin/moderation
- contact/about/legal
- mobile/accessibility/loading/error/empty states

## Content

- official logos
- verified gallery
- structured college data
- no wrong-image filling

## Ops

- production deployment
- monitoring
- backups
- security
- logs
- health checks
