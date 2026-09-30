# RateMyCollezzz — Continuation Protocol

This file defines how future chats and coding agents must continue the project without losing context.

---

# 1. NEW CHAT STARTUP ORDER

A new ChatGPT/Claude/Gemini/Codex session should read:

1. `PROJECT_MASTER_CONTEXT.md`
2. `CURRENT_STATE.md`
3. `DECISION_LOG.md`
4. `CONTINUATION_PROTOCOL.md`
5. repo `ARCHITECTURE.md`
6. repo `IMPLEMENTATION_PLAN.md`
7. repo `FILE_MAP.md`
8. active-phase code/tests
9. current Git status/diff

Then report:

1. what RateMyCollezzz is
2. five-layer architecture
3. locked phases
4. active phase
5. latest exact known commit
6. current blockers
7. exact next action

---

# 2. MASTER NEW-CHAT PROMPT

Paste this into a future chat:

```text
This is a continuation of the RateMyCollezzz project.

Before implementation, read:

docs/continuity/PROJECT_MASTER_CONTEXT.md
docs/continuity/CURRENT_STATE.md
docs/continuity/DECISION_LOG.md
docs/continuity/CONTINUATION_PROTOCOL.md

Then inspect:

ARCHITECTURE.md
IMPLEMENTATION_PLAN.md
FILE_MAP.md

Do not restart completed work.

First tell me:

1. what RateMyCollezzz is,
2. the complete five-layer architecture,
3. which RRM phases are locked,
4. the current active phase,
5. latest exact known commit,
6. current blockers,
7. exact next action.

Treat continuity docs and actual repository state as canonical.

If they conflict, inspect Git/code/tests and report the conflict instead of guessing.
```

---

# 3. STANDARD PHASE WORKFLOW

## Step 1 — Audit/design

Inspect:

- current architecture
- previous locked contracts
- phase ownership
- forbidden scope
- open ambiguity

## Step 2 — Freeze contract

Freeze:

- files
- inputs
- outputs
- schemas
- dimensions
- metrics
- tests
- dependencies
- stop conditions

## Step 3 — Implementation prompt

Include:

- repo
- active phase
- latest locked commit
- previous test baseline
- allowed files
- forbidden files
- exact behavior
- validation
- no commit/push

## Step 4 — Agent implementation

Agent may:

- read
- edit allowed files
- run tests
- inspect Git

Agent must not silently redesign locked architecture.

## Step 5 — User pastes report back

The user often pastes agent output with minimal explanation.

Interpret this as:

> Audit the work against the locked contract and tell me whether it is ready.

## Step 6 — Corrective micro-gate

If issues remain:

- give narrow fix prompt
- do not restart whole phase

## Step 7 — Final validation

Typical gates:

- focused tests
- full RRM suite
- runtime check
- requirements diff
- `git diff --check`
- artifact scan
- final `git status`

## Step 8 — Commit/push

Only after explicit approval.

Capture:

- exact hash
- message
- branch
- remote sync

## Step 9 — Update continuity docs

Update:

- `CURRENT_STATE.md`
- `DECISION_LOG.md` if decisions changed
- `PROJECT_MASTER_CONTEXT.md` if history/architecture changed

---

# 4. USER WORKING STYLE

The user prefers:

- direct answers
- exact next steps
- copyable prompts
- phased implementation
- complete current layer before next
- detailed report audits
- no unnecessary clarification when context is sufficient
- Hinglish when requested
- no restart after interruptions

When the user writes:

- “ab?”
- “done?”
- “continue”
- “you were interrupted”
- pastes an agent transcript

continue from actual project state.

Do not restart from phase zero.

---

# 5. CODING-AGENT PROMPT STYLE

Every coding prompt should include:

## Context

- project name
- repo
- current phase
- latest locked commit

## File scope

- create
- modify
- do-not-touch

## Contract

- exact labels
- dimensions
- constants
- semantics

## Validation

- focused tests
- full regression
- runtime
- requirements
- diff
- artifact scan
- status

## Safety

Use where relevant:

```text
DO NOT COMMIT.
DO NOT PUSH.
DO NOT RESET.
DO NOT RESTORE THE WORKING TREE.
DO NOT DELETE UNRELATED FILES.
DO NOT INSTALL NEW PACKAGES WITHOUT APPROVAL.
```

---

# 6. SWITCHING AGENTS

When moving between:

- Claude Code
- Claude Opus/Sonnet in Antigravity
- Gemini
- Codex
- another agent

do NOT say:

> Start RRM 3.10.

Say:

> Continue current partially implemented RRM 3.10 from the existing working tree.

Require first:

```text
git status --short
git diff -- FILE_MAP.md rrm/
```

Then inspect current partial files.

Give last verified test result.

Do not let the new agent regenerate everything.

Avoid repeatedly switching agents during the same half-finished phase.

---

# 7. CURRENT RRM 3.10 HANDOFF TEMPLATE

```text
RRM 3.10 — CONTINUE CURRENT WORKING TREE

DO NOT restart RRM 3.10.
DO NOT reset/restore/checkout current partial work.
DO NOT redesign the locked scientific protocol.
DO NOT commit.
DO NOT push.

Repository:
C:\Projects\RateMyCollezzz

Latest completed/pushed RRM phase:
RRM 3.9

Commit:
81cf7ce

RRM 3.10 protocol:
LOCKED

First:

git status --short
git diff -- FILE_MAP.md rrm/

Read current:

rrm/scientific_evaluation.py
rrm/experiment_results.py
rrm/tests/test_scientific_evaluation.py

Check whether:
rrm/tests/test_experiment_results.py
exists.

Last verified test:

114 passed
1 failed

Failure:
TestLineageLeakage.test_deterministic_ordering

Known implementation concerns:

- deterministic lineage ordering
- ScientificMacroResult vs old MacroMetrics
- private/nonexistent serialization helper imports
- scientific-status ownership
- no circular imports
- smoke result macro construction
- near-duplicate candidate completeness
- missing test_experiment_results.py
- FILE_MAP not finalized

Finish focused tests, full regression, runtime/hygiene gates, then STOP and report.

DO NOT COMMIT.
DO NOT PUSH.
```

---

# 8. GIT RULES

Allowed by default:

- `git status`
- `git diff`
- `git diff --check`
- `git log`
- `git show`
- `git rev-parse`

Avoid without approval:

- commit
- push
- reset
- restore whole tree
- destructive checkout
- clean
- force push
- history rewrite

Never invent commit hashes.

---

# 9. TEST INTERPRETATION

Do not:

- call skipped tests passes
- treat line-ending warnings as code failures automatically
- use synthetic smoke output as performance evidence
- predict new full test count before running

Always report actual counts.

---

# 10. ARTIFACT HYGIENE

Scan task-generated artifacts:

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

Exclude:

```text
.venv/
.git/
__pycache__/
.pytest_cache/
```

Do not blindly delete.

Report exact paths.

---

# 11. STATUS DISCIPLINE

Keep these states separate:

## Design LOCKED

Contract frozen.

## IMPLEMENTED

Code exists.

## COMMITTED + PUSHED

Validated code is versioned remotely.

Never collapse these.

Example:

```text
RRM 3.10 scientific protocol = LOCKED
RRM 3.10 infrastructure = PARTIAL
RRM 3.10 Git commit = NOT DONE
```

---

# 12. “PROJECT COMPLETE” DISCIPLINE

Do not say RateMyCollezzz is complete when only RRM ends.

After RRM:

- production data/model work remains
- Trust remains
- backend remains
- complete UI remains
- verified college content remains
- security/ops remains
- end-to-end QA remains

---

# 13. CHAT BRANCH STRATEGY

A ChatGPT branch is useful but not sufficient.

Branches do not continuously sync after divergence.

Use branches for:

- alternate exploration
- backup continuation point

Use repository continuity docs as durable source of truth.

Suggested conversation naming:

```text
RateMyCollezzz — Master Continuation 01
RateMyCollezzz — Continuation 02
RateMyCollezzz — Production RRM Training
RateMyCollezzz — Full UI Phase
```

---

# 14. CONTINUITY UPDATE CHECKLIST

After a major milestone:

- [ ] active phase updated
- [ ] exact commit captured
- [ ] test baseline updated
- [ ] blockers updated
- [ ] production model status updated
- [ ] new locked decisions logged
- [ ] UI/backend/Trust progress updated
- [ ] exact next action recorded

---

# 15. IF CONTEXT IS UNCERTAIN

Do not guess.

Inspect:

```text
git status
git log
git show
ARCHITECTURE.md
IMPLEMENTATION_PLAN.md
FILE_MAP.md
tests
actual source
```

Repository evidence outranks chat assumptions.
