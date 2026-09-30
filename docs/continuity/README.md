# RateMyCollezzz Continuity Pack

This package is designed to preserve project context across long ChatGPT conversations, branches, Claude Code sessions, Antigravity sessions, Gemini/Codex handoffs, and future development phases.

Place these files in:

```text
RateMyCollezzz/
└── docs/
    └── continuity/
        ├── PROJECT_MASTER_CONTEXT.md
        ├── CURRENT_STATE.md
        ├── DECISION_LOG.md
        └── CONTINUATION_PROTOCOL.md
```

## Recommended use

At the start of any new chat/session:

1. Read all four continuity files.
2. Read repository `ARCHITECTURE.md`, `IMPLEMENTATION_PLAN.md`, `FILE_MAP.md`.
3. Inspect actual Git status and active-phase files.
4. Continue from `CURRENT_STATE.md`.
5. Do not restart locked work.

## Updating

After every major locked/pushed phase:

- update `CURRENT_STATE.md`
- add new locked decisions to `DECISION_LOG.md`
- update `PROJECT_MASTER_CONTEXT.md` for major architecture/history changes

The continuity pack does not replace Git or the repository. It makes long-term intent and workflow explicit so future agents can correctly interpret the codebase.
