# RateMyCollezzz

RateMyCollezzz is a student-first college discovery, review, community, and
trust platform.

The project is being rebuilt using a deliberately controlled five-layer
architecture.

## Architecture

experience/
- Layer 1 — Experience

platform/
- Layer 2 — Django Platform Core

rrm/
- Layer 3 — Review Intelligence / RRM

trust/
- Layer 4 — Trust & Safety

ops/
- Layer 5 — Data / Security / Operations

## Core Principle

Layer 3 understands review content.

Layer 4 decides platform trust actions.

The RRM does not autonomously delete or moderate reviews.

## Current Status

Repository architecture setup.

Feature implementation has not started in the clean architecture.

## Start Here

Read these files in order:

1. ARCHITECTURE.md
2. IMPLEMENTATION_PLAN.md
3. FILE_MAP.md
4. CLAUDE.md

## Legacy Implementation

The earlier project implementation is preserved separately as a legacy
reference.

Useful behaviour may be migrated after inspection, but the old architecture
will not be copied blindly into the new repository.

## Development Philosophy

The project prioritizes:

- understandable structure
- minimal files
- explicit ownership
- controlled implementation scope
- testable components
- no duplicate logic
- no unnecessary abstractions
- research-grounded RRM development
- clear separation between intelligence and trust decisions
