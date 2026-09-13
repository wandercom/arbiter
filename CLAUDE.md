# Arbiter

Access auditing, consistency analysis, blast-radius classification, and trust enforcement for the Pact/Baton stack. Append-only trust ledger with SHA256 integrity.

## Quick Reference

```bash
arbiter init                       # create arbiter.yaml
arbiter register <access_graph>    # ingest access graph from Pact
arbiter trust <node_id>            # show trust score + history
arbiter blast-radius <change>      # compute blast radius
arbiter canary inject              # seed canary corpus
arbiter canary results <run_id>    # taint escape report
arbiter report <run_id>            # full feedback report
arbiter serve                      # HTTP API (port 7700)
python3 -m pytest --import-mode=importlib tests/ -v  # full artifact sweep; see tests/README.md
```

## Architecture

### Trust Model
Trust is computed, never declared. Six factors multiply:

```
trust(node) = base_weight * age_factor * consistency_factor * taint_factor * review_factor * decay_factor
```

- Floor: 0.1, Ceiling: 1.0 (asymptotic)
- Display tiers: PROBATIONARY (0-0.25), LOW (0.26-0.50), ESTABLISHED (0.51-0.75), HIGH (0.76-0.90), TRUSTED (0.91-1.0)
- **Policy always uses raw scores, never tiers**

### Trust Ledger
Append-only JSONL with SHA256 checkpoints every 100 entries. Fields: ts, node, event, weight, score_before, score_after, sequence_number, detail. Immutable (Pydantic frozen=True).

### Conflict Resolution
Three-step protocol: authority (who owns the domain?) -> trust (who has higher score?) -> human (gate for manual review).

### Data Flow
```
Pact (access_graph.json) -> Registry (authority map, classification)
Baton (OTLP spans) -> Consistency analysis (adapter I/O vs claims)
Canary corpus -> Taint scanner -> Violations -> Trust penalty
Findings -> Stigmergy (signals) + Human (reports)
```

## Structure

```
src/arbiter/
  access/          # Access auditing (classifier, auditor, walker)
  api/             # Flask HTTP API (port 7700)
  blast/           # Blast radius (engine, classification, soak, traversal)
  cli/             # Click CLI
  config/          # Configuration (loader, models)
  conflicts/       # Conflict resolution (detector, resolver, protocols)
  consistency/     # Consistency analysis (analyzer, store)
  models/          # Pydantic models (enums, trust, signals, findings, graph, canary)
  registry/        # Access graph store, authority map, classification
  report/          # Feedback report generation
  stigmergy/       # Signal emission (fire-and-forget HTTP POST)
  subscriber/      # OTLP span subscriber
  taint/           # Canary corpus, taint scanner
  trust/           # Trust engine, factors, ledger (SHA256 checksums)
```

## HTTP API

| Method | Path | Purpose |
|--------|------|---------|
| GET | /health | Health check |
| POST | /register | Ingest access_graph.json |
| POST | /blast-radius | Compute blast radius |
| GET | /trust/<node_id> | Trust score + history |
| POST | /trust/reset-taint | Clear taint lock |
| GET | /authority | Full authority map |
| POST | /canary/inject | Seed canary corpus |
| GET | /canary/results/<run_id> | Taint escape report |
| GET | /report/<run_id> | Full feedback report |
| POST | /findings | Receive OTLP span JSON |

## Integrations

| System | Direction | What | Degradation |
|--------|-----------|------|-------------|
| Pact | Pull | access_graph.json (authority declarations) | Required |
| Baton | Pull | OTLP spans (adapter I/O ground truth) | Optional |
| Stigmergy | Push | Signals (violations, trust changes) | Fire-and-forget, skip if unavailable |
| Sentinel | Peer | Trust events inform attribution severity | Indirect |

## Constraints (C001-C012)

- C001-C009: MUST (data access detection, trust integrity, canary tracking, authority exclusivity, declaration completeness, ledger immutability, consistency verification, human gates, conflict resolution)
- C010-C012: SHOULD (trust factor boundaries, blast radius computation, span enrichment)

## Conventions

- Python 3.12+, Pydantic v2, Click, Flask, hatchling, pytest
- Synchronous (no async — appropriate for CLI/sidecar)
- All policy uses raw trust scores, never display tiers
- Append-only trust ledger with SHA256 integrity checkpoints
- Fire-and-forget for Stigmergy (2s timeout, daemon thread)
- All file I/O via pathlib, UTC timestamps everywhere
- Tests: 855 current-package cases verified; unported generated suites remain. See tests/README.md for exact commands and limits.
- 30 functional assertions (FA-A-001 through FA-A-030)

## Kindex

Arbiter captures discoveries, decisions, and trust model rationale in [Kindex](~/WanderRepos/repos/kindex). Search before adding. Link related concepts.
