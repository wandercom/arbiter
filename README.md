# Arbiter

**Access auditing, consistency analysis, blast-radius classification, and trust enforcement for the Pact/Baton stack.**

Arbiter is the trust enforcement layer that sits between Pact-built components and Baton-orchestrated circuits. It watches, compares, scores, and gates.

## What It Does

Arbiter continuously answers four questions about every node in a circuit:

1. **Access** -- Did this node touch data it was permitted to touch?
2. **Consistency** -- Does what the node claims match what the adapter observed?
3. **Blast Radius** -- Given a proposed change, what tiers are affected and who needs to know?
4. **Trust** -- Has this node earned the right to be treated as reliable?

## Where It Sits

```
constrain  ->  pact  ->  [arbiter]  ->  baton  ->  production
                            ^
                       stigmergy (analysis)
```

Arbiter is a sidecar to Baton -- not in the hot path. It receives OTLP spans asynchronously, analyzes them, and writes findings to an append-only trust ledger.

## Install

```bash
pip install arbiter
```

## Quick Start

```bash
# Initialize registry, config, and trust ledger
arbiter init

# Register an access graph from Pact
arbiter register path/to/access_graph.json

# Start watching OTLP spans
arbiter watch

# Check trust scores
arbiter trust show <node_id>

# Compute blast radius for a proposed change
arbiter blast-radius <node_id> <version>

# Generate a feedback report
arbiter report --run <run_id>
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `arbiter init` | Initialize registry, config, trust ledger |
| `arbiter register <path>` | Ingest Pact access graph |
| `arbiter trust show <node>` | Current score and history |
| `arbiter trust reset-taint <node> --review <id>` | Clear taint lock |
| `arbiter authority show` | Full authority map |
| `arbiter blast-radius <node> <version>` | Compute blast radius |
| `arbiter soak compute <node> <tier>` | Compute soak duration |
| `arbiter report --run <run_id>` | Generate feedback report |
| `arbiter canary inject --tiers <tier>` | Seed canary data |
| `arbiter canary results --run <run_id>` | Taint escape report |
| `arbiter watch` | Continuous OTLP subscriber mode |
| `arbiter serve` | HTTP API server only |
| `arbiter findings --node <node>` | Consistency findings |
| `arbiter conflicts --unresolved` | List unresolved conflicts |

## HTTP API

Start the API server with `arbiter serve` (default port 7700).

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check and version info |
| POST | `/register` | Register an access graph |
| POST | `/blast-radius` | Compute blast radius for a component |
| GET | `/trust/<node_id>` | Get trust score and tier for a node |
| POST | `/trust/reset-taint` | Clear taint lock after human review |
| POST | `/trust/event` | Accept a trust event from Sentinel |
| GET | `/authority` | Get the current authority map |
| POST | `/canary/inject` | Inject canary data for taint detection |
| POST | `/canary/register-fingerprint` | Register external canary fingerprints (Ledger) |
| GET | `/canary/results/<run_id>` | Get canary escape results for a run |
| POST | `/schema/classification-rules` | Push classification rules (Ledger) |
| GET | `/schema/classification-rules` | Read current classification rules |
| GET | `/report/<run_id>` | Get feedback report for a run |
| POST | `/findings` | Ingest consistency findings |

## Trust Model

Trust is earned, not declared. Every node starts at the configured floor (default 0.1) and its score changes based on observed behavior:

```
trust = age_factor * consistency_factor * taint_factor * review_factor * decay_factor
```

A single canary escape zeroes the taint factor. Recovery requires human review.

| Score | Tier |
|-------|------|
| 0.0 - 0.25 | PROBATIONARY |
| 0.26 - 0.50 | LOW |
| 0.51 - 0.75 | ESTABLISHED |
| 0.76 - 0.90 | HIGH |
| 0.91 - 1.0 | TRUSTED |

Tiers are display labels only. All policy uses the raw score.

## Architecture

```
src/arbiter/
  cli/              # Click CLI
  config/           # arbiter.yaml loading
  models/           # Pydantic models and enums
  registry/         # Access graph, authority map, classifications
  trust/            # Trust score computation
  consistency/      # Consistency analysis engine
  access/           # Access auditing and OpenAPI integration
  taint/            # Canary corpus and taint detection
  blast/            # Blast radius computation
  conflicts/        # Conflict resolution protocol
  stigmergy/        # Signal emission to Stigmergy
  report/           # Feedback report generation
  api/              # HTTP API server
  subscriber/       # OTLP span subscriber
```

## Part of the Stack

Arbiter is one layer in the Pact/Baton trust enforcement stack:

- [Constrain](https://github.com/wandercom/constrain) -- Elicit constraints
- [Pact](https://github.com/wandercom/pact) -- Contract-first build
- **Arbiter** -- Trust enforcement (this project)
- [Baton](https://github.com/wandercom/baton) -- Circuit orchestration

## License

MIT
