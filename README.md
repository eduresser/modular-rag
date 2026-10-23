# modular-rag

Retrieval-augmented search over a document corpus, served as an HTTP API and as an MCP server. One codebase runs any number of deployments; each deployment is a folder with one YAML file.

## Overview

- **Sources.** Google Drive folders, enumerated incrementally with a change feed.
- **Processing.** Two interchangeable modules: `pdftext` (free, one piece per page) and `docai` (Google Document AI Layout Parser, pieces with heading path, tables and scans).
- **Embeddings.** Any OpenAI-compatible endpoint (OpenRouter by default).
- **Storage and search.** PostgreSQL with pgvector (`halfvec`, HNSW) plus full-text search, fused in one query. Optional rerank.
- **Surfaces.** REST under `/v1` and an MCP server at `/mcp` with three tools: `search`, `fetch`, `filters`. Both read the same use cases.
- **Operations.** Sync worker with cron schedule, resumable runs, deletion safety checks, per-operation cost accounting, re-embedding with a generation cutover.

## Architecture

```
source (gdrive) → process (pdftext | docai) → embed → PostgreSQL + pgvector
                                                          │
                                            use cases: search · fetch · filters
                                               ├── REST  /v1/*
                                               └── MCP   /mcp
```

One image, two roles: `python -m src serve` (API + MCP) and `python -m src worker` (sync).

## Configuration

A deployment is `apps/<name>/config.yaml`, merged over `apps/base/config.yaml`. Each section is a pipeline stage (`source`, `process`, `embed`, `search`, `api`, `mcp`, `usage`, `schedule`). Pluggable stages carry a `type` and that module's settings. Secrets are never in the file: `*_secret` settings hold `env:NAME` references. See [apps/README.md](apps/README.md) to create a deployment.

## Run locally

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/) and Docker or Podman.

```bash
uv sync
cp apps/example/.env.example apps/example/.env   # fill in the secrets
# edit apps/example/config.yaml: Drive folder, Document AI processor, bucket
make db-up                                       # pgvector on :5433
make enumerate APP=example                       # what the source yields, no cost
make dev APP=example                             # API, MCP and worker on :8080
```

Other targets: `make help`. `make postman APP=example` generates a Postman collection from the running application's own schema.

## Tests

```bash
uv run pytest -q
```

1,500+ tests run against a real `pgvector/pgvector:pg16` container (started by the suite). Google, the embedding provider and HTTP calls are doubled at the boundary. The `AC-`, `ASM-`, `P-` and `Q-` identifiers in comments and `tests/test_spec_*.py` are the acceptance criteria, assumptions, principles and open questions the project was specified with; each criterion has one test carrying its tag.

## Layout

```
apps/        deployments (base layer, template, example)
migrations/  numbered schema migrations
src/         config, ingest, search, store, usecases, routes, mcp, worker
  sources/ processing/ embedding/ analytics/
tests/       unit, integration and acceptance tests
tools/       operator scripts behind the make targets
```
