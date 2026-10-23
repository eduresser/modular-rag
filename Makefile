.DEFAULT_GOAL := help
.PHONY: help guard-app enumerate fetch ingest embed cutover test db-up db-down \
        dev postman docker-build docker-serve docker-worker

# ── Target deployment: everything operates on ONE client ──────────────────────
#
# `APP` names it, and the paths follow — `apps/<app>/config.yaml` is the
# configuration and `apps/<app>/.env` holds the secret values that file points
# at. Naming the app names the file, which removes the way to point at the
# wrong one.
#
# No default, and that is a deliberate divergence from the house's other
# per-client product: there `base` is a working development client, here
# `apps/base/config.yaml` is the layer clients overlay and would not validate
# on its own. A missing APP is an error naming it, never a deployment nobody
# asked for.
APP ?=
APP_DIR := apps/$(APP)
APP_ENV := $(APP_DIR)/.env
APP_CONFIG := $(APP_DIR)/config.yaml

# Extra flags for a tool: make enumerate APP=example ARGS="--limit 3"
ARGS ?=

# Superseded, and refused rather than ignored — a path left on a command line
# would otherwise quietly beat the app that was asked for.
ifeq ($(origin RAG_CONFIG),command line)
$(error RAG_CONFIG= was replaced by APP= (use: make $(MAKECMDGOALS) APP=<app>))
endif

# The app's `.env` carries secret VALUES and nothing else; its `config.yaml`
# names them. Sourced in a subshell per target, so no target leaks it to the
# next one.
ENV_PREAMBLE = set -a; [ -f $(APP_ENV) ] && . $(APP_ENV); set +a; APP=$(APP)

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "  every target that touches a deployment needs APP=<app>:"
	@ls -1 apps 2>/dev/null | grep -v '^_' | sed 's/^/    - /'

guard-app:
	@test -n "$(APP)" || { \
	  echo "usage: make $(firstword $(MAKECMDGOALS)) APP=<app>"; \
	  echo "deployments in this build:"; \
	  ls -1 apps 2>/dev/null | grep -v '^_' | sed 's/^/  - /'; \
	  exit 1; }
	@test -f $(APP_CONFIG) || { \
	  echo "ERROR: $(APP_CONFIG) not found (cp $(APP_DIR)/config.yaml.example $(APP_CONFIG))"; \
	  exit 1; }

# ── The source, and the pipeline over it ──────────────────────────────────────

enumerate: guard-app ## What the source yields, as a tree — no download, no cost (APP=x)
	@bash -c '$(ENV_PREAMBLE) uv run python tools/run_ingest.py enumerate $(ARGS)'

fetch: guard-app ## The same plus the bytes, screened against the reader's limits (APP=x)
	@bash -c '$(ENV_PREAMBLE) uv run python tools/run_ingest.py fetch $(ARGS)'

ingest: guard-app ## The whole pipeline — PAYS the reader per page (APP=x ARGS="--yes --limit 3")
	@bash -c '$(ENV_PREAMBLE) uv run python tools/run_ingest.py ingest $(ARGS)'

# ── The embedding stage, part by part ─────────────────────────────────────────

MODE ?= request

embed: guard-app ## Probe the embedding stage (APP=x MODE=request|call|providers|roles|batches|neighbours)
	@bash -c '$(ENV_PREAMBLE) uv run python tools/embed_probe.py $(MODE) $(ARGS)'

# ── Which generation of vectors the archive serves ────────────────────────────
#
# Re-embedding writes a generation; this moves the pointer at it (AC-055). Two
# steps and not one because a partially failed run is a normal outcome, and
# fusing them would serve a generation missing documents.

cutover: guard-app ## What generations exist and which is served (APP=x ARGS="--to <model>")
	@bash -c '$(ENV_PREAMBLE) uv run python tools/cutover.py $(ARGS)'

# ── The suite ─────────────────────────────────────────────────────────────────

test: ## Run the whole suite (needs a running Docker daemon)
	uv run pytest --tap-stream -q

# ── Local infrastructure ──────────────────────────────────────────────────────

PG_CONTAINER ?= rag-pg
PG_PORT ?= 5433

db-up: ## Start a local PostgreSQL with pgvector (PG_PORT=5433)
	@docker inspect $(PG_CONTAINER) >/dev/null 2>&1 \
	  && docker start $(PG_CONTAINER) >/dev/null \
	  || docker run -d --name $(PG_CONTAINER) \
	       -e POSTGRES_HOST_AUTH_METHOD=trust \
	       -p 127.0.0.1:$(PG_PORT):5432 \
	       pgvector/pgvector:pg16 >/dev/null
	@echo "postgres on 127.0.0.1:$(PG_PORT) — db.url: postgresql://postgres@127.0.0.1:$(PG_PORT)/postgres"

db-down: ## Stop it, keeping the data
	@docker stop $(PG_CONTAINER) >/dev/null && echo "stopped $(PG_CONTAINER)"

# ── The service, running here (T-184's two commands, without the container) ──
#
# `make dev APP=<app>` is the whole service against the local archive: the
# surface role — the conventional routes and the mounted tool transport — and
# the worker role beside it, two processes out of one command.
#
# **Both, because the surface on its own accepts work it will never do.** `POST
# /v1/sync` writes a durable request and answers immediately; the worker is what
# asks for one every five seconds (`src/worker.py:506`) and spends the provider.
# Started alone, the surface leaves a request sitting in `sync_requests` as
# `queued` with nothing to move it — which is indistinguishable from a sync that
# is merely slow, and is not what the platform does, where the two roles are two
# always-on services (docs/deploy.md:31).
#
# They run THE SAME commands the image runs — `python -m src serve` and `python
# -m src worker` — and that is the whole of the design. A dev target that started
# uvicorn against an import string of its own would boot an application nothing
# else in this repository boots: the announcement of which deployment this is
# (AC-176), the compatibility gate, and the one line that runs the mounted
# sub-application's lifespan (ASM-063) are exactly the parts that would then be
# exercised for the first time in a deploy.
#
# `db-up` runs first because the archive it serves is the local one and
# starting it is idempotent. A deployment whose `db.url` names something else
# gets a container it never talks to — harmless, and the alternative is this
# target parsing YAML to find out.
#
# **The worker is what applies the schema**, at its own boot — so this target
# does now, where before it applied none and `make ingest APP=<app>` was what
# filled an empty archive. It also honours the deployment's own schedule: left
# running overnight, `make dev APP=example` fires that client's 03:00 sweep and pays
# that client's bill. Running the real role is the point of this target, and it
# is the reason to stop it rather than leave it.
#
# The port is a knob because 8080 is a popular port on a developer's machine,
# and it is CHECKED BEFORE the roles start because of the order the failure
# arrives in otherwise: uvicorn binds last, so the process announces its
# deployment, runs the compatibility gate, starts the tool transport, logs
# `Application startup complete` — and only then says the address is in use.
# Read top to bottom that looks like a service that came up and broke. So the
# check names what is holding the port instead, before any of it.
#
# `ss` missing degrades to no check rather than to a refusal: the bind itself
# is still the real guard, and a machine without iproute2 should not be unable
# to run the service.
#
# **Either role exiting stops the other**, so a crashed surface never leaves a
# worker indexing behind a terminal that looks idle. The message says which one
# went, because the worker leaving is not always a fault: it stands down and
# exits 0 when another worker — or a test run — already owns synchronisation
# (`src/worker.py:550`). Ctrl-C reaches both, and the worker then finishes the
# unit of work in flight — a page, a batch — before it goes (AC-181), so in the
# middle of a sweep the prompt takes a moment to come back.

PORT ?= 8080

dev: guard-app db-up ## The API, the tool surface and the worker, here (APP=x PORT=8080)
	@ss -ltn "sport = :$(PORT)" 2>/dev/null | grep -q LISTEN && { \
	  echo "port $(PORT) is already listening — the surface role would start, pass"; \
	  echo "its checks, log a clean startup and only then fail to bind:"; \
	  ss -ltnp "sport = :$(PORT)" 2>/dev/null | tail -n +2 | sed 's/^/  /'; \
	  echo "free it, or: make dev APP=$(APP) PORT=<other>"; \
	  exit 1; } || true
	@bash -c '$(ENV_PREAMBLE) PORT=$(PORT) uv run python -m src serve & serve=$$!; \
	  $(ENV_PREAMBLE) uv run python -m src worker & worker=$$!; \
	  trap "kill $$serve $$worker 2>/dev/null" INT TERM; \
	  echo "serve pid $$serve, worker pid $$worker — Ctrl-C stops both"; \
	  wait -n; \
	  kill -0 $$serve 2>/dev/null || echo "the surface role exited; stopping the worker"; \
	  kill -0 $$worker 2>/dev/null || echo "the worker role exited; stopping the surface"; \
	  kill $$serve $$worker 2>/dev/null; \
	  wait'

# ── The collection somebody clicks through ───────────────────────────────────
#
# Derived from the application, never typed: the routes come out of FastAPI's
# own OpenAPI document and the tool arguments out of each tool's input schema,
# so a route or an argument that changes changes the collection on the next
# run. `build/` is not committed — the generator is what ships, which is the
# lesson learned from committing a collection that had neither a generator
# nor a target behind it.
#
# Reads NO secret: it builds the application to ask it what it serves, and the
# credential reaches the collection as a blank secret in the environment file
# for whoever imports it to fill in. `PORT` only because that is what `baseUrl`
# has to name — the same knob `make dev` binds.

postman: guard-app ## The Postman collection + environment, into build/ (APP=x PORT=8080)
	@APP=$(APP) uv run python scripts/postman_collection.py --port $(PORT)

# ── The one image, and the two commands it starts with (T-184) ───────────────
#
# `apps/` is baked into the image (ASM-083), so only the secret VALUES —
# never the deployment's file — cross in from the environment. Locally that
# is the app's own `.env`; on the platform it is variables set there directly
# (see docs/deploy.md).

IMAGE ?= modular-rag

docker-build: ## Build the one image both roles run from
	docker build -t $(IMAGE) .

docker-serve: guard-app ## Run the surface role in a container (APP=x)
	docker run --rm -p 8080:8080 --env-file $(APP_ENV) -e APP=$(APP) $(IMAGE) serve

docker-worker: guard-app ## Run the worker role in a container (APP=x)
	docker run --rm --env-file $(APP_ENV) -e APP=$(APP) $(IMAGE) worker
