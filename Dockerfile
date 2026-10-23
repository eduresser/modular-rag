# One image, two commands (T-184, AC-176): the surface and the worker differ
# by which role a container is started with, never by what is built.
#
# Pinned to the `bookworm` release rather than the floating `slim` tag
# (T-204): `src/processing/render.py` renders an uploaded word-processor file
# to PDF with the office suite installed below, and the suite decides where
# the pages break — a version that moved on its own would leave the corpus
# citing pages that no longer exist. Locking the Debian release is what keeps
# that suite's version fixed across a rebuild instead of tracking whatever
# the floating tag resolves to on the day of the build.
FROM python:3.12-slim-bookworm

# uv resolves and installs from the lock file exactly; the project itself is
# never installed as a package (`tool.uv.package = false` in pyproject.toml),
# so this only ever installs its dependencies.
COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /bin/uv

# The office suite `src/processing/render.py` renders an uploaded
# word-processor file to PDF with (AC-210, ASM-118): the source can export a
# document authored in its own editor to PDF itself, but an uploaded file
# carries no export links at all, so it is rendered here instead.
# `--no-install-recommends` keeps out the parts of the suite nothing here
# calls (the GUI shell, the spellcheckers, the clipart).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libreoffice-writer \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# `apps/` is copied in, not mounted (ASM-083): a deployment's committed file
# rides in the same image as the code that reads it, so one artifact and one
# rollback cover both together. Client `.env` files are excluded by
# .dockerignore — they are untracked by git already, and this is the second
# lock on the same door.
COPY src ./src
COPY apps ./apps
COPY migrations ./migrations

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 1000 runner
USER runner

# The surface listens on $PORT (the platform's own convention); the worker
# opens no port. Documentation only — the platform decides what it maps.
EXPOSE 8080

# `python -m src <role>` is the one command either role starts with
# (src/__main__.py). The role is the container's own argument — `serve` or
# `worker` — never baked into the image; a role this build does not define
# refuses immediately rather than starting something arbitrary (AC-176).
ENTRYPOINT ["python", "-m", "src"]
