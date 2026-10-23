# Deployments

Each folder here is one deployment: one corpus, one API and one MCP server, with its own database and credentials. All of them run the same code.

| Folder | Role |
|---|---|
| `base/` | Layer every deployment is read over. Holds shared defaults and omits anything that identifies a corpus. |
| `_template/` | Annotated starting point with the settings every deployment must state. |
| `example/` | Working reference: Google Drive source, Document AI processing, OpenRouter embeddings. |

## Create a deployment

1. `cp -r apps/example apps/<name>`
2. In `apps/<name>/config.yaml`, set `telemetry.client` to `<name>` (must match the folder), then the Drive folder IDs, `db.url` and `api.allowed_hosts`.
3. Choose the processing module:
   - `process.type: docai`: fill in `read.processor_id`, `read.processor_version` and `read.artifact_bucket`. Cost is about US$ 10 per 1,000 pages.
   - `process.type: pdftext`: delete the `process` section. Free, one piece per page, refuses scans.
4. `cp apps/<name>/.env.example apps/<name>/.env` and fill in the secrets.
5. `make dev APP=<name>`.

A deployment overrides only what differs from `base/`. Mapping and list settings are replaced whole, never merged.

## Add a module

A pluggable stage (`source`, `process`) is a closed set. A new module is one settings class inheriting the stage's base, one entry in `SOURCE_MODULES` or `PROCESS_MODULES` in `src/config.py`, and the implementation registered in `src/sources/base.py` or `src/processing/`. Its settings then appear in any deployment under `type: <module>`.
