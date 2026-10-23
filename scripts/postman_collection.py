"""Generate the Postman collection and environment file from the application.

Usage::

    uv run python scripts/postman_collection.py --app example
    uv run python scripts/postman_collection.py --app example --port 8090
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anyio

from src.auth import Credential
from src.config import AppConfig, ConfigError, load_config
from src.main import MCP_PATH, OPEN_PATHS, Service, create_app

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "build" / "postman"

PLACEHOLDER_TOKEN = "generated-collection-holds-no-credential"  # noqa: S105

ROUTE_NAMES: dict[tuple[str, str], str] = {
    ("GET", "/healthz"): "Está no ar? (sem credencial)",
    ("POST", "/v1/search"): "Pergunta ao arquivo — trechos, ou documentos",
    ("POST", "/v1/sync"): "Iniciar a sincronização dos documentos",
    ("GET", "/v1/sync/{sync_id}"): "Acompanhar uma sincronização",
    ("GET", "/v1/documents/{identifier}/outline"): "O sumário do documento, por títulos",
    ("GET", "/v1/documents/{identifier}/status"): "Este documento foi lido? E se não, por quê?",
    ("GET", "/v1/documents/{identifier}"): "O documento inteiro, em ordem e por janela",
    ("GET", "/v1/documents"): "O que o arquivo guarda, página a página",
    ("GET", "/v1/filters"): "O vocabulário de filtro, e os valores que o arquivo guarda",
    ("GET", "/v1/sync"): "O que está rodando agora, e o que rodou antes",
    ("GET", "/v1/usage"): "Quanto isto custou, e em quê",
    ("GET", "/v1/corpus/reach"): "O que foi indexado e nunca apareceu numa resposta",
    ("GET", "/v1/searches/latency"): "Quanto tempo custa uma busca, por esforço de índice",
    ("GET", "/v1/serving"): "Qual modelo está servindo, e ele bate com o configurado?",
}

PATH_VARIABLES: dict[str, str] = {"identifier": "document_id", "sync_id": "sync_id"}

# bound; `""` satisfies `type: string` and is refused by `minLength: 1`, so a
EXAMPLES: dict[str, Any] = {
    "question": "quais são as obrigações do contratante?",
    "query": "quais são as obrigações do contratante?",
    "caller": "postman",
    "id": "{{document_id}}",
    "identifier": "{{document_id}}",
}

def _example(schema: dict[str, Any], components: dict[str, Any], field: str = "") -> Any:
    if field in EXAMPLES:
        return EXAMPLES[field]
    if "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[-1]
        return _example(components.get(name, {}), components)
    if "default" in schema and schema["default"] is not None:
        return schema["default"]
    if "enum" in schema:
        return schema["enum"][0]
    for option in schema.get("anyOf", []):
        if option.get("type") != "null":
            return _example(option, components, field)
    kind = schema.get("type")
    if kind == "object":
        return {
            name: _example(sub, components, name)
            for name, sub in schema.get("properties", {}).items()
            if name in schema.get("required", [])
            or name in EXAMPLES
            or sub.get("default") is not None
        }
    if kind == "array":
        return []
    if kind == "integer":
        return schema.get("minimum", 0)
    if kind == "number":
        return schema.get("minimum", 0)
    if kind == "boolean":
        return False
    return ""


def _bounds(schema: dict[str, Any]) -> str:
    options = schema.get("anyOf", [schema])
    parts: list[str] = []
    for option in options:
        if option.get("type") == "null":
            parts.append("omitido = o padrão do deployment")
            continue
        if "enum" in option:
            parts.append("um de: " + ", ".join(str(value) for value in option["enum"]))
        if "minimum" in option:
            parts.append(f"mínimo {option['minimum']:g}")
        if "maximum" in option:
            parts.append(f"máximo {option['maximum']:g}")
    return "; ".join(parts)


def _url(path: str, query: list[dict[str, Any]]) -> dict[str, Any]:
    segments = [
        f"{{{{{PATH_VARIABLES.get(part[1:-1], part[1:-1])}}}}}"
        if part.startswith("{") and part.endswith("}")
        else part
        for part in path.strip("/").split("/")
        if part
    ]
    return {
        "raw": "/".join(["{{baseUrl}}", *segments]),
        "host": ["{{baseUrl}}"],
        "path": segments,
        "query": query,
    }


def _rest_request(
    method: str, path: str, operation: dict[str, Any], components: dict[str, Any]
) -> dict[str, Any]:
    try:
        name = ROUTE_NAMES[(method, path)]
    except KeyError:
        raise SystemExit(
            f"{method} {path} has no name in ROUTE_NAMES ({Path(__file__).name}).\n"
            "A route reaches the collection under a name a person wrote, or it does not "
            "reach it: FastAPI's own summary names the function, not the question the "
            "route answers."
        ) from None

    query: list[dict[str, Any]] = []
    for parameter in operation.get("parameters", []):
        if parameter.get("in") != "query":
            continue
        schema = parameter.get("schema", {})
        query.append(
            {
                "key": parameter["name"],
                "value": str(_example(schema, components, parameter["name"])),
                "disabled": not parameter.get("required", False),
                "description": _bounds(schema),
            }
        )

    request: dict[str, Any] = {
        "method": method,
        "header": [],
        "url": _url(path, query),
        "description": (operation.get("description") or "").strip(),
    }
    if path in OPEN_PATHS:
        request["auth"] = {"type": "noauth"}

    body_schema = (
        operation.get("requestBody", {})
        .get("content", {})
        .get("application/json", {})
        .get("schema")
    )
    if body_schema is not None:
        request["header"].append({"key": "Content-Type", "value": "application/json"})
        body = _example(body_schema, components)
        request["body"] = {
            "mode": "raw",
            "raw": json.dumps(body, ensure_ascii=False, indent=2),
            "options": {"raw": {"language": "json"}},
        }
        request["description"] = "\n\n".join(
            part for part in (request["description"], _omitted(body_schema, components, body)) if part
        )
    return {"name": name, "request": request}


def _omitted(schema: dict[str, Any], components: dict[str, Any], body: dict[str, Any]) -> str:
    if "$ref" in schema:
        schema = components.get(schema["$ref"].rsplit("/", 1)[-1], {})
    lines = [
        f"- `{name}`" + (f" — {bounds}" if (bounds := _bounds(sub)) else "")
        for name, sub in schema.get("properties", {}).items()
        if name not in body
    ]
    if not lines:
        return ""
    return (
        "Omitidos do corpo acima, cada um com o padrão do deployment (AC-134 põe "
        "todo botão no escopo da requisição; o teto está no config e um valor acima "
        "dele é recusado em 422):\n" + "\n".join(lines)
    )


def _rpc(name: str, description: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "request": {
            "method": "POST",
            "header": [
                {"key": "Content-Type", "value": "application/json"},
                {"key": "Accept", "value": "application/json, text/event-stream"},
            ],
            "url": {
                "raw": "{{baseUrl}}" + MCP_PATH + "/",
                "host": ["{{baseUrl}}"],
                "path": [MCP_PATH.strip("/"), ""],
                "query": [],
            },
            "description": description,
            "body": {
                "mode": "raw",
                "raw": json.dumps(payload, ensure_ascii=False, indent=2),
                "options": {"raw": {"language": "json"}},
            },
        },
    }


def _tool_folder(app: Any) -> dict[str, Any]:
    tools = anyio.run(app.state.mcp.list_tools)
    requests = [
        _rpc(
            "Aperto de mão (uma vez por cliente)",
            "O transporte é stateless (ASM-060): não devolve id de sessão e as chamadas "
            "seguintes não precisam de um. Está aqui porque um cliente MCP de verdade "
            "começa por ele.",
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "postman", "version": "1.0.0"},
                },
            },
        ),
        _rpc(
            "Quais ferramentas o assistente enxerga",
            "Exatamente o que entra no prompt do assistente em toda conversa — motivo "
            "pelo qual uma terceira ferramenta é decisão de produto (AC-130).",
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        ),
    ]
    for index, tool in enumerate(tools, start=3):
        schema = tool.input_schema
        arguments = {
            field: _example(sub, {}, field)
            for field, sub in schema.get("properties", {}).items()
            if field in schema.get("required", [])
            or field in EXAMPLES
            or sub.get("default") is not None
        }
        summary = (tool.description or "").strip().splitlines()[0]
        requests.append(
            _rpc(
                f"Chama `{tool.name}`",
                f"{summary}\n\nArgumentos obrigatórios: "
                f"{', '.join(schema.get('required', [])) or 'nenhum'}.\n"
                "A ferramenta NUNCA levanta exceção: toda falha volta como payload "
                "(AC-132), porque uma exceção chegando ao assistente o para no meio "
                "da tarefa em vez de deixá-lo contar ao advogado o que aconteceu.",
                {
                    "jsonrpc": "2.0",
                    "id": index,
                    "method": "tools/call",
                    "params": {"name": tool.name, "arguments": arguments},
                },
            )
        )
    return {
        "name": "mcp",
        "description": (
            "A superfície de ferramentas, montada em "
            f"{MCP_PATH}/ (com a barra final). Fala JSON-RPC 2.0 sobre HTTP e responde "
            "em Server-Sent Events, então o corpo da resposta chega como `event: message` "
            "seguido de `data: {...}` — o Postman mostra como texto.\n\n"
            "Ela e as rotas /v1 chamam a MESMA função (src/usecases/): um passo "
            "acrescentado a uma superfície e não à outra faria as duas responderem "
            "diferente à mesma pergunta, e ninguém notaria porque ninguém pergunta às duas."
        ),
        "item": requests,
    }


def build(config: AppConfig, app: Any) -> dict[str, Any]:
    schema = app.openapi()
    components = schema.get("components", {}).get("schemas", {})

    folders: dict[str, list[dict[str, Any]]] = {}
    for path, operations in schema["paths"].items():
        for method, operation in operations.items():
            folder = "health" if path in OPEN_PATHS else path.strip("/").split("/")[0]
            folders.setdefault(folder, []).append(
                _rest_request(method.upper(), path, operation, components)
            )

    variables = [{"key": "baseUrl", "value": "{{baseUrl}}", "type": "string"}] + [
        {"key": name, "value": "", "type": "string"} for name in sorted(set(PATH_VARIABLES.values()))
    ]
    return {
        "info": {
            "name": f"modular-rag ({config.telemetry.client})",
            "description": (
                "As duas superfícies do serviço: as rotas convencionais /v1 e a superfície "
                "de ferramentas MCP montada em /mcp/.\n\n"
                "GERADO por `make postman APP=<app>` a partir da própria aplicação — as "
                "rotas saem do OpenAPI do FastAPI e os argumentos das ferramentas saem do "
                "schema de cada uma. Não edite: regenere.\n\n"
                "Preencha `api_token` no environment (é o valor que `api.credentials_secret` "
                "referencia no config do deployment) e cole um id real em `document_id` — "
                "o campo `document` de qualquer resultado de busca é esse id.\n\n"
                f"O host importa: este deployment só responde a "
                f"{', '.join(config.api.allowed_hosts)}. Qualquer outro nome — 127.0.0.1 "
                "inclusive — é recusado em 400 antes de qualquer rota rodar, e a recusa "
                "diz o nome que ele esperava."
            ),
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
        },
        "auth": {
            "type": "bearer",
            "bearer": [{"key": "token", "value": "{{api_token}}", "type": "string"}],
        },
        "variable": variables[1:],
        "item": [
            {"name": name, "item": requests}
            for name, requests in sorted(folders.items(), key=lambda pair: pair[0] != "v1")
        ]
        + [_tool_folder(app)],
    }


def environment(config: AppConfig, port: int) -> dict[str, Any]:
    host = config.api.allowed_hosts[0]
    return {
        "name": f"modular-rag ({config.telemetry.client}, local)",
        "values": [
            {
                "key": "baseUrl",
                "value": f"http://{host}:{port}",
                "type": "default",
                "enabled": True,
            },
            {"key": "api_token", "value": "", "type": "secret", "enabled": True},
        ],
        "_postman_variable_scope": "environment",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", help="which deployment; defaults to $APP")
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="the port `make dev` binds, which is what baseUrl names (default: 8080)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_DIR,
        help=f"where to write both files (default: {OUTPUT_DIR.relative_to(ROOT)})",
    )
    args = parser.parse_args()

    try:
        config = load_config(app=args.app)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    app = create_app(
        Service(
            config=config,
            embedder=None,
            credential=Credential(token=PLACEHOLDER_TOKEN),
            mcp_credential=Credential(token=PLACEHOLDER_TOKEN),
        )
    )

    args.out.mkdir(parents=True, exist_ok=True)
    written = {
        args.out / f"modular-rag-{config.telemetry.client}.postman_collection.json": build(
            config, app
        ),
        args.out / f"modular-rag-{config.telemetry.client}.postman_environment.json": environment(
            config, args.port
        ),
    }
    for destination, document in written.items():
        destination.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
        print(destination.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
