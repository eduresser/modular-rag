from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.processing.preflight import UNPAGINATED_MEDIA_TYPES

MEDIA_PDF = "application/pdf"

# A hang here is a stuck subprocess; bound it so one bad file cannot stall the run
RENDER_TIMEOUT_SECONDS = 120

_EXTENSIONS: dict[str, str] = {media_type: "docx" for media_type in UNPAGINATED_MEDIA_TYPES}


class RenderError(Exception):
    ...


@dataclass(frozen=True, slots=True)
class RenderedDocument:

    media_type: str
    content: bytes


Renderer = Callable[[str, bytes], bytes]


def docai_rendering_for(media_type: str) -> str:
    return MEDIA_PDF if media_type in UNPAGINATED_MEDIA_TYPES else ""


def render(
    external_id: str,
    media_type: str,
    content: bytes,
    *,
    to: str = "",
    run: Renderer | None = None,
) -> RenderedDocument:
    if not to or media_type == to:
        return RenderedDocument(media_type=media_type, content=content)
    if to != MEDIA_PDF:
        raise RenderError(
            f"document {external_id!r} was asked to be rendered to {to}, and the office "
            f"suite in this image renders to {MEDIA_PDF} and nothing else"
        )
    renderer = run or render_to_pdf
    try:
        pdf_bytes = renderer(media_type, content)
    except RenderError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RenderError(
            f"document {external_id!r} ({media_type}) could not be rendered to a paginated "
            f"format: {exc}"
        ) from exc
    return RenderedDocument(media_type=MEDIA_PDF, content=pdf_bytes)


def render_to_pdf(media_type: str, content: bytes) -> bytes:
    extension = _EXTENSIONS.get(media_type)
    if extension is None:
        raise RenderError(
            f"{media_type!r} is not one of the formats this module renders: "
            f"{', '.join(sorted(_EXTENSIONS))}"
        )
    with tempfile.TemporaryDirectory() as workdir:
        source = Path(workdir) / f"input.{extension}"
        source.write_bytes(content)
        result = subprocess.run(
            [
                "soffice",
                "--headless",
                "--norestore",
                f"-env:UserInstallation=file://{workdir}/profile",
                "--convert-to",
                "pdf",
                "--outdir",
                workdir,
                str(source),
            ],
            capture_output=True,
            timeout=RENDER_TIMEOUT_SECONDS,
        )
        output = source.with_suffix(".pdf")
        if result.returncode != 0 or not output.exists():
            detail = (
                result.stderr.decode(errors="replace").strip()
                or result.stdout.decode(errors="replace").strip()
                or f"exited {result.returncode} and wrote no output"
            )
            raise RenderError(f"the office suite could not render this document: {detail}")
        return output.read_bytes()


def renderer_version() -> str:
    result = subprocess.run(
        ["soffice", "--version"],
        capture_output=True,
        timeout=RENDER_TIMEOUT_SECONDS,
    )
    return result.stdout.decode(errors="replace").strip()
