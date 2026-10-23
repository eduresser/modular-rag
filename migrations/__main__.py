from __future__ import annotations

import logging
import sys

from src.boot import check_boot_compatibility
from src.config import ConfigError, load_config, resolve_database_config
from src.db import connect, create_schema, stored_dimension
from src.secrets import SecretResolutionError


def main(*, app: str | None = None, path: str | None = None) -> int:
    try:
        config = resolve_database_config(load_config(path, app=app))
        with connect(config.db.url) as conn:
            check_boot_compatibility(config, stored_dimension=stored_dimension(conn))
            create_schema(conn, dimension=config.embed.dimension)
        print("Database schema ready.")
        return 0
    except (ConfigError, SecretResolutionError) as exc:
        print(str(exc), file=sys.stderr)
    except Exception as exc:
        logging.getLogger(__name__).error("schema migration failed: error_type=%s", type(exc).__name__)
        print("Database migration failed; check the database and deployment configuration.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
