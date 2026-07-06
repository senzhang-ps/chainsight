"""Self-contained Databricks connection check for the
`chainsight-databricks-connection` skill.

Loads credentials from `config/.env`, confirms all three keys are present
(without ever printing the token), opens a Databricks SQL connection, and runs
`select 1`. Exits non-zero on any failure so it is safe to use in checks.

Run from the workspace root with the venv interpreter:

    .\\.venv\\Scripts\\python.exe .\\.agents\\skills\\chainsight-databricks-connection\\scripts\\validate_connection.py
"""
from __future__ import annotations

import os
from pathlib import Path

REQUIRED_KEYS = ("DATABRICKS_HOST", "DATABRICKS_HTTP_PATH", "DATABRICKS_TOKEN")


def _find_env_file() -> Path:
    """Locate config/.env from the CWD or by walking up from this file."""
    candidates = [Path("config/.env").resolve()]
    candidates += [parent / "config" / ".env" for parent in Path(__file__).resolve().parents]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("config/.env not found (run from the workspace root)")


def main() -> int:
    from dotenv import load_dotenv
    from databricks import sql as dbsql

    env_file = _find_env_file()
    load_dotenv(env_file)
    print(f"env file: {env_file}")

    # Confirm presence only — never print the token value itself.
    print("credentials loaded:", {key: bool(os.environ.get(key)) for key in REQUIRED_KEYS})
    missing = [key for key in REQUIRED_KEYS if not os.environ.get(key)]
    if missing:
        print(f"ERROR: missing keys in {env_file}: {', '.join(missing)}")
        return 1

    try:
        with dbsql.connect(
            server_hostname=os.environ["DATABRICKS_HOST"],
            http_path=os.environ["DATABRICKS_HTTP_PATH"],
            access_token=os.environ["DATABRICKS_TOKEN"],
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("select 1 as ok")
                print("DBX OK:", cur.fetchall())
    except Exception as exc:  # noqa: BLE001 - surface the real failure cause
        print(f"ERROR: Databricks connection failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
