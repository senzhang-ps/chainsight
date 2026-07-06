---
name: chainsight-db-connection-windows
description: Use to connect to the ChainSight simulation-result PostgreSQL database and run queries when the agent itself runs natively on Windows (PowerShell), not WSL. The DB runs on the same Windows host and is reachable directly at localhost:5432, so this skill defines the verified native-Windows execution route (workspace .venv Python), the credential source, and a cheap connection-validation sequence. This is the all-Windows counterpart of chainsight-db-connection (which bridges from WSL). Other skills (e.g. chainsight-result-analysis) can reference either skill for DB access.
---

# Role

You provide the single, verified way to reach the ChainSight simulation-result
database **when the agent runs natively on Windows**. ChainSight stores
simulation config and output tables in **PostgreSQL on the Windows host**,
accessed via `psycopg`. Because the agent, the workspace `.venv`, and the
database are all on the same Windows machine, `localhost:5432` is reachable
directly — no WSL bridge, no path translation.

Any skill that needs to read simulation results (config `cfg_*` tables or
`*_output_*` tables) should connect through this skill when running on Windows.
For the WSL-to-Windows bridge variant, use `chainsight-db-connection` instead.
For table structure and KPI calculation logic, see `chainsight-result-analysis`.

# When to Use

- The agent/terminal is a native **Windows PowerShell** session (not WSL).
- A skill or task needs to query the ChainSight result PostgreSQL DB.
- You need to validate that the DB connection works before running a real
  analysis script.

If the terminal is WSL and `localhost:5432` is refused, use
`chainsight-db-connection` (the WSL bridge) instead of this skill.

# Python Interpreter

Use the workspace virtual environment so `psycopg`, `pandas`, and `PyYAML` are
guaranteed available:

```powershell
.\.venv\Scripts\python.exe --version
```

If `.\.venv` does not exist, fall back to the project interpreter (`python` or
`py -3.12`) but first confirm dependencies:

```powershell
python -c "import psycopg, yaml, pandas; print('deps ok')"
```

# Credentials

On Windows the project already carries the connection settings in the
git-tracked config — there is **no separate `db_credentials.yaml` to create**.
Connection info lives in the `database:` node of `config/defaults.yaml` and is
read by `pgsql_db/settings.py`:

```yaml
# config/defaults.yaml  ->  database: node (read by pgsql_db/settings.py)
database:
  host: localhost
  port: 5432
  database: <project_db_name>
  user: <your_user>
  password: <your_password>     # dev default only; never echo it
  maintenance_database: postgres
```

Never print the password in terminal output or logs. For production, leave
`password: ""` in the file and inject it via an environment variable or an
explicit argument override.

# Verified Windows Execution Pattern

Two equivalent native routes. Prefer the project helper; use the direct
`psycopg.connect(...)` route for throwaway one-liners.

## Route A — Project helper (preferred)

`pgsql_db.db_connection.DatabaseConnection` resolves the `database:` node from
`config/defaults.yaml` automatically (autocommit, retry/backoff, UTF-8 for the
Windows Chinese locale already handled):

```python
from pgsql_db.db_connection import DatabaseConnection

db = DatabaseConnection()        # reads config/defaults.yaml -> database:
conn = db.connect()
with conn.cursor() as cur:
    cur.execute("SELECT 1")
    print(cur.fetchone())
db.close()
```

Run it from the workspace root with the venv interpreter:

```powershell
.\.venv\Scripts\python.exe .\path\to\analysis_script.py
```

## Route B — Direct psycopg (ad-hoc)

Read the `database:` node yourself and connect directly. This is the most
reliable pattern for quick, self-contained queries:

```python
import psycopg
import yaml
from pathlib import Path

cfg = yaml.safe_load(Path("config/defaults.yaml").read_text(encoding="utf-8"))["database"]

with psycopg.connect(
    host=cfg["host"],
    port=int(cfg["port"]),
    dbname=cfg["database"],
    user=cfg["user"],
    password=str(cfg["password"]),
    client_encoding="UTF8",
    connect_timeout=5,
) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT 1")
        print(cur.fetchone())
```

Use `pandas` only after the connection is established and only when a DataFrame
is actually useful. For most result-analysis tasks, run SQL through `psycopg`
and aggregate in the database.

# Cheap Validation Sequence

Run this first, from the workspace root, to confirm the connection works before
running a real script.

Project-helper check:

```powershell
.\.venv\Scripts\python.exe -c "from pgsql_db.db_connection import DatabaseConnection; db=DatabaseConnection(); cur=db.connect().cursor(); cur.execute('select 1'); print(cur.fetchone()); db.close()"
```

Direct-psycopg check (config-driven):

```powershell
.\.venv\Scripts\python.exe -c "import yaml,psycopg; from pathlib import Path; c=yaml.safe_load(Path('config/defaults.yaml').read_text(encoding='utf-8'))['database']; conn=psycopg.connect(host=c['host'],port=int(c['port']),dbname=c['database'],user=c['user'],password=str(c['password']),connect_timeout=5); cur=conn.cursor(); cur.execute('select 1'); print(cur.fetchone()); conn.close()"
```

Expected success output is `(1,)`. Once that passes, run the real analysis
script from the same venv interpreter.

# Notes

- Distinguish scenarios with the `run_id` column on output tables and the
  `config_name` column on config tables.
- Aggregate and filter inside the database; avoid pulling full tables into
  memory. Use `LIMIT` while exploring.
- In PowerShell, chain commands with `;` (not `&&`), and run from the workspace
  root so `config/defaults.yaml` and the `pgsql_db` package resolve.
- For the full config/output table catalog and KPI logic, see
  `chainsight-result-analysis`.
