---
name: chainsight-db-connection
description: WSL-only. Use to connect to the ChainSight simulation-result PostgreSQL database and run queries when the agent runs in a WSL terminal. The DB is hosted on Windows and is not reachable from WSL on port 5432, so this skill defines the verified WSL-to-Windows-Python execution route, the credential format, and a cheap connection-validation sequence. If the agent runs natively on Windows PowerShell, use chainsight-db-connection-windows instead. Other skills (e.g. chainsight-result-analysis) reference this skill for all DB access.
---

# Role

You provide the single, verified way to reach the ChainSight simulation-result
database. ChainSight stores simulation config and output tables in **PostgreSQL
on the Windows host**, accessed via `psycopg`. The agent runs in WSL, where
`psycopg` is typically not installed and `localhost:5432` is refused — so DB
access goes through **Windows Python**.

Any skill that needs to read simulation results (config `cfg_*` tables or
`*_output_*` tables) should connect through this skill **when running in WSL**.
If the agent runs natively on Windows PowerShell, use the all-Windows
counterpart `chainsight-db-connection-windows`, which connects directly without
the WSL bridge. For table structure and KPI calculation logic, see
`chainsight-result-analysis`.

# When to Use

- The agent/terminal is a **WSL** session (not native Windows PowerShell).
- A skill or task needs to query the ChainSight result PostgreSQL DB.
- WSL Python cannot import `psycopg` or cannot reach `localhost:5432`.
- You need to validate that the DB connection works before running a real
  analysis script.

If the agent runs in a native Windows PowerShell terminal, use
`chainsight-db-connection-windows` instead — it connects directly at
`localhost:5432` without the WSL bridge.

# Credentials

Connection info lives in a local-only, git-ignored file at the project root.
Never commit it and never echo the password.

```bash
# ./db_credentials.yaml (project root, git-ignored, never commit)
# Copy from template and fill in actual values.
host: localhost
port: 5432
database: <project_db_name>
user: <your_user>
password: <your_password>
```

Prefer a direct `psycopg.connect(...)` call over building a SQLAlchemy-style URI
first — that is the most reliable pattern in this workspace for ad-hoc queries:

```python
import psycopg
import yaml
from pathlib import Path

creds = yaml.safe_load(Path("db_credentials.yaml").read_text())

with psycopg.connect(
    host=creds["host"],
    port=int(creds["port"]),
    dbname=creds["database"],
    user=creds["user"],
    password=creds["password"],
    connect_timeout=5,
) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT 1")
        print(cur.fetchone())
```

If `PyYAML` is unavailable in a quick throwaway snippet, replace only the
credential-loading step with a line-by-line parse and keep the same
`psycopg.connect(...)` call:

```python
from pathlib import Path

creds = {}
for line in Path("db_credentials.yaml").read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    key, value = line.split(":", 1)
    creds[key.strip()] = value.strip()
```

Use `pandas` only after the connection is established and only when a DataFrame
is actually useful. For most result-analysis tasks, running SQL through
`psycopg` and aggregating in the database is preferred.

# Verified Windows Execution Pattern

When the workspace is open in WSL but the ChainSight PostgreSQL instance is only
reachable from the Windows side, do not keep retrying the same query from WSL
Python. Use this route instead:

1. Keep the analysis script and workspace artifacts in the repo as usual.
2. Convert the WSL path to a Windows path with `wslpath -w`.
3. Launch Windows Python from WSL, preferably through PowerShell or `py -3`.
4. Let Windows Python read `db_credentials.yaml` and connect with `psycopg`
   directly.
5. Write output files back to the workspace through the `\\wsl.localhost\...`
   path or another path returned by `wslpath -w`.

Example command pattern:

```bash
script_win=$(wslpath -w "$PWD/workspace/<project>/scripts/<analysis_script>.py")
powershell.exe -NoProfile -Command "& py -3 \"$script_win\""
```

Use this route when:
- WSL Python can import `psycopg` but connection to `localhost:5432` is refused.
- Windows Python can connect successfully with the same `db_credentials.yaml`.
- The database is clearly running on the Windows host rather than inside WSL.

# Cheap Validation Sequence

Run this first to confirm the connection works before running a real script:

```bash
cred_win=$(wslpath -w "$PWD/db_credentials.yaml")
powershell.exe -NoProfile -Command "& py -3 -c \"from pathlib import Path; import psycopg; creds={};
for line in Path(r'$cred_win').read_text().splitlines():
    line=line.strip()
    if not line or line.startswith('#'):
        continue
    k,v=line.split(':',1)
    creds[k.strip()]=v.strip()
with psycopg.connect(host=creds['host'], port=int(creds['port']), dbname=creds['database'], user=creds['user'], password=creds['password'], connect_timeout=5) as conn:
    with conn.cursor() as cur:
        cur.execute('select 1')
        print(cur.fetchone())\""
```

Expected success output is `(1,)`. Once that passes, run the real analysis
script from Windows Python rather than duplicating the query logic in WSL.

# Notes

- Distinguish scenarios with the `run_id` column on output tables and the
  `config_name` column on config tables.
- Aggregate and filter inside the database; avoid pulling full tables into
  memory. Use `LIMIT` while exploring.
- For the full config/output table catalog and KPI logic, see
  `chainsight-result-analysis`.
