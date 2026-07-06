---
name: chainsight-databricks-connection
description: 'Use to open a connection to the P&G Databricks SQL warehouse and run read-only SQL from the ChainSight workspace on Windows. Covers credential loading from config/.env (DATABRICKS_HOST / DATABRICKS_HTTP_PATH / DATABRICKS_TOKEN), the databricks-sql-connector + python-dotenv dependencies, the verified dbsql.connect pattern, and a bundled connection-validation script. Triggers: Databricks, databricks-sql-connector, dbsql.connect, SQL warehouse, Unity Catalog, access token, server_hostname, http_path, connect to Databricks, config/.env credentials. This is the Databricks counterpart of chainsight-db-connection-windows (which reaches the PostgreSQL simulation-result DB on localhost:5432); use that skill for cfg_* / *_output_* result tables.'
---

# Role

You provide the single, verified way to **open a connection to the P&G
Databricks SQL warehouse** from the ChainSight workspace **when the agent runs
natively on Windows (PowerShell)**. You connect with the
`databricks-sql-connector` (`from databricks import sql`) using credentials read
from `config/.env`, then run read-only SQL.

This is the Databricks counterpart of `chainsight-db-connection-windows`:

- **Databricks (this skill)** → Unity Catalog tables in the cloud SQL warehouse.A
- **PostgreSQL (`chainsight-db-connection-windows`)** → simulation config
  (`cfg_*`) and output (`*_output_*`) tables on `localhost:5432`.

Any skill or script that needs to read from Databricks should open its
connection through this skill.

# When to Use

- You need to run a read-only SQL query against the Databricks SQL warehouse.
- A script or skill needs upstream data that lives in Databricks Unity Catalog.
- You want to validate Databricks connectivity before running a real query.

Do **not** use this for the simulation-result PostgreSQL DB (`cfg_*` /
`*_output_*`) — use `chainsight-db-connection-windows` for that.

# Dependencies

The connection needs two packages:

- `databricks-sql-connector` — provides `from databricks import sql`.
- `python-dotenv` — loads `config/.env`.

Both are **already installed in the workspace `.venv`** but are **not** pinned in
`requirements.txt` (they are script-only). Verify before assuming:

```powershell
.\.venv\Scripts\python.exe -c "import importlib.util as u; print('databricks.sql:', bool(u.find_spec('databricks.sql'))); print('dotenv:', bool(u.find_spec('dotenv')))"
```

If a package is missing, install it into the venv:

```powershell
.\.venv\Scripts\python.exe -m pip install databricks-sql-connector python-dotenv
```

Notes:

- **Optional `pyarrow`**: with `databricks-sql-connector` 4.x you will see a
  one-line `[WARN] pyarrow is not installed` message. It is harmless — Arrow /
  cloud-fetch APIs are simply disabled. Install `pyarrow` (or
  `databricks-sql-connector[pyarrow]`) only if you need those.
- **Proxy caveat (P&G network)**: a fresh `pip install` can fail (exit code 1)
  behind the corporate proxy / SSL inspection. The commented `HTTPS_PROXY` /
  `REQUESTS_CA_BUNDLE` block at the bottom of `config/.env` shows the values to
  set. Because the deps are already in `.venv`, you normally do **not** need to
  install anything.

# Credentials

Connection settings live in **`config/.env`** (git-ignored — never commit it).
Three keys are required:

```ini
# config/.env
DATABRICKS_HOST=<adb-...databricks.azure.cn>          # Azure China workspace host
DATABRICKS_HTTP_PATH=/sql/protocolv1/o/<org>/<warehouse-id>
DATABRICKS_TOKEN=<dapi... personal access token>
```

Load them with `python-dotenv`, then read from the environment:

```python
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path("config/.env"))
host = os.environ["DATABRICKS_HOST"]
http_path = os.environ["DATABRICKS_HTTP_PATH"]
token = os.environ["DATABRICKS_TOKEN"]
```

Security rules:

- **Never print or log the token.** Print only booleans to confirm loading
  (`host loaded: True | http_path loaded: True | token loaded: True`).
- The token is a `dapi…` **personal access token** — treat it as a secret. Do
  not paste it into code, commit it, or echo it in terminal output.
- A missing key raises `KeyError`. Fill it in from the template comments at the
  top of `config/.env`.

# Verified Connection Pattern

Run from the **workspace root** with the venv interpreter so `config/.env`
resolves. Use context managers for both the connection and the cursor.

```python
import os
from pathlib import Path
from dotenv import load_dotenv
from databricks import sql as dbsql

load_dotenv(Path("config/.env"))

with dbsql.connect(
    server_hostname=os.environ["DATABRICKS_HOST"],
    http_path=os.environ["DATABRICKS_HTTP_PATH"],
    access_token=os.environ["DATABRICKS_TOKEN"],
) as conn:
    with conn.cursor() as cur:
        cur.execute("select 1 as ok")        # replace with your query
        for row in cur.fetchall():
            print(row)
```

Execute it from the workspace root:

```powershell
.\.venv\Scripts\python.exe .\path\to\your_script.py
```

Build a `pandas` DataFrame only after `fetchall()`, and only when a DataFrame is
actually useful.

# Cheap Validation Sequence

Run the **bundled validation script** first, from the workspace root with the
venv interpreter. It loads `config/.env`, confirms the three keys are present
(without printing the token), and runs `select 1`
([scripts/validate_connection.py](.agents/skills/chainsight-databricks-connection/scripts/validate_connection.py)):

```powershell
.\.venv\Scripts\python.exe .\.agents\skills\chainsight-databricks-connection\scripts\validate_connection.py
```

Expected output:

```text
credentials loaded: {'DATABRICKS_HOST': True, 'DATABRICKS_HTTP_PATH': True, 'DATABRICKS_TOKEN': True}
DBX OK: [Row(ok=1)]
```

One-liner alternative (no script file):

```powershell
.\.venv\Scripts\python.exe -c "import os; from pathlib import Path; from dotenv import load_dotenv; from databricks import sql as d; load_dotenv(Path('config/.env')); c=d.connect(server_hostname=os.environ['DATABRICKS_HOST'], http_path=os.environ['DATABRICKS_HTTP_PATH'], access_token=os.environ['DATABRICKS_TOKEN']); cur=c.cursor(); cur.execute('select 1'); print('DBX OK:', cur.fetchall()); c.close()"
```

Expected success output is `DBX OK: [(1,)]` (or `[Row(...)]`). Once that passes,
run the real script with the same venv interpreter.

# Query Conventions & Notes

- **Read-only.** This skill is for `SELECT` analytics. Do not run DML/DDL against
  Databricks catalogs.
- **Fully-qualify tables** with the 3-level Unity Catalog name
  `catalog.schema.table`.
- **Aggregate and filter in the warehouse**; use `LIMIT` while exploring. Avoid
  pulling whole tables into memory.
- **Chunk large `IN (...)` lists** (≤ ~500 values) to stay within query limits.
- **PowerShell**: chain commands with `;` (not `&&`), and always run from the
  workspace root so `config/.env` and relative paths resolve.
```