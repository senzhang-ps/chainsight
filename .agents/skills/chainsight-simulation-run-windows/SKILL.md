---
name: chainsight-simulation-run-windows
description: "Use to execute a ChainSight what-if simulation scenario when the agent runs natively on Windows (PowerShell), not WSL. This is the all-Windows counterpart of chainsight-simulation-run (which bridges from WSL). The workspace, the simulation codebase, the .venv interpreter, and the database are all on the same Windows host, so there is no path translation and no cross-shell hop."
---
# Role

You are a ChainSight Simulation Runner for a **native Windows** environment. The
agent, the simulation codebase, the `.venv` interpreter, and the PostgreSQL
database are all on the same Windows host — the VS Code workspace root *is* the
codebase. Your job is to take the validated scenario config folder from
`workspace/<project>/scenarios/<scenario>/config/`, copy it into the codebase
`input/` tree in the runner-required layout, execute the run, and verify that
the run was actually triggered by checking process and output artifacts.

For the WSL-to-Windows bridge variant (agent in WSL, codebase on Windows), use
`chainsight-simulation-run` instead. For DB access, see
`chainsight-db-connection-windows`.

# When to use

Use this skill when the terminal is a native **Windows PowerShell** session, the
scenario design is defined, the configuration artifact is prepared and
validated, and the simulation is ready to run.

If the terminal is WSL, use `chainsight-simulation-run` (the bridge) instead.

# Required Inputs

- confirmed `project`
- confirmed `scenario`
- confirmed config folder path under
  `workspace/<project>/scenarios/<scenario>/config/`
- confirmed start date and end date from
  `workspace/<project>/scenarios/<scenario>/design.md`
- confirmed active workbook name when the config folder contains more
  than one `.xlsx`
- confirmed run mode:
  - normal run
  - rerun
  - force restart

The codebase path is the workspace root itself; do not ask for a separate
Windows codebase path. Run every command below from the workspace root so the
relative `workspace/...` and `input/...` paths resolve.

# Branch Rules

- show available folders under `workspace/` and
  ask the user to choose one.
- show available folders under
  `workspace/<project>/scenarios/` and ask the user to choose one.
- If the config artifact is already present and validated, reuse it. Do
  not regenerate config in this skill.
- Copy the whole
  `workspace/<project>/scenarios/<scenario>/config/` folder into the
  codebase `input/` tree, preserving the `scenarios/` layer, not just a
  single workbook.
- Default runner-side target path (inside the codebase):
  `input/<project>/scenarios/<scenario>/config/`
- Do not copy scenario configs into the codebase `config/` tree; that
  tree holds only `defaults.yaml`. Scenario inputs belong under `input/`.
  Wrong placement derives the wrong `config_name` and produces a garbage
  `outputs/...` path.
- After copying, scan the runner-side config folder for `.xlsx` files.
  The runner requires exactly one active workbook in that folder.
- The active workbook filename stem becomes `config_name`: it is the DB
  config key and the `db_<config_name>_<ts>` run-id prefix. The output
  folder is named from `<project>/<scenario>` (the `scenarios/` layer and
  the `input/` prefix are dropped), not from the workbook.
- If more than one `.xlsx` exists, list them and ask the user which one
  to keep. Do not run until only one active workbook remains.
- Do not guess the active workbook.
- Run mode semantics:
  - normal run / rerun: without `--force-restart`, the runner
    auto-detects an unfinished (running or failed) run for the same
    `config_name` and date range and resumes it, reusing the existing
    run-id.
  - force restart: `--force-restart` ignores resume and starts a fresh
    run-id.
- A per-`config_name` run lock blocks two concurrent runs of the same
  config. For a rerun or force restart, check for running Windows Python
  simulation processes and confirm whether to stop them first.

# Gates

- Do not run until `project` and `scenario` are confirmed.
- Do not run until `design.md` exists and provides one unambiguous
  period with start date and end date.
- Do not run until the config artifact exists under the scenario
  `config/` folder.
- Do not proceed if the runner-side target path is unclear.
- Do not proceed if the runner-side config folder contains multiple
  `.xlsx` files.
- Do not proceed with force restart unless the user explicitly confirms
  that behavior.

# Workflow

1. Resolve `project` and `scenario` using the branch rules above.
2. Read `workspace/<project>/scenarios/<scenario>/design.md` and extract
   the run period.
3. Confirm the validated config folder under
   `workspace/<project>/scenarios/<scenario>/config/`.
4. Use the workspace virtual environment at `.\.venv\Scripts\python.exe`.
   It already exists (Python 3.13) with all dependencies installed, so
   do not recreate it. Only if `.venv` is missing, create it and install
   `requirements.txt`:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -U pip
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

5. Optionally confirm the interpreter can import the runtime deps:

   ```powershell
   .\.venv\Scripts\python.exe -c "import pandas,numpy,duckdb,openpyxl,yaml,tqdm,psycopg,psutil; print('deps ok')"
   ```

6. Copy the validated scenario config folder
   `workspace/<project>/scenarios/<scenario>/config/` into the codebase
   `input/` tree, preserving the `scenarios/` layer, target
   `input/<project>/scenarios/<scenario>/config/`. This is a local copy
   within the same repo:

   ```powershell
   $dst = "input\<project>\scenarios\<scenario>\config"
   New-Item -ItemType Directory -Force -Path $dst | Out-Null
   Copy-Item -Path "workspace\<project>\scenarios\<scenario>\config\*" -Destination $dst -Recurse -Force
   ```

7. After copying, inspect the runner-side folder and ensure exactly one
   `.xlsx` remains. If not, stop and ask the user which workbook to keep.

   ```powershell
   Get-ChildItem "input\<project>\scenarios\<scenario>\config\*.xlsx" | Select-Object -ExpandProperty Name
   ```

8. If the requested run mode is rerun or force restart, detect existing
   Windows Python simulation processes and ask for confirmation before
   stopping them. The runner also holds a per-`config_name` lock, so a
   second concurrent run of the same config is rejected automatically.

   ```powershell
   Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object CommandLine -like '*run.py*' | Select-Object ProcessId, CommandLine
   ```

9. Trigger the simulation as a detached process using the workspace venv
   interpreter and the `--config-dir` short form. `Start-Process` returns
   immediately, so the agent does not block on the run.

   Default: launch **headless with stdout/stderr redirected to files**.
   This is the only detached form that is immune to the pipe-buffer hang
   described below, and it still captures the console output for the
   agent to read. The run also writes its own authoritative log to the
   output folder.

   ```powershell
   $ts = Get-Date -Format 'yyyyMMdd_HHmmss'
   $outDir = "outputs\<project>\<scenario>"
   New-Item -ItemType Directory -Force -Path $outDir | Out-Null
   Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList 'run.py','--config-dir','input/<project>/scenarios/<scenario>','--start-date','<YYYY-MM-DD>','--end-date','<YYYY-MM-DD>','--use-db','--non-interactive' -WorkingDirectory $PWD -WindowStyle Hidden -RedirectStandardOutput "$outDir\console_$ts.out.log" -RedirectStandardError "$outDir\console_$ts.err.log"
   ```

   Example `-ArgumentList` for `fem-cs-test`:
   `'run.py','--config-dir','input/fem-cs-test/scenarios/baseline-fem-hpfd-network-0386','--start-date','2026-05-01','--end-date','2026-05-31','--use-db','--non-interactive'`

   > **CRITICAL — do not launch via `cmd /k` (or any visible console
   > window) without redirecting output.** A detached `cmd /k python ...`
   > inherits a pipe/console whose buffer is never drained by a reader.
   > When a heavy step (observed: Module3 net-demand on simulation day 1,
   > ~19k deployment rows with multi-layer DemandPriority recursion)
   > emits a large burst of stdout, the buffer fills, the next write
   > blocks, and the **entire run hangs with 0% CPU and a frozen log** —
   > looking exactly like a deadlock at Module3. The same command run in
   > the foreground (where the terminal continuously drains stdout) does
   > NOT hang. Always either redirect to files (default above) or run in
   > the foreground; never use the unredirected `cmd /k` form.

   Foreground-equivalent alternative (agent runs it directly in an async
   terminal that continuously drains stdout, identical to a manual
   foreground run): run the plain command without `Start-Process`:

   ```powershell
   .\.venv\Scripts\python.exe run.py --config-dir input/<project>/scenarios/<scenario> --start-date <YYYY-MM-DD> --end-date <YYYY-MM-DD> --use-db --non-interactive
   ```

   Notes:
   - `-WorkingDirectory $PWD` is required so the relative `input/...`
     path resolves: it is matched via the CWD fallback because
     `workspace_root` in `defaults.yaml` is `./workspace`, which is
     empty. Alternatively pass an absolute config path
     `"$PWD\input\<project>\scenarios\<scenario>\config"` and drop the
     CWD dependence.
   - `--start-date` is required in DB mode; `--end-date` is always
     required; `--non-interactive` skips the interactive run-selection
     prompt so the trigger does not block on stdin.
   - DB params default from `config/defaults.yaml::database`; pass
     `--db-host/--db-port/--db-name/--db-user/--db-password` only when
     targeting a non-default DB.

10. If the user explicitly requests force restart, add `'--force-restart'`
    to the `-ArgumentList` to start a fresh run-id instead of resuming an
    unfinished run.

11. Do not wait in the agent for the run to complete; the `Start-Process`
    launch returns immediately.

12. After triggering the run, do a short verification instead of waiting
    for full completion:
    - check that a new `outputs/<project>/<scenario>/db_run_*` folder
      appears and a simulation log file is being written, or
    - check that a Windows `python.exe` running `run.py` exists (reuse
      the `Get-CimInstance` command from step 8).
    If no new `db_run_*` folder appears, the config path is likely wrong
    (a bad `--config-dir` raises `[ConfigError]` and exits before any
    run folder is created). With the default headless redirect form, read
    the captured `outputs/<project>/<scenario>/console_<ts>.err.log` to
    see the error.

13. Read the confirmed run id from the output layout. Do not infer it
    from the folder name: on auto-resume the run-id is reused from an
    earlier run and will not match the new folder timestamp.

    Timing: `db_run_id.txt` is not written at launch. The runner first
    initializes the database and syncs the config, then writes the file,
    so allow roughly 30 seconds after the trigger before reading. Do not
    read immediately. Poll for the file (e.g. check every few seconds up
    to ~60s) and only read it once it exists and is non-empty; the
    `db_run_*` folder itself appears earlier than the file.

    Output layout (DB mode):

    - run folder: `outputs/<project>/<scenario>/db_run_YYYYMMDD_HHMMSS/`
    - authoritative run-id file: `db_run_id.txt` (full run-id, e.g.
      `db_<config_name>_<ts>`, no trailing newline)

    Resolve and read the latest valid run id:

    ```powershell
    $runDir = Get-ChildItem "outputs\<project>\<scenario>" -Directory -Filter 'db_run_*' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $idFile = Join-Path $runDir.FullName 'db_run_id.txt'
    if ((Test-Path $idFile) -and (Get-Item $idFile).Length -gt 0) { Get-Content -Raw $idFile }
    ```

    Required behavior:
    - find the latest matching `db_run_*` folder for the scenario
    - wait until its `db_run_id.txt` exists and is non-empty, then read
      it and use its text content as the confirmed run id; this is the
      filter key for all DB result tables
    - if multiple recent run folders exist, prefer the newest valid
      `db_run_id.txt`

14. Record the confirmed run id and metadata in
    `workspace/<project>/scenarios/<scenario>/results/run_id.md`.
    Record at minimum:
    - project
    - scenario
    - workspace config folder
    - runner config folder
    - active workbook
    - command used
    - whether `--use-db`
    - whether `--force-restart`
    - output run folder
    - `db_run_id.txt` path
    - confirmed run id
    - trigger time

# Output

- a completed run, or an explicit handoff that states exactly what is
  still missing
- `design.md` updated with the config folder path, active workbook,
  command used, run dates, output location, and run-id retrieval method
- `results/run_id.md` updated with the confirmed latest run id read from
  `db_run_id.txt`
