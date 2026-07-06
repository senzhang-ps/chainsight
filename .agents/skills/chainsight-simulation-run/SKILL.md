---
name: chainsight-simulation-run
description: "WSL-only. Use to execute a ChainSight what-if simulation scenario when the agent runs in a WSL terminal while the simulation codebase runs on Windows; this skill bridges WSL to Windows via wslpath and powershell.exe. If the agent runs natively on Windows PowerShell, use chainsight-simulation-run-windows instead. Use after the design is defined and configuration is prepared."
---
# Role

You are a ChainSight Simulation Runner. The agent runs in WSL, but the
simulation codebase runs in Windows. Your job is to take the validated
scenario config folder from the workspace, copy it into the Windows
simulation codebase in the runner-required layout, execute the run, and
verify that the run was actually triggered by checking process and
output artifacts.

If the agent runs natively on Windows PowerShell (no WSL bridge), use the
all-Windows counterpart `chainsight-simulation-run-windows` instead.

# When to use

Use this skill when the terminal is a **WSL** session, the scenario design
is defined, the configuration artifact is prepared and validated, and the
simulation is ready to run.

If the terminal is native Windows PowerShell (not WSL), use
`chainsight-simulation-run-windows` instead — it runs directly in the
workspace without `wslpath` or `powershell.exe` hops.

# Required Inputs

- confirmed `project`
- confirmed `scenario`
- confirmed config folder path under
  `workspace/<project>/scenarios/<scenario>/config/`
- confirmed Windows simulation codebase path. Default:
  `C:\Users\zhang.s.37\OneDrive - Procter and Gamble\Documents\GitHub\chainsight`
- confirmed start date and end date from
  `workspace/<project>/scenarios/<scenario>/design.md`
- confirmed active workbook name when the config folder contains more
  than one `.xlsx`
- confirmed run mode:
  - normal run
  - rerun
  - force restart

# Branch Rules

- show available folders under `workspace/` and
  ask the user to choose one.
- show available folders under
  `workspace/<project>/scenarios/` and ask the user to choose one.
- If the config artifact is already present and validated, reuse it. Do
  not regenerate config in this skill.
- Copy the whole
  `workspace/<project>/scenarios/<scenario>/config/` folder into the
  Windows codebase `input/` tree, preserving the `scenarios/` layer, not
  just a single workbook.
- Default runner-side target path (inside the Windows codebase):
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
- Do not invent the Windows codebase path, use default or check with user.
  If any of them are missing or ambiguous, stop and ask the user.
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
4. Use the Windows simulation codebase, not the WSL workspace, for the
   actual simulation run. Resolve its Windows path once and reuse it for
   every command below:

   ```bash
   win=$(wslpath -w "<windows codebase path>")
   ```

5. Use the codebase virtual environment at `.venv\Scripts\python.exe`.
   It already exists (Python 3.13) with all dependencies installed, so
   do not recreate it. Only if `.venv` is missing, create it and install
   `requirements.txt`:

   ```bash
   powershell.exe -NoProfile -Command "cd '$win'; py -3.12 -m venv .venv; .\.venv\Scripts\python.exe -m pip install -U pip; .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
   ```

6. Optionally confirm the interpreter can import the runtime deps:

   ```bash
   powershell.exe -NoProfile -Command "cd '$win'; .\.venv\Scripts\python.exe -c 'import pandas,numpy,duckdb,openpyxl,yaml,tqdm,psycopg,psutil; print(\"deps ok\")'"
   ```

7. Copy the validated scenario config folder
   `workspace/<project>/scenarios/<scenario>/config/` into the Windows
   codebase `input/` tree, preserving the `scenarios/` layer, target
   `input/<project>/scenarios/<scenario>/config/`. From WSL, copy through
   the codebase `/mnt/c/...` mount:

   ```bash
   cp -r "workspace/<project>/scenarios/<scenario>/config/." \
     "<codebase /mnt/c path>/input/<project>/scenarios/<scenario>/config/"
   ```
8. After copying, inspect the runner-side folder and ensure exactly one
   `.xlsx` remains. If not, stop and ask the user which workbook to
   keep.
9. If the requested run mode is rerun or force restart, detect existing
   Windows Python simulation processes and ask for confirmation before
   stopping them. The runner also holds a per-`config_name` lock, so a
   second concurrent run of the same config is rejected automatically.

   ```bash
   powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | Where-Object CommandLine -like '*run.py*' | Select-Object ProcessId, CommandLine"
   ```
10. Trigger the simulation detached from WSL through `powershell.exe`,
   using the codebase venv interpreter and the `--config-dir` short
   form. `Start-Process` returns immediately, so the agent does not
   block on the run.

   Default: launch a **visible console window** so the user can watch
   live progress, the same as a manual run. Do not hide the window and
   do not redirect output; the run also writes its own authoritative log
   to the output folder.

   ```bash
   powershell.exe -NoProfile -Command "Start-Process -FilePath '$win\.venv\Scripts\python.exe' -ArgumentList 'run.py','--config-dir','input/<project>/scenarios/<scenario>','--start-date','<YYYY-MM-DD>','--end-date','<YYYY-MM-DD>','--use-db','--non-interactive' -WorkingDirectory '$win'"
   ```

   Example `-ArgumentList` for `fem-cs-test`:
   `'run.py','--config-dir','input/fem-cs-test/scenarios/baseline-fem-hpfd-network-0386','--start-date','2026-05-01','--end-date','2026-05-31','--use-db','--non-interactive'`

   The window closes when the run ends. To keep it open after completion
   (closest to the manual experience), wrap the command in `cmd /k`:

   ```bash
   powershell.exe -NoProfile -Command "Start-Process -FilePath 'cmd.exe' -ArgumentList '/k','cd /d \"$win\" && \".venv\Scripts\python.exe\" run.py --config-dir input/<project>/scenarios/<scenario> --start-date <YYYY-MM-DD> --end-date <YYYY-MM-DD> --use-db --non-interactive' -WorkingDirectory '$win'"
   ```

   Headless fallback (no window, capture output for the agent to read on
   failure): add `-WindowStyle Hidden -RedirectStandardOutput
   '$win\.trigger_stdout.log' -RedirectStandardError
   '$win\.trigger_stderr.log'` to the first form.

   Notes:
   - `-WorkingDirectory '$win'` is required so the relative `input/...`
     path resolves: it is matched via the CWD fallback because
     `workspace_root` in `defaults.yaml` is `./workspace`, which is
     empty. Alternatively pass an absolute config path
     `'$win\input\<project>\scenarios\<scenario>\config'` and drop the
     CWD dependence. When launched from a WSL `\\wsl.localhost\...` CWD,
     `cmd` warns about UNC paths; `-WorkingDirectory` (or `cd /d` inside
     `cmd /k`) resolves it.
   - `--start-date` is required in DB mode; `--end-date` is always
     required; `--non-interactive` skips the interactive run-selection
     prompt so the trigger does not block on stdin.
   - DB params default from `config/defaults.yaml::database`; pass
     `--db-host/--db-port/--db-name/--db-user/--db-password` only when
     targeting a non-default DB.
11. If the user explicitly requests force restart, add
   `'--force-restart'` to the `-ArgumentList` to start a fresh run-id
   instead of resuming an unfinished run.
12. Do not wait in the agent for the run to complete; the
   `Start-Process` launch returns immediately.
13. After triggering the run, do a short verification instead of
    waiting for full completion:
    - check that a new `outputs/<project>/<scenario>/db_run_*` folder
      appears and a simulation log file is being written, or
    - check that a Windows `python.exe` running `run.py` exists.
    If no new `db_run_*` folder appears, the config path is likely wrong
    (a bad `--config-dir` raises `[ConfigError]` and exits before any
    run folder is created). In the visible-window form the error shows
    in the window; for a programmatic check, re-trigger once with the
    headless fallback and read `.trigger_stderr.log`.
14. Read the confirmed run id from the output layout. Do not infer it
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

    Required behavior:

    - find the latest matching `db_run_*` folder for the scenario
    - wait until its `db_run_id.txt` exists and is non-empty, then read
      it and use its text content as the confirmed run id; this is the
      filter key for all DB result tables
    - if multiple recent run folders exist, prefer the newest valid
      `db_run_id.txt`
15. Record the confirmed run id and metadata in
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
- `design.md` updated with the Windows codebase path, config folder
  path, active workbook, command used, run dates, output location, and
  run-id retrieval method
- `results/run_id.md` updated with the confirmed latest run id read from
  `db_run_id.txt`
