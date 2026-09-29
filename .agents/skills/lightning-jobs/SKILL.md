---
name: lightning-jobs
description: Launch and manage batch jobs on Lightning AI - run commands on cloud CPUs/GPUs from a Docker image or a Studio snapshot, monitor status, fetch logs, SSH into a running job or multi-machine worker, collect artifacts, and run multi-machine (distributed) training. Use when the user wants to run training, data processing, or any batch workload on lightning.ai, or asks to SSH into a job / MMT.
---

# Lightning AI Jobs

A Job runs a command on a dedicated cloud machine and terminates when done. Two flavors: **image jobs** (run inside any Docker image) and **studio jobs** (run inside a snapshot of an existing Studio's environment). Multi-machine distributed jobs are the same `Job` with `num_machines=N`. The older `MMT` class and `mmt` CLI group are deprecated: each `mmt` subcommand just points you to `lightning job <cmd>`.

## Setup & auth

```bash
# Use the Lightning AI CLI from the current env; install or upgrade it there if it's missing or older than 2026.9.18
v=$(lightning --version 2>/dev/null | sed -n 's/^Lightning CLI version //p')
[ -n "$v" ] && [ "$(printf '%s\n' 2026.9.18 "$v" | sort -V | head -1)" = 2026.9.18 ] \
  || uv pip install -U lightning-sdk || python3 -m pip install -U lightning-sdk
lightning --version   # must print "Lightning CLI version …"; if not, see the table below
lightning login                     # browser flow; or headless:
export LIGHTNING_API_KEY=... LIGHTNING_USER_ID=...   # USER_ID optional: with it Basic auth, without it the key is a Bearer token
```

This uses, and if needed installs into, the agent's current environment (project venv, conda env,
…); then call plain `lightning …` everywhere. If setup doesn't go cleanly:

| Symptom | Fix |
|---|---|
| No active env, `externally-managed-environment`, or a conflict with the project's own pins | Install it on its own: `uv tool install lightning-sdk` (or `pipx install lightning-sdk`), and call `"$(uv tool dir --bin)/lightning"` if that isn't on `PATH` |
| Still no `Lightning CLI version` line after installing | Another `lightning` (PyTorch Lightning's) or an inactive env comes first: check `command -v lightning`, then activate the env or call its `bin/lightning` |
| `No such command`, `No such option` or `unexpected extra argument` | The CLI is too old: upgrade it where `command -v lightning` points (`uv pip install -U lightning-sdk`, or `uv tool upgrade lightning-sdk`) |
| Installing is blocked (read-only env or home, agent sandbox) | Run each command as `UV_CACHE_DIR="${TMPDIR:-/tmp}/uv" UV_TOOL_DIR="${TMPDIR:-/tmp}/uv-tools" uvx lightning-sdk …` |
| SSL/CA errors on every call, even `lightning --version`, only inside an agent sandbox | A `**/*.pem` read-deny rule hides certifi's `cacert.pem`: allow that one path; don't disable the sandbox |
| `NameResolutionError`, only inside an agent sandbox | If `lightning api` fails too, ask the user to allow `lightning.ai` and `*.lightning.ai` (`/sandbox`). If only `studio`/`job` commands or the SDK fail, they bypass the sandbox proxy: run just those outside it. Background pollers fail the same way, silently |

The install above also makes `lightning_sdk` importable in that env, so Python snippets run with
its `python`. If the CLI came from a uv tool or `uvx` fallback instead, run one-off scripts with
`uv run --with lightning-sdk python script.py`. For code that stays in the user's project, ask,
then declare `lightning-sdk` as a dependency with the project's own tool (`uv add`, `poetry add`).

### First run: no CLI or account yet

The user may never have heard of Lightning and may not have asked for it by name. Say in one line
that the job would run on Lightning AI and why (e.g. "no GPU here"), then set up for them instead
of stopping at the first error. Run the setup block above (it installs the CLI if it's missing),
then check for a signed-in user:

```bash
lightning auth whoami || echo "not signed in"
```

If nobody is signed in, choose the path that fits the machine:

- **Has a browser (a laptop):** run `lightning login`. It opens lightning.ai, where a new user can
  create a free account. The command returns once the browser sign-in completes, so give it a few
  minutes and tell the user a tab is opening.
- **No browser (Claude Code cloud session, CI, remote box):** ask the user to sign up at
  lightning.ai, then copy `LIGHTNING_USER_ID` and `LIGHTNING_API_KEY` from their avatar →
  **Global Settings → Keys → Login via CLI**. They go in as environment variables where the agent
  runs. **Never ask for the key in chat.** In a Claude Code cloud session, both values belong in
  the cloud environment's **Environment variables**. The same dialog's **Network access** must be
  **Custom**, allowing `lightning.ai` and `*.lightning.ai` with the default list kept. Both
  changes take effect in a **new** session only.

Then continue with *Resolving org and teamspace*. A new account has one teamspace, so there's
nothing to ask.

## Resolving org and teamspace (do this first)

Jobs live in a teamspace owned by an organization or a user. **Never guess.** Use an explicit `--teamspace owner/teamspace` flag (Python: `teamspace="owner/teamspace"`), or env vars `LIGHTNING_ORG` / `LIGHTNING_TEAMSPACE`, or the config default (`lightning config get teamspace`). If none is set, list the options and **ask the user which org/teamspace to use**:

```bash
lightning api /v1/memberships | jq -r '.memberships[] | [.ownerType, .name, .projectId] | @tsv'
```

Persist the choice: `lightning config set teamspace <owner>/<teamspace>`.

The separate `--org`/`--user` flags on `job run` and the Python `org=`/`user=` arguments are deprecated in favour of the combined form, which works headlessly with env-var auth. The CLI prints a deprecation warning for the flags and will remove them. They still work today, so an existing command using them doesn't need rewriting.

**From a scoped API key** (an agent with no user to ask): `--teamspace` needs the owner's *slug*,
while `/v1/memberships` only gives the owner's id, and the same teamspace can appear twice (an
`organization` row and a `user` row with one `projectId`). Continue only if a single teamspace is
listed, prefer its organization row, and resolve the owner by `ownerType` (needs `jq`):

```bash
M=$(lightning api /v1/memberships)
if [ "$(printf "%s" "$M" | jq '[.memberships[].projectId] | unique | length')" != 1 ]; then
  echo "several teamspaces: ask the user which one" >&2
else
  ROW=$(printf "%s" "$M" | jq -c '(.memberships | map(select(.ownerType=="organization"))) + .memberships | .[0]')
  TS=$(printf "%s" "$ROW" | jq -r .name); OID=$(printf "%s" "$ROW" | jq -r .ownerId)
  if [ "$(printf "%s" "$ROW" | jq -r .ownerType)" = organization ]; then
    OWNER=$(lightning api "/v1/orgs/$OID" | jq -r .name)
  else   # a personal teamspace; the search is fuzzy, so match the id exactly
    OWNER=$(lightning api /v1/users/search -X GET -f "query=$OID" \
      | jq -r --arg id "$OID" '.users[] | select(.id==$id) | .username')
  fi
  lightning config set teamspace "$OWNER/$TS"   # or pass --teamspace "$OWNER/$TS" each time
fi
```

## CLI reference

Subcommands: `run`, `list`, `inspect`, `logs`, `ssh`, `stop`, `delete`, `rename`, `tag`, `untag`. `inspect`, `logs` and `ssh` take `--rank N` to pick one machine of a multi-machine job; on a single job `--rank` errors.

```bash
# image job
lightning job run --name my-job --teamspace owner/teamspace \
  --image python:3.11-slim --machine CPU \
  --command "python -c 'print(\"hello\")'" \
  [-e KEY=VALUE ...] [--interruptible] [--cloud PROVIDER]

# studio job (snapshot of an existing studio's environment; command required)
lightning job run --name my-job --teamspace owner/teamspace \
  --studio my-studio --machine A100 --command "python train.py"

# omit both --studio and --image while running inside a Studio to target THAT Studio
# (only if its teamspace matches the resolved --teamspace) — no lookup needed:
lightning job run --name my-job --teamspace owner/teamspace --machine A100 --command "python train.py"

# private registry image
lightning job run ... --image-credentials <secret-name> [--cloud-account-auth]  # --cloud-account-auth for ECR-type registries

# monitor / manage
lightning job list --teamspace owner/teamspace [--all] [--sort-by status] [--json]   # includes multi-machine jobs
lightning job inspect my-job --teamspace owner/teamspace      # JSON incl. status, machine, cost
lightning job stop my-job --teamspace owner/teamspace
lightning job delete my-job --teamspace owner/teamspace -y   # -y/--yes: required non-interactively

# logs — snapshot by default; --follow streams a running job until it finishes or Ctrl-C
lightning job logs my-job --teamspace owner/teamspace [--follow] [--tail 100] [--timestamps]
lightning job logs my-job --teamspace owner/teamspace --query error --severity error   # filter server-side
lightning job logs my-job --teamspace owner/teamspace --since 2h --until 30m           # window: duration (30s/2h/3d/1w) or RFC3339

# ssh into a running job (fails unless status is Running)
lightning job ssh my-job --teamspace owner/teamspace

# multi-machine training: same command plus --num-machines
lightning job run --name my-mmt --teamspace owner/teamspace \
  --image pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime --num-machines 2 --machine L4 \
  --command "python -m torch.distributed.run --nproc_per_node=1 train.py"

# multi-machine logs: every rank merged and labelled; --rank N reads one machine
lightning job logs my-mmt --teamspace owner/teamspace [--follow] [--tail 50] [--rank 1]

lightning job ssh my-mmt --teamspace owner/teamspace              # rank 0 by default
lightning job ssh my-mmt --rank 1 --teamspace owner/teamspace    # pick a worker
```

- **There is no `status` subcommand: read `status` from `inspect`.** It prints plain JSON (`--json` is accepted too) with `command`, `image`, `machine`, `name`, `status`, `studio`, `tags`, `teamspace` and `total_cost`, so `lightning job inspect <name> | jq -r .status` is a fine polling source. For many jobs at once, `lightning job list --json` includes status, `started_at`/`stopped_at`, `total_cost` and `num_machines`.
- **A `Pending` job may have no logs yet**, because no machine is scheduled. Once it runs, `logs` returns what is available so far.
- **`ssh` is for a human user.** It opens an interactive shell, so agents should prefer `inspect`, `logs` and the Python SDK. On a Pending, Completed, Failed or Stopped job it raises a clean error.

## Python SDK

```python
from lightning_sdk import Job, Machine, Status, Studio, Teamspace

ts = Teamspace("my-org/my-teamspace")   # owner/teamspace; org=/user= are deprecated

job = Job.run(
    name="my-job",                    # on a clash the platform renames it and the SDK warns
    machine=Machine.CPU,              # or "A100", Machine.from_str("L4"), ...
    image="python:3.11-slim",         # OR studio=<Studio|name> — mutually exclusive
    command="python train.py",        # required for studio jobs, optional for image jobs
    teamspace=ts,
    env={"RUN_MODE": "prod"},
    interruptible=False,              # True = spot: cheaper, can be preempted
    max_run_attempts=2,               # optional retries; None = backend default
)
print(job.link)                       # web UI URL

job.wait(interval=10, timeout=3600, stop_on_timeout=True)   # blocks until terminal
print(job.status)                     # Status.Pending / Running / Stopping / Stopped / Completed / Failed
if job.status == Status.Failed:
    print(job.logs)                   # snapshot of logs so far (also: `lightning job logs <name>`)
print(job.total_cost)                 # USD

job.stop(); job.delete()

# distributed job — same Job.run plus num_machines; per-worker access via .machines
mmt = Job.run(name="my-mmt", num_machines=2, machine=Machine.L4, image="...", command="...", teamspace=ts)
mmt.wait()
for worker in mmt.machines:           # rank-ordered; each worker is a Job
    print(worker.name, worker.status) # per-node logs via worker.logs; mmt.logs (or `lightning job logs`) merges all ranks
```

Fetch an existing job: `Job("my-job", teamspace=ts)` (raises `ValueError` if it doesn't exist).

Leaving both `studio=` and `image=` unset targets the Studio you're currently running inside (via `LIGHTNING_CLOUD_SPACE_ID`), like omitting both `--studio` and `--image` on the CLI. Outside a Studio, or when its teamspace doesn't match, either form raises an error asking for one explicitly.

### Image vs studio jobs

| | Studio job | Image job |
|---|---|---|
| `command` | required | optional (falls back to image entrypoint) |
| `entrypoint`, `image_credentials`, `cloud_account_auth` | forbidden | allowed |
| artifacts | write outputs to the job's **home** (`$LIGHTNING_ARTIFACTS_DIR`); see below | none by default; route via `path_mappings={"<container-path>": "<connection>:<path>"}` |
| scratch disks | `scratch_disks={"data": 100}` (GiB, under `/teamspace/scratch/`) | forbidden |

### Outputs & artifacts (studio jobs)

A studio job runs with its home at the current-Studio home mount,
`/teamspace/studios/this_studio`. The path is the same whichever Studio you launched from, and
`$LIGHTNING_ARTIFACTS_DIR` points there. **To keep an output, write it under home through the env
var, not a hardcoded path.** Every file the job creates or modifies under home during the run is
captured as a job artifact.

```python
import os, joblib
out = os.environ["LIGHTNING_ARTIFACTS_DIR"]     # == the source Studio's home path
joblib.dump(model, f"{out}/model.joblib")        # a new/changed file under home -> artifact
```

- **Read the results back from the source Studio under `/teamspace/jobs/<job-name>/artifacts/`
  once the job is terminal.** That path is read-only: writing to it from inside the job fails
  with `OSError: [Errno 30] Read-only file system`.
- **A job cannot change the live Studio filesystem.** Outputs never reappear in the Studio's
  home after the run.

**Fetch artifacts from anywhere through the teamspace Drive.** The capture also surfaces there
under `jobs/<job-name>/`, so reading it does not require a Studio. In Python,
`job.artifacts_uri` gives that `lit://` address, and `job.list_artifacts(path, recursive=True)` /
`job.download_artifacts(target_dir, path)` read it (per machine on a multi-machine job). From the shell:

```bash
lightning cp lit://<owner>/<teamspace>/jobs/<job-name>/model.joblib ./model.joblib   # one file
lightning cp -r lit://<owner>/<teamspace>/jobs/<job-name>/outputs/ ./outputs         # a folder
```

**List and copy a specific subfolder, never the `jobs/<job-name>/` root.** The capture can hold
the job's whole home, dotfiles and caches included. `lightning ls` on the root can hang, while
`lightning ls lit://<owner>/<teamspace>/jobs/<job-name>/outputs` returns quickly. In Python,
`job.list_artifacts("outputs", recursive=True)` is instant, and
`job.download_artifacts("outputs", "outputs")` writes the *contents* of the job's `outputs/`
into the local `./outputs`. Deleting the job deletes this tree with it.

**Image (docker) jobs have no home-artifact collection and no Lightning credentials inside.**
The container gets `LIGHTNING_CLOUD_URL`, the job and project ids and `LIGHTNING_USERNAME`, but no
`LIGHTNING_API_KEY`, so `lightning cp` to the Drive fails from there. Print small results to the
log (read them back with `lightning job logs`), mount an output location with `path_mappings`,
or use a studio job when files must come back.

### Machines

`CPU_SMALL`, `CPU`, `CPU_X_2/4/8/16`, `DATA_PREP(_MAX/_ULTRA)`, `T4_SMALL`, `T4(_X_2/4/8)`, `L4(_X_2/4/8)`, `L40S(_X_2/4/8)`, `RTXP_6000(_X_2/4/8)`, `A100(_X_2/4/8)`, `H100(_X_2/4/8)`, `H200(_X_2/4/8)`, `B200(_X_8)`. Multi-GPU `_X_N` variants bill N GPUs; multi-machine jobs bill per machine × `num_machines`.

The names above are current as of writing and SKUs do get added, so treat the list as a starting
point, not a closed set. `lightning machine list` prints the names your installed CLI accepts, but
it is an offline list and says nothing about what a teamspace can launch.

**Pick the machine from live data, per cloud account.** A teamspace can launch on several cloud
accounts, and the same GPU differs between them in name, price and wait. List what each account
can start right now, fastest first:

```python
from lightning_sdk import Teamspace
ts = Teamspace("my-org/my-teamspace")
FAMILY = "H100"   # the GPU family the workload needs
# Accounts go by id: ts.cloud_accounts holds display names, and list_machines() returns [] for a name
rows = [(m.wait_time, m.cost, a.cluster_id, m) for a in ts.cloud_account_objs
        for m in ts.list_machines(cloud_account=a.cluster_id) if m.family == FAMILY]
for wait, cost, acct, m in sorted(rows, key=lambda r: (r[0] is None, r[0] or 0, r[1] or 0)):
    print(f"{acct:34} {m.name:28} x{m.accelerator_count}  ${cost}/h  wait ~{wait}s")
```

`cost` is USD per hour and `wait_time` the expected seconds until a machine is free. Launch with
the row's machine on that row's account (`Job.run(machine=m, cloud=acct, …)`, or a Studio created
with `cloud=acct`): what matters is that the account sells that GPU at that count. The SDK maps a
GPU's names (`H200`, `lit-h200-1`, `lit-h200-141gb-1`) to one machine. `list_machines()` with no
argument merges several accounts without saying which row belongs to which, so it can't tell you
where to launch. Show the user the top rows with price and wait. **Ask before starting any GPU, and wait for a yes.** Show the machine, its account, the price
per hour and the most the run can cost under its deadline. Ask even when the user named the GPU
or gave a budget: a budget is a limit, not approval to spend it. For
quotes before login, `GET /v1/core/accelerators?cloudProvider=<PROVIDER>` needs no auth (the `lightning-cost-estimation`
skill has the provider values and the costing recipes). Don't invent a catalog endpoint:
`/v1/accelerators`, `/v1/accelerator-catalog`, `/v1/pricing` and `/v1/compute/accelerators` all
return `code: 5`.

<!-- TODO: remove this workaround once the backend rejects a GPU request it can't fill instead of
starting a CPU machine (Task Board: "CLI silently downgrades to CPU when a GPU SKU isn't available"). -->
**Pick the cloud account for a GPU job before launching.** The default cloud doesn't sell every
GPU at every count (1× H200 isn't on AWS). A request the account can't fill can fail with
`accelerator … not found for this AWS cluster`, whatever the machine is called, but Studios asked
for one have also silently come up on CPU (see `lightning-studios`), so don't count on a loud
failure. Take the account's `cluster_id` from the listing above, not a display name from
`Teamspace.cloud_accounts` such as `Lightning Cloud`: `Studio(cloud=…)` fails on a name with
`clusterID Lightning Cloud is invalid`. From the CLI, pass the account as `--cloud` with the GPU
name:

```bash
lightning job run --name my-job --teamspace owner/teamspace --machine H200 --cloud lightning-baremetal \
  --image python:3.12-slim --command "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
```

### The first minute after launch

Most job failures happen in the first seconds: a wrong machine, a missing package, a renamed
argument. Once the job is `Running`, read its log before settling in to wait, and stop it at
the first traceback instead of letting it sit:

```bash
lightning job logs my-job --teamspace owner/teamspace --tail 40   # confirm the GPU line and the first steps
lightning job stop my-job --teamspace owner/teamspace             # only if the log shows a crash or the wrong machine
```

Put a hardware check first in the job's own command, e.g.
`nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && python train.py`. The first log
line then tells you what you actually got.

## Example workflows

Prompts this skill handles: *"run this script on an A100 as a batch job"*, *"launch my docker image on lightning"*, *"why did my job fail — show me the logs"*, *"SSH into my running job"*, *"SSH into rank 1 of my multi-machine job"*, *"run a 2-node distributed training"*.

**Run a local script on a GPU and bring its output files back** (a studio job, since image jobs
can't return files on their own). Pick `ACCOUNT` and `MACHINE` from the live listing in
*Machines*, and confirm the price with the user first:

```python
import time
from lightning_sdk import Job, Machine, Studio
# start_ready, upload_landed, stop_quietly: from the lightning-studios skill (*Set up on CPU, switch to a GPU*),
# with its other helpers and imports

studio = Studio("run-train", teamspace="my-org/my-teamspace", cloud=ACCOUNT, create_ok=True)
# an existing Studio of that name is reused as is, on whatever cloud it was created on
assert studio.cloud_account == ACCOUNT, f"{studio.name} is on {studio.cloud_account}: pick a new name"
try:                                                  # the Studio stops on any failure, and once the job starts
    start_ready(studio, Machine.CPU)                  # CPU is enough: it only holds the files
    upload_landed(studio, "train.py", "train.py")     # the Studio home, the job's working dir; the
                                                      # snapshot is taken at Job.run, so it must be there
    job = Job.run(name=f"train-{int(time.time())}", machine=MACHINE, studio=studio,  # no teamspace=,
                  cloud=ACCOUNT,                      # else the default cloud wins and can mismatch the Studio's
                  command="env -u UV_LIGHTNING_VIRTUALENV_ROOT uv run train.py",       # see Gotchas
                  max_run_attempts=1)
    deadline = time.time() + 20 * 60                  # the first job from a Studio waits ~5 min for a snapshot
    while str(job.status).endswith("Pending"):
        if time.time() > deadline:
            job.stop()
            raise TimeoutError(f"{job.name} still Pending after 20 min")
        time.sleep(15)
finally:                                              # the job runs on the snapshot, not the Studio
    stop_quietly(studio)
job.wait(interval=15, timeout=2 * 3600, stop_on_timeout=True)
print(job.status, job.total_cost)
job.download_artifacts("outputs", "outputs")         # whatever train.py wrote under ./outputs
```

Later jobs from the same Studio reuse its snapshot and leave `Pending` sooner.

**Prefer a job to a Studio for one-shot runs (train, eval, batch).** A job stops billing when its
command exits, crash included. A crashed run on a Studio leaves the GPU billing idle until someone
notices. The exception is a short run on a tight deadline: there, a Studio started directly on the
GPU, with a deadline that stops it, finishes sooner than waiting for the snapshot (see the
`lightning-studios` skill). **Pick one before showing the plan, and name it there** with where
the outputs will land: a job's under `jobs/<job-name>/` on the teamspace Drive, a Studio's in its
home, which stays downloadable while the Studio is stopped. Don't promise one and run the other.

**Parameter sweep: several jobs from one loop.**

```python
for lr in ["1e-3", "3e-4", "1e-4"]:
    Job.run(name=f"sweep-lr-{lr}", machine=Machine.T4, studio="exp-1",
            command=f"python train.py --lr {lr}", env={"WANDB_RUN": f"lr-{lr}"},
            teamspace="my-org/my-teamspace", interruptible=True)
# each writes outputs to home ($LIGHTNING_ARTIFACTS_DIR); read them from the Studio under /teamspace/jobs/<name>/artifacts
```

## Raw API fallback

For what the CLI doesn't wrap, chiefly exact-cost JSON and other raw resource fields.

**Call the list endpoints as plain GETs and slice client-side with `-q`.** `lightning api` sends
any request with `-f`/`-F` fields and no `-X` as a POST (the create call). So `-F limit=…` without
`-X GET` gets `400 "spec is required"` on `/jobs` and `400 "name is required"` on
`/multi-machine-jobs`.

```bash
PROJECT_ID=$(lightning api /v1/memberships | jq -r '.memberships[] | select(.name=="<teamspace>") | .projectId' | head -1)

# list jobs (id + name) — plain GET, no -F
lightning api "/v1/projects/${PROJECT_ID}/jobs" -q '.jobs[] | [.id, .name] | @tsv'

# inspect one job as JSON (status, machine, cost, timestamps) — by JOB ID (job_...), not name
lightning api "/v1/projects/${PROJECT_ID}/jobs/${JOB_ID}"

# list multi-machine jobs — also a plain GET, no -F
lightning api "/v1/projects/${PROJECT_ID}/multi-machine-jobs" -q '.multiMachineJobs[].name'
```

`JOB_ID` is the `job_...` id from the list call; the per-job endpoints 404 on the human name.
**To find one job by name, filter the list.** The `/jobs/find` route returned `501 Not
Implemented` when last tested live. The SDK's `Job("<name>")` calls it as
`GET .../jobs/find?name=<name>`, which may be what works, but that form is untested here. For
everyday use prefer the CLI: `lightning job list --json`, `lightning job inspect <name>`,
`lightning job logs <name>`.

## Gotchas

- Jobs bill machine time while allocated; always confirm with the user before launching on a GPU or with `num_machines` > 1 (see *Machines*), and prefer `wait(..., stop_on_timeout=True)` so runaway jobs get stopped.
- **Treat a silent monitor as a failure.** A poller with no deadline, one that only matches progress lines, or one that can't reach lightning.ai (a sandboxed background command) stays quiet through a crash. Poll `job.status` with a deadline, react to `Failed`/`Stopped` as well as `Completed`, and check the poller prints its first line.
- **`max_runtime` is not a time or spend cap.** It is a DWS reservation duration, applied only on non-spot machines that are `dws_supported`/`dws_only`, and a no-op elsewhere (verified inert on an L40S run). `job.wait(timeout=..., stop_on_timeout=True)` does stop a job, but only while the calling process survives. So don't promise an unattended bound on an ordinary machine: say what enforces it, and keep something alive to call `job.stop()`.
- **A slow `Pending` start is not a failure, but `Pending` doesn't mean unbilled.** Image jobs can sit in `Pending`/`creating` for a long time before a machine is scheduled, and that queue time is free. Provisioning also reports `Pending` and is billed: billing runs `startedAt`→`stoppedAt`, which covers the image pull (see `lightning-cost-estimation`), so stopping mid-provision isn't free. The machine listing's `wait_time` predicts the queue. If you give up at your deadline, `job.stop()` then `job.delete()`.
- **`lightning job delete` without `-y` exits without deleting.** Non-interactively it reads the prompt from a closed stdin, prints `Are you sure you want to delete? [y/N]: Aborted.` and leaves the job listed and costing money. The failure is easy to miss in a log.
- **`--query` and `--severity` can't filter every *finished* job.** Where a job's logs are stored decides this: if its lines aren't in the newer log storage, a finished job falls back to its saved log file, which can't be filtered server-side. The CLI and SDK then raise `This job's logs are only available as a saved file ... filter locally` rather than returning nothing, so fetch unfiltered and `grep` locally. `--timestamps` works on both paths. While a job is still `Running` the filters are applied server-side and work.
- **`job inspect` has no exit code or failure message, so read the raw record's `message` field to learn why a job failed:** `lightning api "/v1/projects/$PID/jobs" -q '.jobs[] | select(.name=="<name>") | .message'`. Start/stop times are on `Job.started_at`/`Job.stopped_at` and `job list --json`.
- A studio job's Studio must be in the same teamspace and cloud account as the job.
- **In Python, don't pass `teamspace=` next to a `Studio` object.** `Job.run(studio=s, teamspace="owner/ts")` raises `ValueError: Studio teamspace does not match provided teamspace` even when it is the Studio's own teamspace. Leave `teamspace` out: the job takes it from the Studio.
- **`uv run` fails inside a studio job** with `error: failed to symlink file from /system/conda/miniconda3/uv/cache/… to /system/conda/miniconda3/uv/venvs/…: No such file or directory`. The Studio's `UV_LIGHTNING_VIRTUALENV_ROOT` points at a folder that doesn't exist there (the same happens inside a fresh Studio). Run `env -u UV_LIGHTNING_VIRTUALENV_ROOT uv run …` (unsetting every `UV_*` variable works too).
- **A job goes `Running` → `Pending` → `Completed` (or `Failed`), and the second `Pending` is not a retry.** It is the platform saving the job's outputs after the command exits. Don't stop the job then; wait with `job.wait()`, which returns only on a terminal state.
- **`job.logs` is not a string.** `print(job.logs)` works, but slicing it (`job.logs[-2000:]`) raises `TypeError: '_Logs' object is not subscriptable`. Use `lightning job logs <name> --tail N` for the end of a log.
- **A taken job name is silently replaced, so read `job.name` back.** Names are unique per teamspace: on a clash the platform creates the job under a new name and the SDK warns `the job was created as '<new>' instead`. Omitted `--name` auto-generates one.
- `--machine` is **case-sensitive** (`--machine a100` fails with `Invalid value for '--machine'`); use the exact names above. A100_40GB/A100_80GB variants are SDK-only (hidden from CLI).
- `job.stop()` blocks (polls every 1s) until the job reaches a terminal state.
