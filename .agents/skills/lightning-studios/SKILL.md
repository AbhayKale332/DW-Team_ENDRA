---
name: lightning-studios
description: Manage Lightning AI Studios (cloud dev machines with CPUs/GPUs) - create, start, stop, delete studios, switch machine types, run commands in them, upload/download files, and SSH in. Use when the user wants to work with lightning.ai Studios, needs a cloud GPU dev box, or asks to run something "on a studio".
---

# Lightning AI Studios

A Studio is a persistent cloud development machine on [lightning.ai](https://lightning.ai). Its filesystem persists across sessions; compute (CPU/GPU) attaches on start and detaches on stop. Everything below uses the `lightning` CLI and `lightning_sdk` Python package.

## Setup & auth

```bash
# Use the Lightning AI CLI from the current env; install or upgrade it there if it's missing or older than 2026.9.18
v=$(lightning --version 2>/dev/null | sed -n 's/^Lightning CLI version //p')
[ -n "$v" ] && [ "$(printf '%s\n' 2026.9.18 "$v" | sort -V | head -1)" = 2026.9.18 ] \
  || uv pip install -U lightning-sdk || python3 -m pip install -U lightning-sdk
lightning --version   # must print "Lightning CLI version …"; if not, see the table below
lightning login                     # browser flow; or set env vars for headless use:
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

Credentials are stored in `~/.lightning/credentials.json`.

The install above also makes `lightning_sdk` importable in that env, so Python snippets run with
its `python`. If the CLI came from a uv tool or `uvx` fallback instead, run one-off scripts with
`uv run --with lightning-sdk python script.py`. For code that stays in the user's project, ask,
then declare `lightning-sdk` as a dependency with the project's own tool (`uv add`, `poetry add`).


## Resolving org and teamspace (do this first)

Every studio lives in a teamspace owned by either an organization or a user. **Never guess.** Resolve in this order:

1. Explicit `--teamspace owner/teamspace` flag (owner = org or user name) / `Teamspace("owner/teamspace")` in Python (the separate `org=`/`user=` arguments are deprecated).
2. Env vars `LIGHTNING_ORG`, `LIGHTNING_TEAMSPACE`, `LIGHTNING_USERNAME`, or config defaults in `~/.lightning/config.yaml` (`lightning config get teamspace`).
3. Otherwise list the options and **ask the user which org/teamspace to use**:

```bash
lightning api /v1/memberships | jq -r '.memberships[] | [.ownerType, .name, .projectId] | @tsv'
```

Persist the user's choice so they aren't asked again: `lightning config set teamspace <owner>/<teamspace>`.

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

`--teamspace` always takes `owner/teamspace`. Omitting `--name`/`--teamspace` in an interactive terminal opens a picker menu; in scripts always pass them explicitly.

```bash
# create (registers the studio, does NOT attach compute)
lightning studio create --name my-studio --teamspace owner/teamspace [--cloud PROVIDER] [--studio-type TEMPLATE]

# start compute (blocking; --create makes it if missing)
lightning studio start --name my-studio --teamspace owner/teamspace --machine CPU [--create] [--interruptible]
lightning studio start --name my-studio --teamspace owner/teamspace --gpus L4:4    # --gpus and --machine are mutually exclusive

# manage
lightning studio list --teamspace owner/teamspace [--all] [--sort-by status]   # default lists only your studios
lightning studio switch --name my-studio --teamspace owner/teamspace --machine A100   # requires Running studio
lightning studio stop --name my-studio --teamspace owner/teamspace
lightning studio delete my-studio --teamspace owner/teamspace -y              # NAME is positional here; -y/--yes: required non-interactively

# ssh / one-shot connect (create + start + ssh)
lightning studio ssh --name my-studio --teamspace owner/teamspace
lightning studio connect my-studio --teamspace owner/teamspace --machine CPU

# machine names the CLI accepts (offline list; says nothing about availability)
lightning machine list [--json]
```

### Files: `lit://` paths

Studio paths use `lit://<owner>/<teamspace>/studios/<studio-name>/<path>`. Exactly one side of a copy must be `lit://`; directories need `-r`.

```bash
lightning cp ./train.py lit://owner/teamspace/studios/my-studio/train.py     # upload
lightning cp -r lit://owner/teamspace/studios/my-studio/logs/ ./logs         # download dir
lightning ls lit://owner/teamspace/studios/my-studio                         # list files (-r recursive, --json)
lightning rm lit://owner/teamspace/studios/my-studio/old.txt [-r] [-f]
```

**`cp -r` copies a folder's contents, not the folder.** `lightning cp -r ./proj lit://…/studios/my-studio/`
puts `proj`'s files straight into the Studio's home, whether or not either path ends in `/`. To
keep the folder, name it in the destination: `lightning cp -r ./proj lit://…/studios/my-studio/proj/`.
Check the result with `lightning ls -r` before running anything that expects the files in place.

**Uploads reach the machine late, even on a ready Studio.** `lightning cp` and
`studio.upload_file` / `upload_folder` go through storage and return before the file is on the
Studio's disk, so a command launched right after can fail with `No such file or directory`. Wait
until the file is there before running anything that needs it: `upload_landed` in *Example
workflows* does this.

## Python SDK

```python
from lightning_sdk import Machine, Studio, Teamspace

ts = Teamspace("my-org/my-teamspace")                  # owner/teamspace; org=/user= are deprecated
studio = Studio(name="my-studio", teamspace=ts, create_ok=True)  # create_ok=False -> error if missing

studio.start(machine=Machine.CPU)          # blocking; interruptible=True for spot pricing
print(studio.status)                       # NotCreated|Pending|Running|Stopping|Stopped|Completed|Failed

# run commands — studio must be Running
out = studio.run("nvidia-smi")                                  # raises RuntimeError on non-zero exit
out, code = studio.run_with_exit_code("pytest -q")              # no raise on non-zero
out, code = studio.run_and_detach("python train.py", timeout=30)  # keeps running in background

studio.upload_file("train.py", remote_path="train.py")          # remote path relative to studio home
studio.download_file("outputs/model.ckpt", "model.ckpt")
studio.upload_folder("./src"); studio.download_folder("outputs/")

studio.switch_machine(Machine.A100)        # change machine while running
studio.set_env({"WANDB_API_KEY": "..."})   # env vars; partial=True merges
studio.stop()                              # releases compute, keeps filesystem
studio.delete()
```

### Machine types

Pass as `Machine.<NAME>` or string (`Machine.from_str("A100")` accepts name or slug):

- CPU: `CPU_SMALL`, `CPU` (default, 4 cores), `CPU_X_2/4/8/16`; big-disk: `DATA_PREP`, `DATA_PREP_MAX`, `DATA_PREP_ULTRA`
- GPU: `T4_SMALL`, `T4`, `T4_X_2/4/8`, `L4`, `L4_X_2/4/8`, `L40S`, `L40S_X_2/4/8`, `RTXP_6000` (+`_X_2/4/8`), `A100` (+`_X_2/4/8`), `H100` (+`_X_2/4/8`), `H200` (+`_X_2/4/8`), `B200`, `B200_X_8`
- `A100_40GB*`/`A100_80GB*` variants exist in the SDK but are hidden from CLI `--machine` (usable with `studio switch` and in Python).

SKUs get added, so treat this list as a starting point, not a closed set. `lightning machine list`
shows the names your CLI accepts, not what a teamspace can launch.

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

`cost` is USD per hour and `wait_time` the expected seconds until a machine is free.
`list_machines()` with no argument merges several accounts without saying which row belongs to
which. Show the user the top rows with price and wait. **Ask before starting any GPU, and wait for a yes.** Show the machine, its account, the price
per hour and the most the run can cost under its deadline. Ask even when the user named the GPU
or gave a budget: a budget is a limit, not approval to spend it.

**A Studio's cloud account is fixed at creation (`studio switch` can't move it), so create it on
the chosen row's account** and start or switch it to that row's machine. What matters is that the
account sells that GPU at that count, not its name: the SDK maps `H200`, `lit-h200-1` and
`lit-h200-141gb-1` to one machine.

**`cloud=` / `--cloud` takes the account id (`cluster_id`), not a display name.**
`Studio(..., cloud="Lightning Cloud")` fails with `400 clusterID Lightning Cloud is invalid`.

For quotes before login, `GET /v1/core/accelerators?cloudProvider=<PROVIDER>` needs no auth (the
`lightning-cost-estimation` skill has the provider values and the costing recipes). Don't invent a
catalog endpoint: `/v1/accelerators`, `/v1/accelerator-catalog`, `/v1/pricing` and
`/v1/compute/accelerators` all return `code: 5`.

Interruptible (spot) is a flag, not a machine type: `--interruptible` / `interruptible=True`.

### Launching on a GPU

<!-- TODO: remove this workaround once the backend rejects a GPU request it can't fill instead of
starting a CPU machine (Task Board: "CLI silently downgrades to CPU when a GPU SKU isn't available"). -->

**Ask for a GPU the Studio's cloud doesn't sell and `studio start` comes up on CPU with only a
"hasn't been vetted" warning.** The default cloud doesn't sell every GPU at every count (e.g. 1×
H200 isn't on AWS). The first workflow below guards against this:

1. **Create the Studio on an account that sells the GPU at your count** (listing above). A reused
   Studio keeps its old cloud: check it in `lightning studio list`, and create a new one if wrong.
2. **Do setup on CPU on that account, because installs, model downloads and CUDA source builds bill
   at GPU rates otherwise.** Builds need their target set, e.g. `TORCH_CUDA_ARCH_LIST=9.0` for
   H100/H200.
3. **After switching to that row's `Machine`, wait until the Studio can run commands and check the
   hardware.** No GPU means stop the Studio and go back to step 1; don't retry.

## Example workflows

Prompts this skill handles: *"spin up a GPU studio and run my training script"*, *"copy this repo to my studio and start a long run"*, *"SSH into exp-studio"*, *"my studio is idle, stop it"*.

**Set up on CPU, switch to a GPU, run, collect results, stop.** `Running` comes before a Studio
can run commands, and `studio.start()` / `lightning studio start` can keep blocking after it can,
or never return. So start in the background and poll for readiness with a deadline (see Gotchas).
Other SDK calls (`run*`, `switch_machine`, `stop`) poll with no time limit of their own, so every
call below goes through `bounded`: a stalled call raises instead of billing forever, and the
`except` stops the Studio. The `lightning-jobs` skill reuses the helpers below, with their imports:

```python
import hashlib, os, re, shlex, threading, time
from lightning_sdk import Machine, Studio, Teamspace

def bounded(fn, seconds):
    """fn()'s result, or TimeoutError after `seconds`: the SDK's own polls have no deadline."""
    done, box = threading.Event(), {}
    def _go():
        try:
            box["ok"] = fn()
        except BaseException as e:
            box["err"] = e
        finally:
            done.set()
    threading.Thread(target=_go, daemon=True).start()
    if not done.wait(seconds):
        raise TimeoutError(f"{getattr(fn, '__name__', 'call')} still blocking after {seconds}s")
    if "err" in box:
        raise box["err"]
    return box.get("ok")

def run_if_up(studio, cmd, seconds=120):
    """(output, exit code), or ("", -1) while the Studio isn't Running (starting, between machines)
    or the command doesn't answer within `seconds`."""
    try:
        return bounded(lambda: studio.run_with_exit_code(cmd), seconds)
    except (RuntimeError, TimeoutError):
        return "", -1

def stop_quietly(studio):
    """Stop billing; fall back to `lightning studio stop` if this raises."""
    if str(studio.status).endswith(("Running", "Pending")):
        bounded(studio.stop, 300)

def wait_ready(studio, minutes=10, failed=()):
    deadline = time.time() + minutes * 60
    while time.time() < deadline and not failed:
        out, code = run_if_up(studio, "python -c pass")        # not `true`: see Gotchas
        if code == 0 and "setting things up" not in out:
            return
        time.sleep(10)
    raise failed[0] if failed else TimeoutError(f"{studio.name} never finished setup: use a new Studio")

def start_ready(studio, machine, minutes=10):
    """Start on `machine` in the background and return once it can run commands."""
    if str(studio.status).endswith("Running") and studio.machine != machine:
        raise RuntimeError(f"{studio.name} is already running on {studio.machine}: stop it or use a new name")
    failed = []                                                # start() errors surface here, not in the thread
    def _start():
        try:
            studio.start(machine=machine)
        except Exception as e:
            failed.append(e)
    threading.Thread(target=_start, daemon=True).start()
    wait_ready(studio, minutes, failed)

def upload_landed(studio, local, remote, minutes=3, skip=(".git", "__pycache__", ".venv", "node_modules")):
    """Upload a file or folder (paths relative to the Studio home), hidden files included, then
    wait until the Studio's disk has every file with the right content: the upload returns before
    it lands (see *Files*). Folders named in `skip` stay behind."""
    files = [local]
    if os.path.isdir(local):
        files = []
        for root, dirs, names in os.walk(local):
            dirs[:] = [d for d in dirs if d not in skip]
            files += [os.path.join(root, n) for n in names]
    dest = {f: remote if f == local else f"{remote}/{os.path.relpath(f, local).replace(os.sep, '/')}"
            for f in files}
    for f in files:
        bounded(lambda: studio.upload_file(f, dest[f], progress_bar=False), 600)
    want = [hashlib.md5(open(f, "rb").read()).hexdigest() for f in files]
    check = "cd ~ && md5sum " + " ".join(shlex.quote(dest[f]) for f in files)
    deadline = time.time() + minutes * 60
    while time.time() < deadline:
        out, code = run_if_up(studio, check)
        if code == 0 and re.findall(r"^[0-9a-f]{32}", out, re.M) == want:
            return
        time.sleep(5)
    raise TimeoutError(f"{local} hasn't fully reached {studio.name} after {minutes} min")

ts = Teamspace("my-org/my-teamspace")
# the fastest, then cheapest, 1x H200 across the teamspace's cloud accounts (see *Machine types*)
rows = [(m.wait_time, m.cost, a.cluster_id, m) for a in ts.cloud_account_objs
        for m in ts.list_machines(cloud_account=a.cluster_id)
        if m.family == "H200" and m.accelerator_count == 1]
if not rows:
    raise SystemExit("no account here sells a 1x H200: ask the user for another GPU or teamspace")
wait, cost, acct, gpu = min(rows, key=lambda r: (r[0] is None, r[0] or 0, r[1] or 0))
print(acct, gpu.name, f"${cost}/h", f"wait ~{wait}s")        # confirm with the user before the GPU starts

studio = Studio("exp-1", teamspace=ts, cloud=acct, create_ok=True)
# an existing Studio of that name is reused as is, on whatever cloud it was created on
assert studio.cloud_account == acct, f"{studio.name} is on {studio.cloud_account}: pick a new name"

try:                                                          # any failure from here stops the Studio
    start_ready(studio, Machine.CPU)
    upload_landed(studio, "./src", "src")                     # returns once the files are on disk
    out, code = run_if_up(studio, "cd ~/src && pip install -r requirements.txt", 30 * 60)  # at CPU rates
    assert code == 0, out or "pip install failed or ran over 30 min"
    try:                                                      # the row picked from this account's list
        bounded(lambda: studio.switch_machine(gpu), 15 * 60)  # TimeoutError if it never lands
    except Exception as e:                                    # it can report this and switch anyway
        if "cannot switch to a Studio" not in str(e):
            raise
    for _ in range(30):                                       # so trust nvidia-smi, not the call
        out, code = run_if_up(studio, "nvidia-smi --query-gpu=name --format=csv,noheader")
        gpus = out.splitlines() if code == 0 else []
        if gpus:
            break
        time.sleep(10)
    print(gpus)                                               # e.g. ['NVIDIA H200']
    assert len(gpus) == 1 and "H200" in gpus[0], "wrong hardware: pick another account"
    wait_ready(studio)                                        # a new machine: wait until it can run commands
    # train.exit gets the exit code when training ends; clear the last run's before launching
    bounded(lambda: studio.run_and_detach("cd ~/src && rm -f train.exit && nohup sh -c 'python train.py > train.log 2>&1; echo $? > train.exit' </dev/null >/dev/null 2>&1", timeout=30), 120)
    print(run_if_up(studio, "tail -n 40 ~/src/train.log")[0])          # within the first minute: crashes show up in seconds
    deadline = time.time() + 2 * 3600                         # a bit over the expected run time
    while run_if_up(studio, "test -f ~/src/train.exit")[1] != 0:
        if time.time() > deadline:
            raise TimeoutError(run_if_up(studio, "tail -n 40 ~/src/train.log")[0])
        print(run_if_up(studio, "tail -n 1 ~/src/train.log")[0])  # progress line each poll
        time.sleep(60)
    assert run_if_up(studio, "cat ~/src/train.exit")[0].strip() == "0", \
        run_if_up(studio, "tail -n 40 ~/src/train.log")[0]
except BaseException:
    stop_quietly(studio)
    raise
```
```bash
lightning cp -r lit://my-org/my-teamspace/studios/exp-1/src/outputs/ ./outputs
lightning studio stop --name exp-1 --teamspace my-org/my-teamspace
```

For a short run where setup is light (a `uv` script that declares its own dependencies), skip
the CPU phase but keep the `try` block and its deadline: `start_ready(studio, gpu)`, then
`upload_landed(studio, "train.py", "src/train.py")`, drop the `pip`, switch and `nvidia-smi` steps,
and detach `env -u UV_LIGHTNING_VIRTUALENV_ROOT uv run train.py` in place of `python train.py`
(see Gotchas).

**SSH in and run scripts interactively** (for a human user; agents should prefer `studio.run*` above since `ssh` opens an interactive shell):

```bash
lightning studio ssh --name exp-1 --teamspace my-org/my-teamspace          # interactive shell
lightning ssh configure --name exp-1 --teamspace my-org/my-teamspace      # writes ~/.ssh/config Host block...
ssh exp-1 'python ~/src/eval.py'                                          # ...then plain ssh runs one-off commands
lightning studio connect exp-1 --teamspace my-org/my-teamspace --machine CPU   # create+start+ssh in one shot
```

## Raw API fallback

For anything the CLI doesn't wrap, `lightning api <path>` makes an authenticated request (`-X` method, `-F` typed field, `-f` string field, `-q` jq filter):

```bash
lightning api /v1/memberships -q '.memberships[].name'
PROJECT_ID=$(lightning api /v1/memberships | jq -r '.memberships[] | select(.name=="<teamspace>") | .projectId' | head -1)
# studios are "cloudspaces" in the API — one word, and call it BARE (see Gotchas)
lightning api "/v1/projects/${PROJECT_ID}/cloudspaces" -q '.cloudspaces[].name'
```

## Gotchas

- **Stop Studios when done: attached compute bills, and GPUs cost more.**
- **`start()` on a Studio already running on a different machine raises.** Use `switch_machine` instead.
- Disabling auto-sleep (`studio.auto_sleep = False`) or setting `auto_sleep_time` converts a free CPU studio to paid.
- **For a one-shot run (train, eval, batch), prefer a job (`lightning-jobs`).** A job stops billing
  when its command exits, crash included; a crashed run on a Studio keeps the GPU billing. The
  exception is a short run on a tight deadline: the first job from a Studio waits about 5 minutes
  for its snapshot, so run on a Studio started on the GPU, with a deadline that stops it. Pick
  one before showing the plan and name it there, with where the outputs will land.
- **Before relaunching on a GPU, check the last attempt isn't still holding it**, or the new run
  hits `CUDA out of memory`: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`
  should print nothing.
- **Read a detached run's log within the first minute, and treat a silent monitor as a failure.**
  A poller with no deadline, one that only matches progress lines, or one that can't reach
  lightning.ai (a sandboxed background command) stays quiet through a crash. Give it a deadline,
  match `Traceback`/`Error`/`Killed`/`CUDA out of memory` too, and check it prints its first line.
- **A non-interactive `lightning studio delete` without `-y` deletes nothing.** It prints
  `Are you sure you want to delete? [y/N]: Aborted.` and exits, leaving the studio (and its
  billing) alive. Confirm with the user first, then pass `-y`/`--yes`.
- **The studios list endpoint is `/cloudspaces`, one word, and takes no `-F` fields.** The hyphenated `/v1/projects/{pid}/cloud-spaces` returns `HTTP 404 Not Found`. Adding `-F limit=20` fails with `HTTP 400 Bad Request`, because `lightning api` sends any request with `-f`/`-F` fields and no `-X` as a POST (the create call). Call it bare and slice with `-q`, or pass `-X GET` to send the fields as query params.
- Inside a Studio, `Studio()` with no args resolves to the current studio (via `LIGHTNING_CLOUD_SPACE_ID`).
- **`uv run` fails in a fresh Studio** (and in jobs launched from one) with `error: failed to symlink file from /system/conda/miniconda3/uv/cache/… to /system/conda/miniconda3/uv/venvs/…: No such file or directory`. The Studio's `UV_LIGHTNING_VIRTUALENV_ROOT` points at a folder that doesn't exist yet. Run `env -u UV_LIGHTNING_VIRTUALENV_ROOT uv run …`.
- **`Running` is not ready: poll with a `python` command until the setup message is gone.** Right
  after a start, commands can fail with `Error: We are still setting things up for you, please try again after the progress bar at the top of the Studio disappears.`
  `python`, `pip` and `uv` are shell aliases (`load_conda_and_run`) that refuse until setup is
  done, while a plain `true` succeeds much earlier, so poll `studio.run_with_exit_code("python -c pass")`
  as in *Example workflows*. Setup usually takes 1–2 minutes. If it is still going after several
  minutes, stop the Studio and create a new one: starting the same one again can stay stuck.
- **`studio.switch_machine()` can raise `ApiException (400) … "cannot switch to a Studio"` and still
  switch.** Catch that error and poll `nvidia-smi` for the hardware, as in *Example workflows*;
  don't restart or re-switch on it.
- **Detach with `nohup … </dev/null >/dev/null 2>&1`.** Without the redirects, every later
  `studio.run*()` output starts with `nohup: ignoring input`, which floods a log monitor.
- **`studio.run*()` raises when the command's output isn't valid UTF-8.** Output cut through a multi-byte character (`tail -c N` on a log with `é` or a progress bar, `head -c` on a text file) fails in `cloud_space_service_get_long_running_command_in_cloud_space` (HTTP 500), even though the command itself succeeded. Read logs with `tail -n N`, or pipe through `iconv -c -f utf-8 -t utf-8`.
- **Passing both `teamspace="owner/teamspace"` and the deprecated `org=`/`user=` raises `ValueError`**
  (for `Teamspace`, `Studio` and `Job`). The old arguments alone still work but emit a `DeprecationWarning`.
