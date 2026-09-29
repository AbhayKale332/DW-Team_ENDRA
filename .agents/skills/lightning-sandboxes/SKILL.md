---
name: lightning-sandboxes
description: Run code in Lightning AI Sandboxes - fast, isolated ephemeral VMs with optional persistence, snapshots, custom images, Docker-in-sandbox (docker / docker compose), network egress policies, file I/O, public port URLs, and interactive PTY sessions. Use when the user wants to execute untrusted or experimental code safely, needs a throwaway cloud VM, wants to build/run containers or preview a containerized app, or asks about lightning.ai sandboxes or snapshots.
---

# Lightning AI Sandboxes

A Sandbox is a fast-booting isolated VM for code execution. It is ephemeral by default: stop deletes it, and only `persistent=True` enables stop/resume via auto-snapshots. Import from the subpackage with `from lightning_sdk.sandbox import Sandbox`. The top-level `lightning_sdk.sandbox.py` `_Sandbox` class is legacy; ignore it.

## Setup & auth (sandbox-specific)

```bash
# Use the Lightning AI CLI from the current env; install or upgrade it there if it's missing or older than 2026.9.18
v=$(lightning --version 2>/dev/null | sed -n 's/^Lightning CLI version //p')
[ -n "$v" ] && [ "$(printf '%s\n' 2026.9.18 "$v" | sort -V | head -1)" = 2026.9.18 ] \
  || uv pip install -U lightning-sdk || python3 -m pip install -U lightning-sdk
lightning --version   # must print "Lightning CLI version …"; if not, see the table below
# `sandbox <cmd>` is also installed standalone == `lightning sandbox <cmd>`
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

**Org scope comes from the API key — there is no org flag or `LIGHTNING_ORG_ID` env var (it's rejected).** Sandboxes need an **org- or teamspace-scoped API key** in `LIGHTNING_SANDBOX_API_KEY`; a personal `lightning login` credential fails with *"Use a teamspace- or org-scoped API key (Members → API keys), not your personal login key."*

You can mint an org-scoped key from the CLI (requires working personal auth):

```bash
export LIGHTNING_SANDBOX_API_KEY=$(lightning api-key create --org my-org --name sandbox-agent)
# or reuse the org default key: lightning api-key get --org my-org
```

If the user belongs to multiple orgs and none is configured, **ask which org to use**:

```bash
lightning api /v1/memberships | jq -r '.memberships[] | select(.ownerType=="organization") | [.ownerId, .name] | @tsv'
```

Snapshot/stop of persistent sandboxes needs a **teamspace-scoped** key (org-scoped is not enough).

## CLI reference

```bash
# create — blocks until running (usually seconds)
lightning sandbox create --name devbox \
  [--instance-type cpu-1]            # default cpu-1 \
  [--runtime python313]              # DEFAULT IS node24 (Node.js, NO python!) — use python313 for Python \
                                     # node22 | node24 | python313, each with a -docker variant \
  [--image ghcr.io/org/img:latest]   # custom rootfs, CPU-only; mutually exclusive with --runtime \
  [--image-secret-ref <docker-registry-secret>] \
  [--docker]                         # dockerd at boot; == appending -docker to --runtime \
  [--teamspace owner/teamspace] [--persistent] [--spot] [--port 8000] \
  [--snapshot-id snap-...]           # warm start from a snapshot \
  [--timeout 3600000]                # auto-stop lifetime in MILLISECONDS \
  [--json]                           # prints the sandbox id — capture it for later commands

lightning sandbox list [--teamspace owner/teamspace] [--limit N] [--json]

# run a command (use -- before the command); CLI exits with the command's exit code
lightning sandbox run <SANDBOX_ID> [--cwd /workspace] [--env KEY=VALUE] -- python -c "print('hi')"
lightning sandbox run <SANDBOX_ID> --detached -- bash -lc "long-task"   # prints cmd_id
lightning sandbox command  <SANDBOX_ID> <COMMAND_ID>     # status + output
lightning sandbox logs     <SANDBOX_ID> <COMMAND_ID>
lightning sandbox commands <SANDBOX_ID>                  # history

# lifecycle
lightning sandbox stop   <SANDBOX_ID>    # persistent: auto-snapshot + pause; ephemeral: == delete
lightning sandbox start  <SANDBOX_ID>    # resume a stopped persistent sandbox
lightning sandbox delete <SANDBOX_ID> -y # destroys sandbox AND its auto-snapshot; -y required non-interactively

# snapshots (filesystem only — running processes are not preserved)
lightning sandbox snapshot create <SANDBOX_ID> [--expiration <MS>] [--exclude PATH]
lightning sandbox snapshot list [--name N] [--teamspace owner/teamspace]
lightning sandbox snapshot get <SNAPSHOT_ID>
lightning sandbox snapshot delete <SNAPSHOT_ID> -y
```

Runtime ids are `node22`, `node24` and `python313`, each with a `-docker` variant. Any other id fails with `invalid runtime: <id>`. A custom `--image` runs on CPU under gVisor. In the SDK the same options are `image=` and, for a private registry, `image_secret_ref=` naming a Docker-registry secret.

There is no `cp`/upload CLI. Move files with `run` and shell commands, or with the SDK file API below.

## Python SDK

```python
from lightning_sdk.sandbox import Sandbox, SandboxConfig, RunCommandOpts, NetworkPolicy, PtyCreateOpts

# optional explicit config; otherwise env (LIGHTNING_SANDBOX_API_KEY) / lightning login creds are used
Sandbox.configure(api_key="...")

sb = Sandbox.create(
    name="devbox",
    instance_type="cpu-1",
    runtime="python313",                   # default is node24 (Node.js only — no python!)
    teamspace="owner/teamspace",
    persistent=True,                       # enables stop()/resume()
    docker=True,                           # dockerd at boot; see "Docker inside a sandbox"
    ports=[8080],                          # exposes public HTTPS URLs for these ports
    network_policy=NetworkPolicy(allow_cidrs=["10.0.0.0/8"]),  # or "deny-all" / default open egress
    timeout=30 * 60 * 1000,                # auto-stop, MILLISECONDS
)                                          # blocks until running

# commands — non-detached blocks until exit
cmd = sb.run_command("python -c 'print(42)'")
print(cmd.exit_code, cmd.output)           # output = combined stdout+stderr
bg = sb.run_command(RunCommandOpts(cmd="python", args=["train.py"], cwd="/workspace",
                                   env={"MODE": "test"}, detached=True))
bg.wait(timeout=600); print(bg.output)     # or sb.kill_command(bg.cmd_id)

# files — content strings via REST; no host<->sandbox copy helper
sb.write_file("/workspace/app.py", "print('hi')")
text = sb.read_file("/workspace/out.txt")  # None if missing
sb.fs.mkdir("/workspace/data", recursive=True)
sb.fs.exists("/workspace/app.py"); sb.fs.readdir("/workspace"); sb.fs.rm("/tmp/x", recursive=True)

# exposed ports (only those passed as ports=[...] at create) — proxied public HTTPS
sb.port_urls                               # {"8080": "https://8080-<sandbox-id>-s.cloudspaces.litng.ai"}
sb.get_port_url(8080)                      # ValueError if 8080 wasn't exposed at create
# normally populated by create; if empty, re-get: Sandbox().get(sb.sandbox_id).port_urls

# lifecycle
sb.extend_timeout(10 * 60 * 1000)          # heartbeat; milliseconds, min 1000
snap = sb.snapshot()                       # filesystem snapshot; later: Sandbox.create(snapshot_id=snap.id)
auto_snap_id = sb.stop()                   # persistent only; sb.resume() brings it back with same id
sb.delete()                                # ALWAYS clean up — GC does not delete remote sandboxes

# find existing
client = Sandbox()
for s in client.list(teamspace="owner/teamspace").sandboxes: print(s.sandbox_id, s.status)
sb = client.get("sbx-...")
```

Interactive PTY (`websocket-client` ships with the SDK): `sb.process.create_pty(PtyCreateOpts(session_name="main"))` → `pty.send_input("ls\n")`, `pty.wait()`. The result's `exit_code` is the shell's real status on a clean close, `0` when the backend reports none, `-1` if the connection broke (`error` says why) and `None` while running. Prefer `run_command` when you need a guaranteed exit code.

## Docker inside a sandbox

The `-docker` runtimes ship Docker (engine, CLI, buildx, compose), and **`dockerd` is already running when create returns**, so `docker version` works at once. Pick the variant explicitly, or pass `--docker` / `docker=True` to append the suffix to the base runtime:

```bash
lightning sandbox create --runtime python313-docker --port 8080 --json   # cpu-1 default is fine for Docker
lightning sandbox create --runtime python313 --docker   # identical — resolves to python313-docker
lightning sandbox create --docker                       # no runtime -> node24-docker (Node, NOT Python)
```

`docker=True` is rejected together with `image` (a custom image opts in by carrying the OCI label `ai.lightning.sandbox.docker=true`) or `snapshot_id` (the runtime comes from the snapshot).

### Where images are stored, and how much room you get

**`cpu-1` and `cpu-4` keep images on disk; every other shape keeps them in RAM.** Pick the instance type for where `/var/lib/docker` lands, not for its RAM. On the disk shapes an image never touches memory, so the default `cpu-1` is a usable Docker box despite its small RAM.

| `--instance-type` | RAM       | `/var/lib/docker` | image room | `storage_gb` raises it? |
| ----------------- | --------- | ----------------- | ---------- | ----------------------- |
| `cpu-1` (default) | 1.25 GiB  | disk              | 5 GB       | yes                     |
| `cpu-2`           | 8.75 GiB  | RAM (tmpfs)       | 4.4 GB     | no                      |
| `cpu-4`           | 18.75 GiB | disk              | 40 GB      | yes                     |
| `cpu-8`           | 38.75 GiB | RAM (tmpfs)       | 20 GB      | no                      |
| `cpu-16`          | 77.5 GiB  | RAM (tmpfs)       | 39 GB      | no                      |

- **`df -h /var/lib/docker` reports the real ceiling; trust it over this table.** The table holds measured current defaults, so use it to plan and check `df` before sizing a pull.
- **On the disk shapes there is one quota for everything.** `/`, `/tmp` and `/var/lib/docker` draw on the same pool, so a 3 GB file in `/root` leaves 3 GB less for images. Raise it with the SDK-only `storage_gb` (`Sandbox.create(..., instance_type="cpu-1", storage_gb=20)` gives docker 20 GB). There is no CLI flag for it.
- **On the RAM shapes the cap is half the sandbox's memory, and it counts against your memory budget.** An image plus a hungry process can still cause an out-of-memory kill. `storage_gb` buys disk for `/`, never image room.
- **Choose by workload:** `cpu-1` for image-heavy work on a budget, `cpu-4` for both room and RAM for builds, `storage_gb` when 5 GB is tight. Use `cpu-2`/`cpu-8` only when the workload needs the memory, and keep images small there.

**A pull into a full data root fails with `no space left on device`, and the sandbox keeps running.** Delete images and retry:

```bash
lightning sandbox run $SBX -- bash -lc 'docker system df; docker image prune -af; df -h /var/lib/docker'
```

**Older clusters kill the sandbox instead.** There `df` reports a meaningless size of hundreds of GB, nothing bounds docker but the sandbox's memory, and overrunning it terminates the sandbox: every later call returns `429 sandbox terminated: out of memory`. Keep images well under the RAM column, or move to `cpu-1`/`cpu-4`.

### Networking: the part you must adapt

`dockerd` runs `--bridge=none --iptables=false --ip6tables=false`, so only the `host` and `none` docker networks exist (`docker network ls` confirms it):

- **Run containers with `--network=host`, or they have no network at all.** DNS fails with `wget: bad address`. Image pulls still work either way, since dockerd itself sits on the sandbox's netstack.
- **Build with `--network=host`, or every `RUN` step dies** with `failed to solve: ... network bridge not found`. In compose: `build: {context: ., network: host}`.
- **Reach other containers at `127.0.0.1:<port>`, never by compose service name.** Host networking has no bridge DNS, so `db` or `api` dies with `gaierror: [Errno -2] Name or service not known`. Plain `localhost` resolves fine.
- **`-p 8080:80` is silently ignored: `docker run -p` exits 0 and serves nothing.** Port publishing belongs to the bridge. The process inside must listen on the port you want (`nginx` configured for 8080, `uvicorn --port 8080`). That port is what the sandbox's public URL serves, if you declared it at create (`--port 8080` / `ports=[8080]`, then `sb.get_port_url(8080)`).

Compose template that works as-is (FastAPI + Redis). Run `docker compose up -d --build` detached with a generous timeout, since builds pull layers:

```yaml
services:
  web:
    build: {context: ., network: host}    # host net so pip install reaches PyPI
    network_mode: host
    environment: [REDIS_HOST=127.0.0.1]   # NOT "redis" — no bridge DNS
    depends_on: [redis]
  redis:
    image: redis:7-alpine
    network_mode: host
```

### Docker gotchas

- **Image pulls fail under restricted egress; pull under the default open egress or bake images into a custom image.** Under `network_policy="deny-all"` dockerd still starts, but every pull stalls, then fails with `Get "https://registry-1.docker.io/v2/": context deadline exceeded`. A CIDR allowlist that includes the resolvers (see the DNS gotcha below) fixes name resolution, but the pull then waits on registry IPs, which Docker Hub serves from a shifting CDN.
- **Docker Engine is pinned to v27; don't upgrade it in-sandbox.** v28+ is known-broken here and unsupported. `docker info` shows the exact build. Docker is CPU-sandbox only, never GPU.
- **You cannot add Docker to a plain runtime after create; pick a `-docker` runtime up front.** On a plain runtime `docker` and `dockerd` are absent and `mount` returns `permission denied`.
- **The REST API has no `docker` field.** If you drive `/v1/core/sandboxes` directly, send `"runtime": "node24-docker"`.

## Example workflows

Prompts this skill handles: *"run this untrusted script somewhere safe"*, *"test my code in a clean VM"*, *"spin up a container from ghcr.io/... and poke around"*, *"give me a devbox that survives restarts"*.

**Safely execute untrusted/generated code (no network egress, auto-cleanup):**

```python
from lightning_sdk.sandbox import Sandbox

sb = Sandbox.create(name="quarantine", teamspace="my-org/my-teamspace", runtime="python313",
                    network_policy="deny-all", timeout=15 * 60 * 1000)  # hard kill after 15 min
try:
    sb.write_file("/workspace/suspect.py", open("suspect.py").read())
    cmd = sb.run_command("python /workspace/suspect.py")
    print(cmd.exit_code, cmd.output)
finally:
    sb.delete()
```

**Test code in a clean environment from the CLI:**

```bash
SBX=$(lightning sandbox create --name test-run --teamspace my-org/my-teamspace --runtime python313 --timeout 1800000 --json | jq -r .id)
lightning sandbox run $SBX -- python --version
lightning sandbox run $SBX --cwd /workspace -- bash -lc "pip install requests && python -c 'import requests; print(requests.__version__)'"
lightning sandbox run $SBX --detached -- bash -lc "pytest -q > /workspace/test.log 2>&1"   # prints cmd_id
lightning sandbox command $SBX <cmd_id>          # poll status + output
lightning sandbox delete $SBX -y
```

**Run a containerized app and preview it on a public URL:**

```bash
SBX=$(lightning sandbox create --name preview --runtime python313-docker \
  --port 8080 --timeout 1800000 --json | jq -r .id)     # cpu-1 default: images go to disk

# --network=host is mandatory, and the container must LISTEN on 8080 (-p is ignored)
lightning sandbox run $SBX -- docker run -d --network=host --name web \
  -v /workspace:/srv -w /srv python:3.13-alpine python -m http.server 8080

lightning sandbox run $SBX -- curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8080
lightning sandbox list --json | jq -r ".sandboxes[] | select(.id==\"$SBX\") | .port_urls[\"8080\"]"
lightning sandbox delete $SBX -y
```

**Try out a custom image:**

```bash
lightning sandbox create --name img-test --image ghcr.io/myorg/myimage:latest \
  --teamspace my-org/my-teamspace --timeout 1800000
# private registry: --image-secret-ref <docker-registry-secret-name>
```

**Persistent devbox with snapshot-based branching:**

```python
sb = Sandbox.create(name="devbox", teamspace="my-org/my-teamspace", runtime="python313", persistent=True)
sb.run_command("pip install torch numpy pandas")     # slow setup, do once
snap = sb.snapshot()                                 # golden image
sb.stop()                                            # pause billing; sb.resume() later, same id
fresh = Sandbox.create(name="experiment-1", snapshot_id=snap.id,
                       teamspace="my-org/my-teamspace")   # warm clone with deps preinstalled
```

## Raw API fallback

```bash
lightning api /v1/core/sandboxes -X GET -f "organizationId=${ORG_ID}" -f "projectId=${PROJECT_ID}" -f limit=20
lightning api /v1/core/sandboxes -X GET ... | jq -r '.sandboxes[] | .name // .id'
lightning api "/v1/core/sandboxes/${SANDBOX_ID}" -X GET -f "organizationId=${ORG_ID}"   # -X GET: fields without it send a POST
lightning api "/v1/core/sandboxes/${SANDBOX_ID}/commands" -X POST -f command=ls -F detached=false
```

`ORG_ID` from memberships: `lightning api /v1/memberships | jq -r '[.memberships[] | select(.ownerType == "organization") | .ownerId][0]'`.

## Gotchas

- **Timeout units differ.** `create --timeout`, `extend_timeout()` and snapshot `--expiration` take milliseconds. `sandbox run --timeout` (detached wait) takes seconds.
- **Sandboxes bill until stopped or deleted, and garbage collection never removes them.** Always `stop()`/`delete()`, and set a create-time `timeout` as a safety net.
- **Without `-y`, `sandbox delete` and `sandbox snapshot delete` exit without deleting.** Non-interactively they read the prompt from a closed stdin, print `Are you sure you want to delete? [y/N]: Aborted.` and leave the sandbox billing. This is the easiest way to leak money here.
- **There is no cost surface for sandboxes, so budget with a create-time `timeout`.** `/v1/billing/usage`, `/v1/projects/<pid>/usage` and `/v1/core/sandboxes/<id>/usage` all 404. No CLI reports spend, and sandbox instance types (`cpu-1`, `cpu-2`) are missing from the priced accelerator catalog and from `lightning machine list`, so a run cannot be priced even by hand.
- **Egress policy is SDK-only and fixed at create.** `sandbox create` has no network-policy flag, so `deny-all` and CIDR allowlists need `lightning_sdk`. Everything else here is CLI-doable. You cannot change egress rules on a running sandbox. The default is open egress (`allow-all`); use `"deny-all"` or a CIDR allowlist for untrusted code.
- **A CIDR allowlist does not permit DNS by itself.** `allow_cidrs` is enforced at the IP layer, and `/etc/resolv.conf` points at `1.1.1.1` and `8.8.8.8`. So every hostname lookup fails with `Temporary failure in name resolution` while raw-IP connections work. Add `1.1.1.1/32` and `8.8.8.8/32` to the allowlist, or point the sandbox at a resolver inside it.
- **Files restored from a snapshot come back with mtime `1970-01-01T00:00:00Z`.** Epoch-zero timestamps break `make`, `ccache`, and pip/setuptools staleness checks. Touch files you depend on, or avoid mtime-based staleness logic in a restored sandbox.
- **Commands run as root inside the sandbox.**
- **Scoped-key errors come back as hints from the CLI and SDK.** `Use a teamspace- or org-scoped API key ...` (raw: "organization_id is required") → you're on a personal login key. `Your teamspace-scoped API key is not authorized for the project requested via teamspace= ...` (raw: "API key is not authorized for this project") → the key is bound to a different teamspace. `This operation requires a teamspace-scoped API key ...` → snapshot/stop with an org-scoped key. Only raw `lightning api` calls show the raw wording.
