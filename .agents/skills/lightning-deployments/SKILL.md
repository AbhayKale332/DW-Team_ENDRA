---
name: lightning-deployments
description: Deploy and operate services on Lightning AI - run Docker containers or HuggingFace models (vLLM) behind autoscaled HTTPS endpoints, manage replicas, releases, endpoint auth, and logs. Use when the user wants to deploy an API, model server, or container to lightning.ai, or manage an existing deployment.
---

# Lightning AI Deployments

A Deployment runs a container (or an auto-built vLLM server for a HuggingFace model) behind an HTTPS endpoint with autoscaling. Sources are mutually exclusive: `--image` (Docker), `--model` (HuggingFace id, GPU required), or `--studio` (deploy a Studio's environment).

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

## Resolving org and teamspace (do this first)

Deployments live in a teamspace owned by an organization or a user. **Never guess.** Use explicit `--teamspace owner/teamspace` (Python: `teamspace="owner/teamspace"`; the separate `org=`/`user=` arguments are deprecated), env vars `LIGHTNING_ORG` / `LIGHTNING_TEAMSPACE`, or the config default (`lightning config get teamspace`). If none is set, list the options and **ask the user which org/teamspace to use**:

```bash
lightning api /v1/memberships | jq -r '.memberships[] | [.ownerType, .name, .projectId] | @tsv'
```

Persist the choice: `lightning config set teamspace <owner>/<teamspace>`.

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

Subcommands: `create`, `list`, `inspect`, `update`, `delete`, `logs` (+ `logs download`), `reload-weights`, and `env` / `secret` (`list`/`set`/`delete` on a running deployment). There is no `stop`: scale to zero with `update --min-replicas 0 --max-replicas 0`.

```bash
# container deployment (a --port is required for non-model deployments)
lightning deployment create my-api --teamspace owner/teamspace \
  --image nginx:latest --machine CPU --port 80 --replicas 1 --api-key-auth

# HuggingFace model via vLLM (GPU machine required; port defaults to 8000)
lightning deployment create llama --teamspace owner/teamspace \
  --model meta-llama/Llama-3.1-8B-Instruct --machine L40S \
  [--hf-token-secret <secret-name>] [--tensor-parallel-size 2] [--max-model-len 8192] \
  [--quantization fp8] [--dtype bfloat16] [--dry-run] [--force]

# cloud account — if you pass none, the SDK picks one (config `cloud-account`, then
# LIGHTNING_CLUSTER_ID, then the teamspace's default binding) and prints which on stderr.
# Other deployments in the same teamspace may sit on a different account, which
# `deployment list` shows in its last column; match it with --cloud when relevant.
lightning deployment create ... --cloud gcp-lightning-public-prod

# autoscaling / env / secrets
lightning deployment create ... \
  --min-replicas 0 --max-replicas 4 --autoscale-metric GPU --autoscale-threshold 90 \
  -e KEY=VALUE --secret MY_LIGHTNING_SECRET --interruptible   # check the spot price first (below)

# endpoint auth: mutually exclusive; OMITTING ALL THREE MAKES THE ENDPOINT PUBLIC (confirm that's intended)
  --api-key-auth                    # require a Lightning USER API key (Bearer)
  --basic-auth USER:PASS
  --token-auth TOKEN                # any bearer string you choose

# operate
lightning deployment list --teamspace owner/teamspace [--all] [--sort-by state]
lightning deployment inspect my-api --teamspace owner/teamspace     # full JSON incl. endpoint URLs
lightning deployment logs my-api --teamspace owner/teamspace [-f] [--tail 100] [--timestamps]
lightning deployment logs my-api --teamspace owner/teamspace --query timeout --severity error   # filter server-side
lightning deployment logs my-api --teamspace owner/teamspace --since 30m --until 5m             # window (replicas merged + labelled)
lightning deployment update my-api --teamspace owner/teamspace --max-replicas 8
lightning deployment update my-api --teamspace owner/teamspace --min-replicas 0 --max-replicas 0   # "stop"
lightning deployment update my-api --teamspace owner/teamspace --image myimg:v2 [--max-surge 1] [--max-unavailable 0]  # new release, rolling update
lightning deployment reload-weights llama --teamspace owner/teamspace   # hot-reload BYOM weights
lightning deployment delete my-api --teamspace owner/teamspace --yes
```

- **Pick the auth flag from `lightning auth whoami`: `--api-key-auth` for `user` callers, `--token-auth` for `scoped-api-key` callers.** `--api-key-auth` only accepts a Lightning *user* key (it serializes to `userApiKey: true`). A scoped-key caller gets **401 on every request** while replicas run and seconds are billed. Use `--token-auth <token>` when a scoped key, CI job or agent will call the endpoint. Use `--api-key-auth` when humans with their own Lightning logins will.
- **`--interruptible` is not reliably cheaper, so check the price before you pass it.** On some SKUs and clouds spot costs more than on-demand, and nothing warns you at create time. Fetch both rates from the accelerator catalog (see the `lightning-cost-estimation` skill) and take `min(cost, spotPrice)`.
- **Changing image, command, env, entrypoint, path mappings, health check, cloud account, spot, quantity or max runtime creates a new release; switching the machine alone does not.** The CLI adds a rolling update automatically. The SDK raises `RuntimeError` if the deployment has no release strategy, neither passed nor already stored.

## Python SDK

```python
from lightning_sdk import Deployment, Machine
from lightning_sdk.api.deployment_api import (
    ApiKeyAuth, BasicAuth, TokenAuth, AutoScaleConfig, Env, Secret,
    RollingUpdateReleaseStrategy, HttpHealthCheck,
)

dep = Deployment("my-api", teamspace="my-org/my-teamspace")   # owner/teamspace, same as the CLI
dep.start(
    image="nginx:latest",
    machine=Machine.CPU,
    ports=[80],                                 # REQUIRED for non-model deployments; a LIST, not an int
    autoscale=AutoScaleConfig(min_replicas=0, max_replicas=4, metric="CPU", threshold=90),
    env=[Env("MODE", "prod"), Secret("HF_TOKEN")],   # or a plain dict for env vars
    auth=ApiKeyAuth(),                          # None => PUBLIC endpoint
    health_check=HttpHealthCheck(path="/health", port=80),
    spot=False,                                 # True = interruptible replicas
)

print(dep.urls)                                 # endpoint URL(s); no .status property — use replica counts
print(dep.running_replicas, dep.pending_replicas, dep.failing_replicas)
r = dep.get(path="/health")                     # convenience HTTP; auto-auths only for ApiKeyAuth

# a new release needs a release strategy (see the release rule under CLI reference)
dep.update(image="nginx:1.27", release_strategy=RollingUpdateReleaseStrategy(max_surge=1, max_unavailable=0))
dep.update(min_replicas=1, max_replicas=8)      # scaling: no new release, no strategy needed
dep.stop()                                      # scales to 0; blocks until replicas reach 0
```

HuggingFace model serving via SDK: `dep.start(model="meta-llama/...", machine=Machine.L40S, ports=[8000], hf_token_secret="...")`. `image`/`studio` must be None; still pass `ports`.

**`dep.delete()` does NOT delete the deployment.** Like `dep.get()`/`dep.post()`, it is an HTTP helper: it sends an HTTP `DELETE` request to the deployed service's endpoint. To delete the deployment resource use `lightning deployment delete NAME --yes` or the raw API.

### Private container images

A deployment whose image needs registry credentials must set `V1JobSpec.image_secret_ref` to the name of a `SECRET_TYPE_DOCKER_REGISTRY` secret in the teamspace. **Neither the CLI nor `Deployment.start()` exposes `image_secret_ref`, and without it the server rejects the image with an opaque `500`.** The error names neither the image nor the missing credential:
`Exception: The jobs_service_create_deployment_with_http_info request failed to reach the server, response: 500.`

Until the SDK exposes it, patch the field onto the create request:

```python
from lightning_sdk.lightning_cloud.openapi.api import jobs_service_api

_create = jobs_service_api.JobsServiceApi.jobs_service_create_deployment_with_http_info

def _with_image_secret(self, body, project_id, **kwargs):
    if getattr(body, "spec", None) is not None:
        body.spec.image_secret_ref = "MY_REGISTRY_SECRET"
    return _create(self, body, project_id, **kwargs)

jobs_service_api.JobsServiceApi.jobs_service_create_deployment_with_http_info = _with_image_secret
```

List the teamspace's secrets, and their types, with:

```bash
PROJECT_ID=$(lightning api /v1/memberships | jq -r '.memberships[] | select(.name=="<teamspace>") | .projectId')
lightning api "/v1/projects/${PROJECT_ID}/secrets" | jq -r '.secrets[] | "\(.name) \(.type)"'
```

**To diagnose this 500, retry with a public image (`nginx:latest`) and otherwise identical arguments.** If that create succeeds, the image is the variable, not the machine, port, replica counts or cloud account.

## Example workflows

Prompts this skill handles: *"deploy this docker image behind an API"*, *"serve Llama 3.1 8B on lightning"*, *"scale my deployment down at night"*, *"roll out the new image version"*.

**Deploy a container, verify it responds, then clean up.** Verify against the endpoint URL from `deployment inspect`, never the app inside the source Studio. The Studio proves the image; only the endpoint proves the service.

```bash
lightning auth whoami    # pick the auth flag from this (see the auth rule under CLI reference)
lightning deployment create hello-api --teamspace my-org/my-teamspace \
  --image nginx:latest --machine CPU --port 80 --min-replicas 0 --max-replicas 1 --api-key-auth
lightning deployment inspect hello-api --teamspace my-org/my-teamspace   # JSON: status, endpoint URL
URL=<endpoint URL from the inspect output above>
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $LIGHTNING_API_KEY" "$URL/"   # verify the ENDPOINT, not the Studio
```
```python
from lightning_sdk import Deployment
dep = Deployment("hello-api", teamspace="my-org/my-teamspace")
print(dep.urls)                       # e.g. ['https://80-dep-<id>-d.cloudspaces.litng.ai'] — available even while scaled to zero
print(dep.get(path="/").status_code)  # sends the caller's Lightning API key for ApiKeyAuth
```
```bash
lightning deployment delete hello-api --teamspace my-org/my-teamspace --yes
```

**Serve an open-weights LLM (vLLM, built server-side, no Dockerfile needed):**

```bash
lightning deployment create qwen-served --teamspace my-org/my-teamspace \
  --model Qwen/Qwen2.5-7B-Instruct --machine L40S --min-replicas 0 --max-replicas 1 --api-key-auth --dry-run
# review the model spec it prints, then re-run without --dry-run; if the real create reports
# unacknowledged warnings, re-run with --ack <code> (repeatable) or --force
lightning deployment logs qwen-served --teamspace my-org/my-teamspace -f   # watch it come up
```

The endpoint serves an OpenAI-compatible API on port 8000. Point any OpenAI client's `base_url` at `dep.urls[0]`.

**Operate: scale, update, roll back traffic costs:**

```bash
lightning deployment update hello-api --teamspace my-org/my-teamspace --min-replicas 1 --max-replicas 4   # keep warm
lightning deployment update hello-api --teamspace my-org/my-teamspace --min-replicas 0 --max-replicas 0   # stop (scale to zero)
lightning deployment update hello-api --teamspace my-org/my-teamspace --image nginx:1.27                  # new release, rolling by default
```

## Raw API fallback

```bash
PROJECT_ID=$(lightning api /v1/memberships | jq -r '.memberships[] | select(.name=="<teamspace>") | .projectId' | head -1)
lightning api "/v1/projects/${PROJECT_ID}/deployments" -q '.deployments[].name'   # plain GET; -f/-F without -X would send a POST
lightning api "/v1/projects/${PROJECT_ID}/deployments/${DEPLOYMENT_ID}"
lightning api "/v1/projects/${PROJECT_ID}/deployments/${DEPLOYMENT_ID}" -X DELETE
```

## Gotchas

- **Deployments bill for as long as a replica is up; confirm the spend with the user before creating one.** Get an explicit go-ahead before GPU machines, before `--min-replicas >= 1` (which bills continuously), and before raising `--max-replicas`, since the ceiling is what the bill can reach. Price the machine first with the `lightning-cost-estimation` skill.
- **Verify a deployment against its public URL from `deployment inspect`, never against the app inside the source Studio.** The Studio proves the image; only the endpoint proves the service. A deployment can show a healthy replica and bill normally while returning 401 to every request.
- **`min_replicas=0` stops billing at zero, and the next request waits for a cold start instead of failing.** The edge holds the connection open until a replica is ready (about 6 minutes on a GPU image), then returns a normal 200. It does not return 503 or refuse, so a client without a short timeout just waits. `min_replicas >= 1` bills continuously: flag this cost to the user.
- **The scale-to-zero idle window and the health check are SDK-only.** `AutoScaleConfig(idle_threshold_seconds=...)` and `HttpHealthCheck(path=..., port=...)` have no `deployment create`/`update` flag (no `--health-check-path`/`--readiness-probe`). From the CLI you get the default idle window, and nothing stops traffic reaching a replica that is up but not ready.
- **`Deployment.start()` on an existing deployment silently becomes an update/restart.** It won't error on a name collision.
- **A bad port fails without naming the port.** A non-model deployment with no port raises `ValueError`. In the SDK, an int `ports` fails with `TypeError: 'int' object is not iterable` from inside the SDK, which does not mention `ports`.
- **`AutoScaleConfig` without `metric` fails later, at `start()`.** It raises `ValueError: The autoscaling metric is required. Currently supported metrics are ['GPU', 'CPU', 'RPM']`, and the traceback points at `start()`, not at the config object. Always pass `metric=`.
- **`Secret("NAME")` showing an empty `name` in `deployment inspect` is not a bug.** It appears as `{"from_secret": "NAME", "name": "", "value": ""}`; the platform resolves the reference and names the environment variable after the secret.
- **Deployment logs are readable by anyone with teamspace access, so never echo real secrets on boot.** A container that prints its environment leaks secret values into `deployment logs`.
- **`Deployment deleted` is not "gone": the deployment can stay listed for up to ~2.5 minutes.** It may show in `deployment list` (and `list --all`) in `PENDING` with a stale replica count and `deletedAt: None`. Don't treat that as a failed delete or poll `list` to confirm teardown; re-issue `delete` only if it is still there after a few minutes. Deleting is destructive and the CLI prompts unless `--yes`, so confirm with the user first.
- **`deployment logs` without `-f` can hang with no output and never return.** Give a scripted bounded read (`--tail 25`, no `--follow`) its own timeout. `-f` works.
- **`total_cost` in `deployment inspect` stays `0.0` for the deployment's whole life, so measure spend from the teamspace credit balance.** Unlike jobs, no billing/usage endpoint resolves deployment cost either. Difference the credit balance before and after. That field flips between full float precision and 2-decimal rounding between consecutive calls, so use a window long enough that the rounding is noise.
- **`--dry-run` never calls the server, so it never shows the unacknowledged-warnings error.** A real `--model` create can fail with `Deployment has unacknowledged warnings: ... Re-run with --ack <code> (repeatable) or --force.` The dry run only echoes the model spec you set (`served_model_name`, `weight_source` and any vLLM flags you passed), not the machine, image variant or replica config.
- **An opaque `403` on `--model` means the feature is gated on your account; fall back to `--image`.** The error is `Exception: The jobs_service_create_deployment_with_http_info request failed to reach the server, response: 403.` with no server message, and `LIGHTNING_DEBUG=1` adds a traceback but still no reason. It is an entitlement, not a bad argument, so don't debug the model id or flags. Deploy a vLLM container image directly (`--image`), which needs no entitlement.
- **Passing both `teamspace="owner/teamspace"` and `org=`/`user=` raises `ValueError`.** `org=`/`user=` alone emits a `DeprecationWarning`. On SDKs older than 2026.7.31 the combined form failed with "Teamspace owner/name does not exist"; upgrade rather than splitting it.
