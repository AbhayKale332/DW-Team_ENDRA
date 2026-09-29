---
name: lightning-artifacts
description: Publish a local file (HTML report, PDF, image, dataset sample, build output) to Lightning AI and get a durable, public lightning.ai/artifacts/<id> link that never expires and renders inline in the browser. Also lists the artifacts drive, unpublishes (revokes) links and deletes the files behind them, all through the `lightning` CLI (`lightning-sdk` package) with `lightning login` or an API key, no code. Use when the user wants to share a file, a generated one-pager, or an agent-made artifact as a permanent URL, hand a file to a teammate or CI job, see or revoke existing shared links, or asks to "get a public / shareable link for this file".
---

# Lightning AI Artifacts (durable shareable file links)

Publish a file to a teamspace and get a `https://lightning.ai/artifacts/<id>`
URL that anyone can open without a Lightning login. The control plane streams
the bytes from storage on every request, so unlike a presigned S3 URL the link
never expires (no ~1h cap).

**The whole flow is CLI-only.** `lightning cp` / `ls` / `rm` handle the files,
and `lightning api` makes the publish and unpublish calls. No Python or `curl`.

## Setup & auth

```bash
# Use the Lightning AI CLI from the current env; install or upgrade it there if it's missing or older than 2026.9.18
v=$(lightning --version 2>/dev/null | sed -n 's/^Lightning CLI version //p')
[ -n "$v" ] && [ "$(printf '%s\n' 2026.9.18 "$v" | sort -V | head -1)" = 2026.9.18 ] \
  || uv pip install -U lightning-sdk || python3 -m pip install -U lightning-sdk
lightning --version   # must print "Lightning CLI version …"; if not, see the table below
lightning login                       # interactive browser sign-in — enough for everything here
# or: export LIGHTNING_API_KEY=...    # non-interactive alternative (CI, agents)
```

Either credential works for **every call in this skill** — no token minting or
extra auth steps. Get a key from lightning.ai → user/org settings, or
`lightning api-key create --org <org> --name artifacts`. **Never hardcode the
key** — read it from the environment. To target a non-prod control plane, set
`LIGHTNING_CLOUD_URL` (default `https://lightning.ai`).

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

`lightning api` flags: `-X` method, `-f key=val` string field, `-F key=val`
typed field, `-H` header, `--input <file>` request body (`--input -` to pipe
one), `-q` jq filter (needs the `jq` binary for `-q`), `-i` include response
headers, `--silent` no output. Fields with no `-X` make it a POST. Fields are JSON body for POST/PUT-with-body and **query
params** when the request also has `--input` or is a GET.

## Resolve the teamspace (do this first)

Artifacts live in a teamspace (a "project" in the REST API). You need the
**owner name** and **teamspace name** (for `lit://` paths) and the **project
id** (for REST calls). **Never guess.** If more than one membership fits and
none is configured, **ask the user which to use**:

```bash
lightning api /v1/memberships -q '.memberships[] | [.name, .projectId, .ownerType, .ownerId] | @tsv'
```

The membership carries only the owner's id. Resolve the name by `ownerType`:

```bash
PID=<projectId-from-above>
TSNAME=<name-from-above>
OWNER_ID=<ownerId-from-above>
if [ "<ownerType-from-above>" = organization ]; then
  OWNER=$(lightning api "/v1/orgs/$OWNER_ID" | jq -r .name)
else   # a personal teamspace: yours, or another user's you were added to
  OWNER=$(lightning api /v1/users/search -X GET -f "query=$OWNER_ID" \
    | jq -r --arg id "$OWNER_ID" '.users[] | select(.id==$id) | .username')
fi
[ -n "$OWNER" ] && [ "$OWNER" != null ] || echo "could not resolve the owner of $TSNAME — ask the user" >&2
```

The user search is fuzzy, so match `.id` exactly (the SDK does the same). Never
fall back to your own username: a wrong owner in the `lit://` path uploads into
a different teamspace.

Teamspaces reached through org-level permissions don't appear in
`/v1/memberships`. If the user names one you can't find, ask them for the
`<owner>/<teamspace>` pair and the project id. Don't try to look it up by name:
`/v1/projects?name=…` doesn't exist.

## Publish a durable link (the CLI flow)

**A public link (`private=false`, the default below) needs the user's say-so in this request.**
"Publish this" or "get me a public link" is authorization — don't ask twice. "Send this to the
team" is not: say what the link exposes and get a yes, or use `-F private=true`.

Three calls: upload into the `artifacts/` drive, read back the storage cluster
it landed on, then register the object as a shared artifact. `share` and
`shares` (below) need the `jq` binary; without it, parse the JSON yourself.

```bash
# share <local-file> [remote-name] [content-type]
# Pretty status goes to stderr; the bare URL is the only thing on stdout, so
# URL=$(share file.html) captures cleanly while interactive use looks like:
#   ✅ Published report.html (text/html; charset=utf-8)
#   🔗 https://lightning.ai/artifacts/art_...
share() {
  local FILE="$1" NAME="${2:-$(basename "$1")}" CT="${3:-$(file -b --mime-type "$1")}"
  local KEY="artifacts/${NAME#artifacts/}"     # publish only finds objects under artifacts/
  # 1. upload; the server picks where to store it (see Gotchas)
  lightning cp "$FILE" "lit://$OWNER/$TSNAME/$KEY" >&2 || return 1
  # 2. publish needs the blob's storage cluster, read from the listing (see Gotchas)
  local CLUSTER; CLUSTER=$(lightning ls --json "lit://$OWNER/$TSNAME/$KEY" | jq -r '.[0].clusterId')
  # 3. register it -> durable, no-expiry lightning.ai/artifacts/<id>
  local LINK; LINK=$(lightning api "/v1/projects/$PID/shared-artifacts" -X POST \
    -f clusterId="$CLUSTER" -f filename="$KEY" -f contentType="$CT" -F private=false -q .url | tr -d '"')
  echo "✅ Published $NAME ($CT)" >&2
  printf '🔗 ' >&2
  echo "$LINK"
}

share report.html                                  # -> https://lightning.ai/artifacts/art_...
share dashboard.html reports/dash.html text/html   # custom remote name + explicit type
```

**Always show the user the full URL on its own line**, never just "done".
Terminals make it clickable.

**Shares are confined to the `artifacts/` folder.** Publish only finds objects
under `lit://<owner>/<teamspace>/artifacts/...`. Files under `uploads/` or
`lightning_storage/` can be copied with `lightning cp`, but publish won't find
them. The `KEY` variable keeps the upload path and the publish `filename`
identical. A bare `filename` gets `artifacts/` prepended by the server, but
matching the two explicitly is clearest.

**The default `private=false` link is open to anyone who has it, with no
auth.** Set `-F private=true` so only authorized project readers can open it.

### Content types that render inline

**The publish call's `contentType` decides whether HTML/PDF render or
download.** It is set at publish time, not upload time, and overrides the
stored object's type on every request. `file -b --mime-type` guesses most
cases; pass an explicit type when the extension is ambiguous.

| File | Content-Type |
| --- | --- |
| `.html` | `text/html; charset=utf-8` |
| `.pdf` | `application/pdf` |
| `.svg` | `image/svg+xml` |
| `.png` / `.jpg` | `image/png` / `image/jpeg` |
| `.json` / `.txt` / `.csv` | `application/json` / `text/plain` / `text/csv` |
| anything to force-download | `application/octet-stream` |

## List, see what's published, unpublish, delete

**`lightning ls` lists the drive completely, even for folders with many
thousands of files.** It follows the server's pages. It shows one level by
default, `-r` for every file underneath, and `--json` for entries with `size`
and `clusterId`:

```bash
lightning ls "lit://$OWNER/$TSNAME/artifacts/"              # one level; folders end in /
lightning ls -r "lit://$OWNER/$TSNAME/artifacts/reports"    # full relative file paths
lightning ls --json "lit://$OWNER/$TSNAME/artifacts/reports"
```

**`shares` lists every published link in the project, newest first.** It calls
`GET /v1/projects/{pid}/shared-artifacts`, which returns id, filename, content
type, public/private, download count and URL:

```bash
# shares  ->  one block per published artifact
shares() {
  lightning api "/v1/projects/$PID/shared-artifacts" | jq -r '.artifacts // [] | .[] |
    "\(if .private then "🔒" else "🌐" end) \(.filename)  ·  ⬇ \(.downloads // 0)  ·  \(.createdAt // "")
   id: \(.id)
   🔗 \(.url)"'
}
```

**`unshare` revokes a link but keeps the file in the drive:**

```bash
# unshare <artifact-id>
unshare() {
  lightning api "/v1/projects/$PID/shared-artifacts/$1" -X DELETE --silent \
    && echo "🗑️  Unpublished $1 — link is dead, file kept in the drive" >&2
}

unshare art_01kxgaep54zzs84arfns1j21wd   # illustrative id; take yours from shares
```

**`lightning rm` deletes the file itself, with no cluster id.** The server
removes it wherever it is stored. Use it after unpublishing, or to clean up an
abandoned upload:

```bash
lightning rm "lit://$OWNER/$TSNAME/artifacts/report.html"
lightning rm -r "lit://$OWNER/$TSNAME/artifacts/reports"     # a folder and everything under it
```

`rm` fails on a path that doesn't exist (pass `-f` to ignore) and refuses a
folder without `-r`.

**Re-running `lightning cp` to the same `artifacts/<name>` updates the bytes
behind existing links.** Their ids keep serving the new contents. Repeating the
publish call with the same `filename` gives a new id and URL; do that to change
the served Content-Type.

## Example workflows

Prompts this skill handles: *"share this HTML report as a permanent link"*,
*"give me a public URL for output.pdf"*, *"publish this one-pager so I can send
it"*, *"drop this file somewhere CI can curl it"*.

**An agent-generated HTML one-pager needs only `share`; anyone can open the printed URL:**

```bash
share summary.html                                 # hand the printed URL to anyone
```

**A loop over `share` publishes a folder of reports, one link each:**

```bash
for f in out/*.html; do share "$f" "reports/$(basename "$f")" "text/html; charset=utf-8"; done
```

**`share` prints only the URL on stdout, so a CI job can capture and curl it:**

```bash
URL=$(share model-metrics.json)
# elsewhere:  curl -sL "$URL" -o metrics.json
```

## Gotchas

- **Publishing with a compute cluster's id fails with HTTP 500; publish needs
  the blob's storage cluster.** A teamspace can be bound to compute clusters
  that store files under a parent cluster's bucket, so the upload cluster is not
  always the storage cluster. Read the real `clusterId` from the listing, as
  `share` does.
- **`lightning cp` needs no cluster flag, because the server picks the
  storage.** Only if the server rejects the upload for a missing cluster does
  `cp` choose one itself (`LIGHTNING_CLUSTER_ID` first, which is set inside a
  Studio, then the teamspace's only or default cloud account) and warn `No
  cloud account specified. Using cloud account: <id>.` Pass `--cloud-account
  <id>` only to steer placement deliberately, and pick a cluster whose
  `status.phase` is `CLUSTER_STATE_RUNNING` (`/v1/projects/$PID/clusters`).
  A bound but unusable cluster makes the upload fail with a drive error that
  says nothing about cluster health.
- **Unpublish reports success for links that don't exist.**
  `DELETE /v1/projects/{pid}/shared-artifacts/{id}` returns `HTTP 200` with `{}`
  for an id that was already revoked or mistyped. Exit code 0 is not proof;
  check that the public URL 404s. (`lightning rm`, by contrast, fails on a
  missing path unless you pass `-f`.)
- **Served HTML is not byte-identical to the upload, so don't checksum it
  against the source.** Cloudflare injects a Browser-Insights RUM beacon
  (`static.cloudflareinsights.com/beacon.min.js`) into HTML responses. It is
  harmless for viewing, but don't promise a bit-exact document.
- **`shares` fails with HTTP 501 "Method Not Allowed" on control planes built
  before mid-July 2026.** The list endpoint `GET
  /v1/projects/{pid}/shared-artifacts` is newer than the rest. Publish and
  unpublish still work there.
- **A public link is open to anyone who has it, and revoking it doesn't recall
  a copy already fetched.** Use `-F private=true` for anything that shouldn't
  be world-readable.
