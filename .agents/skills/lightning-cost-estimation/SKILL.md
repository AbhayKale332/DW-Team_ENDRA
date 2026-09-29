---
name: lightning-cost-estimation
description: Estimate and compare what a workload costs on Lightning AI - fetch live per-hour GPU/CPU machine prices from the accelerator catalog for every cloud (Lightning Cloud, AWS, GCP, Lambda Labs, Nebius, Voltage Park), size hardware for a model, then quote a training run, fine-tune, serving deployment or data-prep job over hours to months, including spot rates, multi-node fan-out and Drive storage. Use when the user asks "how much would it cost to train/serve X", "what's the price of an H100/B200 on lightning", "how much for 32 8xH100 nodes for 2 weeks", "which cloud is cheapest for this", "what will my monthly bill be", or wants hardware sized for a model before pricing it.
---

# Lightning AI cost estimation

Every machine price on [lightning.ai/pricing](https://lightning.ai/pricing) comes from one public
endpoint. **Always fetch prices, capacity and SKU slugs live; never quote them from memory.** They
all change, and this file deliberately carries no price snapshot.

**1 Lightning credit = $1 USD.** Machines bill **per second** while allocated; storage bills daily.
Docs: [billing FAQ](https://lightning.ai/docs/platform/overview/faq/billing.md) ·
[manage costs](https://lightning.ai/docs/platform/team-management/organizations/manage-costs.md).

## Setup & auth

**The accelerator catalog needs no auth**, so plain `curl` works before login:

```bash
curl -s "https://lightning.ai/v1/core/accelerators?cloudProvider=MACHINE" | jq '.accelerator | length'
```

`lightning api "<path>"` returns the same JSON; use it when an agent's permission rules block `curl`
but allow `lightning`. It always sends credentials, so it needs `lightning login` or an API key.
Once signed in, every `curl` recipe below works with `lightning api "<path>"` in its place:

```bash
# Use the Lightning AI CLI from the current env; install or upgrade it there if it's missing or older than 2026.9.18
v=$(lightning --version 2>/dev/null | sed -n 's/^Lightning CLI version //p')
[ -n "$v" ] && [ "$(printf '%s\n' 2026.9.18 "$v" | sort -V | head -1)" = 2026.9.18 ] \
  || uv pip install -U lightning-sdk || python3 -m pip install -U lightning-sdk
lightning --version   # must print "Lightning CLI version …"; if not, see the table below
lightning api "/v1/core/accelerators?cloudProvider=MACHINE"
```

Auth is only needed to check which clouds a teamspace can reach (`lightning login`, or
`LIGHTNING_API_KEY`, optionally with `LIGHTNING_USER_ID`). Every recipe here needs `jq`.

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

## Resolving org and teamspace (only for availability checks)

Pricing is global: you **don't** need a teamspace to quote. You need one to check which cloud
accounts the user can launch on. **Never guess it**; resolve it, and ask if it's ambiguous:

```bash
lightning api /v1/memberships | jq -r '.memberships[] | [.ownerType, .name, .projectId] | @tsv'

PID=$(lightning api /v1/memberships | jq -r '.memberships[] | select(.name=="<teamspace>") | .projectId' | head -1)
lightning api "/v1/projects/$PID/clusters" | jq -r '.clusters[].id'
```

Map the cluster ids to catalog providers with the *Cloud providers* table. Don't quote a cloud the
teamspace can't launch on as the answer; mention it only as "available if you enable that cloud
account".

## The accelerator catalog

`GET https://lightning.ai/v1/core/accelerators?cloudProvider=<PROVIDER>` → `{"accelerator": [ ... ]}`.
The fields that matter for costing:

| Field | Meaning |
|---|---|
| `cost` | **On-demand USD per hour for the whole instance, not per GPU.** `36` on an 8×H100 SKU is $4.50/GPU-hr; quoting it per GPU inflates an estimate 8×. |
| `spotPrice` | Interruptible USD per hour. **`0` means the provider has no spot**, not free. Spot is **not always cheaper** than `cost`: take `min(cost, spotPrice)` over the non-zero values and say which you quoted. |
| `family`, `resources.gpu`, `resources.gpuType` | The hardware: `H100` × `8` × `nvidia-h100-80gb`. **Compare clouds on these, not on slugs** (see below). `gpuType` sometimes omits VRAM (`nvidia-b200`); use the VRAM table. The SDK spells the RTX family `RTX PRO`, the catalog `RTXP`. |
| `slugMultiCloud` / `slug` / `instanceId` | SKU names. They differ between clouds for the same hardware. |
| `resources.memoryMb`, `.cpu`, `.storageGb` | Host RAM / vCPU / disk. Occasionally wrong (one AWS H200 SKU reported `141000000` MB); sanity-check RAM before quoting it. |
| `outOfCapacity`, `availableInSeconds` (`…Spot`) | Whether you can get it now, and the expected wait. **A cheap SKU that is out of capacity, or waits 10⁴–10⁶ s, is not a real quote.** |
| `dwsOnly` / `dwsCost` | GCP Dynamic Workload Scheduler: queued batch allocation, not on-demand, so not for an always-on endpoint. |
| `reservable` / `capacityBlockPrice` | AWS capacity blocks: a much cheaper reserved rate for long runs. The discount varies widely by SKU, so read it per SKU and surface it for runs measured in weeks. |
| `isTierRestricted` | Needs a paid plan tier; a free-tier account can't launch it at any price. |
| `enabled`, `availableZones`, `clusterId` | Whether it's live, where, and which cloud account it belongs to. |

### Cloud providers

| `cloudProvider=` | What it is | Cluster id (for `--cloud`) | Notes |
|---|---|---|---|
| `MACHINE` | **Lightning Cloud** (Lightning's own bare metal) | `lightning-baremetal` | The default cloud. Not the SDK's `LIGHTNING`, which is a placeholder here. |
| `AWS` | AWS | `lightning-public-prod` | Widest SKU range; capacity blocks. |
| `GCP` | GCP | `gcp-lightning-public-prod` | Only source of TPUs; some GPUs `dwsOnly`. |
| `LAMBDA_LABS` | Lambda Labs | `lightning-lambda-prod` | |
| `NEBIUS` | Nebius | `lightning-nebius-prod` | |
| `VOLTAGE_PARK` | Voltage Park | `lightning-voltagepark-prod` | |
| `VULTR` | Vultr | — | Niche SKUs (A16/A40/A100/L40S). |

Quote Lightning Cloud first, then compare the others live, and recommend another cloud only when it
wins on price **and** is in capacity for the user's teamspace.

### From a quote to a launch

Launch with the `--machine` name (`Machine.<NAME>` in Python) used by `lightning-jobs`,
`lightning-studios` and `lightning-deployments`, plus `--cloud <cluster-id>` for the cloud you
priced; without `--cloud` the run lands on the teamspace default and may bill a different rate.

| `--machine` | Catalog hardware | | `--machine` | Catalog hardware |
|---|---|---|---|---|
| `CPU_SMALL` / `CPU` / `CPU_X_8` / `CPU_X_16` | `cpu-2` / `cpu-4` / `cpu-8` / `cpu-16` | | `A100` … `A100_X_8` | `A100` × N, 40 or 80 GB by cloud |
| `DATA_PREP` / `_MAX` / `_ULTRA` | `data-prep-mid` / `-max-large` / `-ultra-extra-large` | | `H100` … `H100_X_8` | `H100` × N |
| `T4` / `L4` / `L40S` (`_X_N`) | `T4` / `L4` / `L40S` × N | | `H200` / `B200` (`_X_N`) | `H200` / `B200` × N |
| `T4_SMALL` | small T4 | | `RTXP_6000` (`_X_N`) | `RTXP` × N |

`A100_40GB*` / `A100_80GB*` pin the memory size but are hidden from CLI `--machine`; they work only
as `Machine.A100_80GB_X_8` in Python or with `studio switch`.

**Slugs differ between clouds for the same hardware, so never hardcode or join on one.** H200 is
`lit-h200-1` on Nebius and `lit-h200-141gb-1` on Lightning Cloud, and `Machine.H100.slug` is
`lit-h100-1` while the catalog says `lit-h100-80gb-1`. The SDK (2026.9.18+) maps all of these to the
same machine. Match on `family` + `resources.gpu` (+ VRAM from `gpuType`), as the recipes below do;
filtering on an assumed slug makes a machine that exists look unavailable.

## Price lookup recipes

Load these into the shell once (bash or zsh), then use them for every quote.

```bash
# _litfetch <PROVIDER> — catalog JSON, or an error and a non-zero exit
_litfetch() {
  curl -sf --retry 3 --retry-delay 1 --max-time 20 \
    "https://lightning.ai/v1/core/accelerators?cloudProvider=$1" \
  || { echo "ERROR: could not fetch the $1 catalog; don't quote from a partial result" >&2; return 1; }
}
_LITJQ='def spot: if .spotPrice > 0 then (.spotPrice*100|round/100) else "n/a" end;
  def count: if .acceleratorType=="GPU" then .resources.gpu else .resources.cpu end;
  def status: [ (if .outOfCapacity then "OUT-OF-CAPACITY" else empty end),
                (if .dwsOnly then "DWS-ONLY" else empty end),
                (if .isTierRestricted then "TIER-RESTRICTED" else empty end),
                (if .reservable then "RESERVABLE" else empty end) ]
              | if length==0 then "ok" else join(",") end;'

# litprice <PROVIDER> [family-regex] — one cloud's live catalog, one row per SKU
litprice() {
  local json; json=$(_litfetch "${1:-MACHINE}") || return 1
  printf '%s' "$json" | jq -r --arg re "${2:-.}" "$_LITJQ"'
      ["SKU","FAMILY","N","USD/HR","SPOT/HR","RAM_GB","WAIT_S","STATUS"],
      (.accelerator[] | select(.family|test($re;"i"))
       | [.slugMultiCloud, .family, count, .cost, spot,
          ((.resources.memoryMb|tonumber)/1024|round), .availableInSeconds, status])
      | @tsv' | column -t -s $'\t'
}

# litcompare <family> <gpus> [gpuType-regex] — the same hardware on every cloud, cheapest first.
# Stops if any cloud's catalog can't be fetched: a missing cloud could hide the real winner.
litcompare() {
  local p json rows=""
  for p in MACHINE AWS GCP LAMBDA_LABS NEBIUS VOLTAGE_PARK VULTR; do
    json=$(_litfetch "$p") || return 1
    rows+=$(printf '%s' "$json" | jq -r --arg p "$p" --arg f "$1" --argjson n "$2" --arg m "${3:-.}" "$_LITJQ"'
        .accelerator[]
        | select((.family|ascii_upcase)==($f|ascii_upcase) and .resources.gpu==$n
                 and ((.resources.gpuType // "")|test($m;"i")))
        | [$p, .slugMultiCloud, (.resources.gpuType // "?"), .cost, spot, .availableInSeconds, status]
        | @tsv')$'\n' || return 1
  done
  { printf 'CLOUD\tSLUG\tGPU\tUSD/HR\tSPOT/HR\tWAIT_S\tSTATUS\n'
    printf '%s' "$rows" | grep . | sort -t $'\t' -k4,4 -g; } | column -t -s $'\t'
}

# litquote <usd_per_hour> <instances> <hours> [storage_gb] [storage_days]
litquote() {
  jq -n --argjson r "$1" --argjson n "$2" --argjson h "$3" \
        --argjson gb "${4:-0}" --argjson d "${5:-0}" '
    ($r*$n*$h) as $c | (([$gb-10,0]|max) * 0.10 * ($d/30)) as $s
    | { rate_per_hour: $r, instances: $n, hours: $h,
        cluster_per_hour: ($r*$n*100|round/100),
        compute_usd: ($c*100|round/100),
        storage_usd:  ($s*100|round/100),
        total_usd:   (($c+$s)*100|round/100) }'
}
```

```bash
litprice MACHINE                 # everything on Lightning Cloud
litprice GCP 'B200|H200'         # just those families on GCP
litcompare H100 8                # every cloud's 8×H100, cheapest first
litcompare A100 1 80gb           # 1×A100, 80 GB variants only
litquote 36 32 336               # 32 nodes × a $36/hr rate × 2 weeks
```

## Cost formulas

Durations: 1 day = 24 h · 1 week = 168 h · 1 month = 730 h (or `days × 24`) · 1 year = 8760 h.

```
compute_usd  = cost_per_hour × instances × hours          # billed per second, no rounding up
spot_usd     = spotPrice     × instances × hours          # only if spotPrice > 0
storage_usd  = max(0, drive_gb − 10) × $0.10 × months     # first 10 GB free, billed daily
total_usd    = compute_usd + storage_usd
```

- **Per-GPU-hour** = `cost / resources.gpu`.
- **Multi-node**: per-node `cost` × `num_machines`. `--num-machines 32` on `H100_X_8` bills 32 × the
  8×H100 rate.
- **Serving**: `cost` × replicas × hours. With autoscaling, quote a **range**: `min_replicas` is the
  floor you pay 24/7 (0 with scale-to-zero), `max_replicas` the ceiling. Per-token economics:

  ```
  usd_per_1M_tokens = (cost_per_hour × replicas) / (tokens_per_sec × 3600 / 1e6)
  ```

  `tokens_per_sec` is the *measured* aggregate output throughput at your batch size: benchmark it or
  give a range. If the user only wants to *call* a model, compare against the per-token models in
  `lightning-llm-gateway`; a dedicated GPU only wins above a fairly high steady load.
- **Spot / interruptible** (`--interruptible`, `interruptible=True`) is typically 15–60% off but can
  be preempted: recommend it for checkpointed training, never for a serving SLA.
- **Billing covers the whole allocation window**, `startedAt` → `stoppedAt`, including the image pull
  (during which the SDK reports `Pending`) and shutdown. A 20-minute job on a large image billed 28
  minutes. Quote startup as its own line, or give a range; the shorter the job, the bigger the gap.
- **Storage**: count teamspace **folders**, not just the Drive. Data *added* as a folder bills like
  Drive; a bucket *connected* from outside is billed by its own cloud. Both show up under
  `/v1/projects/<pid>/data-connections`, so go by `isBillableFolder: true`. `Teamspace.new_folder()`
  creates the billed kind. For multi-week training, ask about checkpoint and dataset size: 20 TB is
  ~$2,000/month.
- Every account gets **one free 4-CPU Studio**; other machines bill at list rate.

State the assumptions with every number: rate, instances, hours, spot or on-demand, storage, and the
date you fetched prices.

## Sizing hardware for a model (do this before pricing)

When the user names a model instead of a machine, size it first, state the assumptions, then price
it. VRAM per GPU (the catalog doesn't always carry it):

| Family | VRAM/GPU | Family | VRAM/GPU |
|---|---|---|---|
| T4 | 16 GB | H100 | 80 GB |
| L4 | 24 GB | H200 | 141 GB |
| A100 | 40 or 80 GB (check `gpuType`) | B200 | 180 GB |
| L40S | 48 GB | RTXP (RTX PRO 6000) | 96 GB |
| A40 | 48 GB | TPU v6e | 32 GB HBM/chip |

**Inference / serving memory**

```
weights_gb  ≈ params_B × bytes_per_param        # BF16/FP16 = 2, FP8 = 1, INT4 = 0.5
kv_cache_gb ≈ 2 × layers × kv_heads × head_dim × bytes × ctx_len × concurrency / 1e9
vram_needed ≈ weights_gb × 1.15 + kv_cache_gb   # ~15% for activations/fragmentation
```

Pick the smallest SKU where `vram_needed ≤ gpus × vram_per_gpu`, keeping the tensor-parallel degree a
power of 2. Run the whole formula before picking, and keep units straight: "80 GB" is the vendor
label, while `nvidia-smi` reports **81,559 MiB ≈ 85.5 decimal GB** per H100 (an H200 reads
143,771 MiB ≈ 150.8 GB against a "141 GB" label). A 70B model in BF16 is 140 GB of weights, or
`140 × 1.15 = 161 GB` before KV cache. That leaves only ~10 GB of the 171 GB on 2×H100: it fits at
modest context and concurrency, so size the KV cache before promising more, or take 1×B200. Total
memory is necessary, not sufficient: the model must also split across the cards at your
tensor-parallel degree.

**Training memory** = base weights + trainable state + activations:

| Method | Base weights | Trainable state (grads + AdamW) | 70B, before activations |
|---|---|---|---|
| Full fine-tune, mixed precision | BF16, 2 B/param, all trained | ~14–16 B/param (grads, FP32 master copy, two FP32 moments) | ~1.1–1.3 TB → multi-node with FSDP/ZeRO-3 |
| LoRA | Frozen, BF16 (2 B/param) | Adapters only, usually < 1% of params | ~140 GB → 2×H100 or 1×H200/B200 |
| QLoRA | Frozen and quantized to 4-bit (~0.5 B/param) | Adapters only | ~35–40 GB → one 80 GB GPU |

**Activations come on top in every case** and grow with batch size × sequence length (and vocabulary
size, for the logits); gradient checkpointing trades compute to shrink them. They can dominate a
small model: a 4B LoRA run at batch 8 × 768 tokens peaked at 64 GB. QLoRA is LoRA plus quantizing the
frozen base ([PEFT: quantization](https://huggingface.co/docs/peft/developer_guides/quantization)).

**Training time** (this is what turns into money):

```
flops       = 6 × params × tokens                       # fwd + bwd, dense transformer
gpu_hours   = flops / (peak_flops × mfu × 3600)
wall_hours  = gpu_hours / (num_gpus × scaling_efficiency)
cost        = wall_hours × nodes × node_cost_per_hour
```

Dense BF16 peak (no sparsity): A100 312 TFLOPS · H100/H200 989 TFLOPS · B200 ≈ 2250 TFLOPS ·
L40S 362 TFLOPS · L4 121 TFLOPS. Use **MFU 0.35–0.5** (0.4 is a fair default) and **scaling
efficiency ~0.9 at 8–16 nodes, ~0.8 at 32+**. MFU, dataloader stalls, restarts and scaling losses
move the real number by tens of percent: give a range and list the assumptions.

## Example workflows

Prompts this skill handles: *"how much to train on 32 8×H100 nodes for 2 weeks?"*, *"what would it
cost to serve a model on B200?"*, *"cheapest way to fine-tune Llama-70B"*, *"what's my monthly bill
if I keep 2 A100s and 5 TB of data?"*.

**A multi-node training quote, end to end.** The rate here is illustrative; use what `litcompare`
returns.

```bash
litcompare H100 8                   # pick the cheapest row that's "ok" and in the user's teamspace
litquote 36 32 336                  # 32 nodes, 2 weeks, if that row is $36/hr: $1,152/hr -> $387,072
litquote 36 32 336 20480 14         # + 20 TB of checkpoints kept for the fortnight
```

Report it with the assumptions: *"32 × 8×H100 on <cloud> at $36/node/hr = $1,152/hr; 2 weeks
(336 h) ≈ $387k on-demand, plus ~$955 for 20 TB of checkpoints. Prices fetched <date>."* Add the spot
figure only if `spotPrice > 0` and the job checkpoints, and the capacity-block figure if the row is
`RESERVABLE`.

The other prompts follow the same steps:
- **Serving on a GPU family Lightning Cloud may not carry:** `litcompare B200 1` and `litcompare B200 8`,
  then flag `DWS-ONLY` and `OUT-OF-CAPACITY` rows, and quote scale-to-zero as a range.
- **A fine-tune:** size it with the training-memory table (a 7B LoRA fits one 80 GB GPU), then
  `litcompare H100 1` and `litquote <rate> 1 <hours>`.
- **A steady monthly bill:** `litquote <rate> 2 730 5120 30` for two machines 24/7 plus 5 TB; pass
  fewer hours (8 h/day ≈ 243 h/month) if they don't run all day.

## Raw API fallback

```bash
# full record for one SKU (all capacity/quota/reservation fields)
curl -s "https://lightning.ai/v1/core/accelerators?cloudProvider=AWS" \
| jq '.accelerator[] | select(.family=="H100" and .resources.gpu==8)'

# AWS reserved capacity-block rate vs on-demand
curl -s "https://lightning.ai/v1/core/accelerators?cloudProvider=AWS" \
| jq -r '.accelerator[] | select(.reservable) | [.slugMultiCloud, .cost, .capacityBlockPrice] | @tsv'

# what a finished job actually cost (needs auth): the ground truth to check an estimate against
lightning job inspect <job-name> --teamspace <owner>/<teamspace>
```

`job.total_cost` gives the realized USD for a finished job (see `lightning-jobs`), once it settles
(see Gotchas). With auth, `Teamspace("owner/teamspace").list_machines(cloud_account=<id>)` returns
`Machine` objects with `cost`, `interruptible_cost`, `wait_time` and `provider` for machines in
capacity on that account. Take `<id>` from `Teamspace(...).cloud_account_objs[i].cluster_id`: the
display names in `.cloud_accounts` return an empty list, and with no argument the rows of several
accounts are merged without saying which is which. `lightning-jobs` has a per-account listing. For
cross-cloud comparisons and quotes before login, use the REST catalog.

## Gotchas

- **`Machine.<NAME>.cost` and `.interruptible_cost` are `None` for every constant**, although `.slug`
  is set, so the object looks hydrated. Prices come from the catalog or `list_machines()`.
- **`job.total_cost` is provisional when a job first reports terminal, and keeps climbing** for a few
  minutes, with no flag saying so; it can also read exactly `0.0` for about a minute. Poll until it
  is unchanged across two or three reads, and treat `0.0` on a just-finished job as "not computed
  yet".
- **Unknown `cloudProvider` values silently return the AWS catalog.** `AZURE`, `KUBERNETES` and
  `TENSORDOCK` echo AWS's SKUs byte for byte. Only the providers in the table have been checked;
  before quoting from another (the enum also lists `CUDO`, `MITHRIL`, `THUNDER_CAT`, `CLOUDFLARE`,
  `RAFAY`, `SLURM`), diff its response against AWS's.
- **Never price from `cloudProvider=LIGHTNING` or `DGX`.** They return one internal cluster entry
  with placeholder costs of `1` and `2`. The SDK's `CloudProvider.LIGHTNING` is the same platform as
  the catalog's `MACHINE`; to launch on it, pass `--cloud lightning-baremetal`.
- **Stored bytes aren't observable from the platform.** `currentStorageBytes` on `/v1/memberships`
  didn't move while a billable folder was filled and emptied. Size storage from what you know you
  uploaded.
- Confirm with the user before *launching* anything you priced; this skill only quotes.
