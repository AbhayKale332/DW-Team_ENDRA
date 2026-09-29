---
name: lightning-llm-gateway
description: Call hosted LLMs (OpenAI GPT, Anthropic Claude, Google Gemini, open-weights models) through the Lightning AI models API / LLM gateway - chat, streaming, multi-turn conversations, images, and model metadata; plus the teamspace model checkpoint registry. Use when the user wants to run inference through lightning.ai, compare gateway models, or manage model artifacts on the platform.
---

# Lightning AI LLM Gateway (Models API)

The LLM gateway gives one API + one bill for models from multiple providers. Model names are `provider/model` (see [Finding model names](#finding-model-names)). **Import `LLM` from `lightning_sdk.llm`; it is not exported at package top level.**

## Setup & auth

```bash
# Use the Lightning AI CLI from the current env; install or upgrade it there if it's missing or older than 2026.9.18
v=$(lightning --version 2>/dev/null | sed -n 's/^Lightning CLI version //p')
[ -n "$v" ] && [ "$(printf '%s\n' 2026.9.18 "$v" | sort -V | head -1)" = 2026.9.18 ] \
  || uv pip install -U lightning-sdk || python3 -m pip install -U lightning-sdk
lightning --version   # must print "Lightning CLI version …"; if not, see the table below
lightning login   # or headless: export LIGHTNING_USER_ID=... LIGHTNING_API_KEY=...
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

The `LLM` class is Python. The install above also makes `lightning_sdk` importable in that env,
so snippets run with its `python`. If the CLI came from a uv tool or `uvx` fallback instead, run
one-off scripts with `uv run --with lightning-sdk python script.py`. For code that stays in the
user's project, ask, then declare `lightning-sdk` as a dependency with the project's own tool
(`uv add`, `poetry add`).

Inference is **billed to a teamspace** — the `LLM` class refuses to run without one. Resolution: explicit `teamspace=` arg (`"owner/teamspace"`, or a bare org name to use that org's first teamspace) → `LIGHTNING_TEAMSPACE` + `LIGHTNING_CLOUD_PROJECT_ID` env (both set inside a Studio; the name alone raises `Teamspace ID is missing`) → the user's default teamspace. If the user belongs to multiple orgs/teamspaces and none is configured, **ask which one should be billed**:

```bash
lightning api /v1/memberships | jq -r '.memberships[] | [.ownerType, .name] | @tsv'
```

## Python SDK

```python
from lightning_sdk.llm import LLM

llm = LLM("openai/gpt-4o", teamspace="my-org/my-teamspace")

# single-shot
print(llm.chat("Summarize this...", system_prompt="Be terse", max_completion_tokens=500))

# streaming
for chunk in llm.chat("Tell a story", stream=True):
    print(chunk, end="")

# multi-turn: reuse the same conversation name across calls. History persists server-side
# under that name; set LIGHTNING_EPHEMERAL=true to persist nothing
llm.chat("What is CUDA?", conversation="cuda-help")
llm.chat("Show an example", conversation="cuda-help")
print(llm.get_history("cuda-help"))        # [{"role": "user"|"assistant", "content": ...}]
llm.reset_conversation("cuda-help")
llm.list_conversations()

# images (local paths are base64-encoded automatically; URLs passed through)
llm.chat("What's in this image?", images=["./photo.png"])

# knobs
llm.chat("...", reasoning_effort="high")   # none|low|medium|high
# there is no temperature/top_p: extra **kwargs are silently dropped, not sent to the model

# model info & pricing: every chat() call is billed to the resolved teamspace
print(llm.context_length)
m = llm.metadata                            # prompt_price, completion_price, max_completion_tokens,
                                            # capabilities, throughput, time_to_first_token
```

Async: `LLM("...", enable_async=True)` makes `chat()` awaitable (async generator when streaming). The async path silently ignores `tools` and `reasoning_effort`.

### Finding model names

**There is no `llm.list_models()`; get the current catalog from the lightning.ai Model APIs page.** The SDK also keeps a static map of known public models in `lightning_sdk/llm/public_assistants.py`. It is used only when `LIGHTNING_CLOUD_URL` is `https://lightning.ai` (in practice, inside Studios). Elsewhere the SDK asks the server, so models outside the map can still resolve.

Illustrative names (the set changes): `openai/gpt-4o`, `anthropic/claude-3-5-sonnet-20240620`, `google/gemini-2.5-pro`, `lightning-ai/gpt-oss-120b`.

- An unknown `provider/model` raises at construction.
- Any other prefix (`myorg/my-assistant`) resolves as a custom org/user assistant.

## Example workflows

Prompts this skill handles: *"summarize this file with gpt-4o via lightning"*, *"compare Claude and Gemini answers on this prompt"*, *"which gateway model is cheapest for this task?"*, *"push this checkpoint to the model registry"*.

**One-off inference from the shell:**

```bash
uv run --with lightning-sdk python -c "
from lightning_sdk.llm import LLM
llm = LLM('openai/gpt-4o', teamspace='my-org/my-teamspace')
print(llm.chat('Explain CUDA streams in two sentences.'))"
```

**Compare models on the same prompt (price-aware):**

```python
from lightning_sdk.llm import LLM
prompt = "Extract the action items from this meeting transcript: ..."
for model in ["openai/gpt-4o", "anthropic/claude-3-5-sonnet-20240620", "google/gemini-2.5-flash"]:
    llm = LLM(model, teamspace="my-org/my-teamspace")
    m = llm.metadata
    print(f"--- {model} (in ${m.prompt_price}/tok, out ${m.completion_price}/tok)")
    print(llm.chat(prompt, max_completion_tokens=300))
```

**Long-running assistant with memory:** use the multi-turn `conversation=` calls in the [Python SDK](#python-sdk) block, and `reset_conversation()` when done.

## API keys for direct REST access

For calling public inference endpoints outside the SDK (`Authorization: Bearer <key>`):

```bash
lightning api-key get [--org NAME]        # get-or-create the default model-API key, prints raw key
lightning api-key create --org NAME --name my-key
lightning api-key list; lightning api-key delete <KEY_ID>
```

If the user has multiple orgs, pass `--org` or set `LIGHTNING_ORG`; otherwise the key may be scoped to the wrong org. The exact OpenAI-compatible base URL is documented on the lightning.ai Model APIs page (it is not hardcoded in the SDK); the SDK's own `LLM.chat()` uses Lightning's assistants endpoint (`POST /v1/agents/{assistant_id}/conversations`) instead.

Don't conflate auth schemes: platform SDK/REST calls use Basic auth (`user_id:api_key`) when both `LIGHTNING_USER_ID` and `LIGHTNING_API_KEY` are set, and send the key alone as `Bearer` otherwise; model-API endpoints take `Bearer` with a key from `lightning api-key get`.

## Model checkpoint registry (separate concept)

`lightning model` manages **binary model artifacts** in a teamspace store, unrelated to gateway inference. Names are `org/teamspace/model[:version]`:

```bash
lightning model upload   my-org/my-teamspace/my-model ./checkpoints          # PATH is positional; only option is --cloud-account
lightning model download my-org/my-teamspace/my-model:v2 --download-dir ./out
```

```python
from lightning_sdk.models import upload_model, download_model, delete_model, list_model_versions
info = upload_model("my-org/my-teamspace/my-model", path="./ckpt")   # auto-versions vX
paths = download_model("my-org/my-teamspace/my-model")
```

## Gotchas

- **The first `LLM(...)` in a process freezes teamspace/auth resolution for all later instances, so set `teamspace=` on the first one.** The cache is class-level.
- **A personal teamspace (`teamspace="<user>/<teamspace>"`) can fail outside a Studio, so use an org-owned teamspace or run inside a Studio of the teamspace you want billed.** The error is `Teamspace ID is missing from the resolved authentication information.` (an SDK bug). Inside a Studio it silently bills the Studio's own teamspace instead.
- **Reasoning models (`openai/gpt-5*`) return empty responses on small `max_completion_tokens` budgets, so give them 1000+ or omit the cap.** They spend the budget on internal reasoning first. At 100 or less you may also get intermittent client-side deserialization `TypeError`s.
