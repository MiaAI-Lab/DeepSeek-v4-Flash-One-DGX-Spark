# DeepSeek-v4-Flash EXL3 on one DGX Spark

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://ko-fi.com/Z8Z3SPLOD" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://storage.ko-fi.com/cdn/kofi6.png?v=6" alt="Buy Me a Coffee at ko-fi.com" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

Single-node launcher for **DeepSeek V4 Flash 0731 (EXL3/ExLlamaV3)** with DSpark speculative decoding on one **NVIDIA DGX Spark** (GB10, SM121, 128 GiB unified memory).

Serves the `0xSero/deepseek-v4-flash-0731-spark` build (3.0 bpw EXL3) via the `sparkinfer` (formerly `b12x`) kernel stack — a complete, self-contained Docker recipe tuned for speed and KV-cache headroom on a single device.

> ⚙️ **Defaults changed (2026-08-20):** the default launcher `start.sh` now serves the **deep-context NVFP4 config** — `KV_RECORD=stock432` (native 432-byte records), `GPU_MEMORY_UTILIZATION=0.94`, `MAX_MODEL_LEN=334000`, `MAX_NUM_SEQS=1`, DSpark K5 healthy → **337,841-token KV pool** (a 255,400-token context served live). The prior 180k/584-byte setup lives on as `start-180k.sh` (writes `compose-180k.yaml`; default `start.sh` writes `compose.yml`). The NVFP4 dual-cache prefill bugs behind the old either/or are fixed — full story in the internal postmortem (kept local, not in this repo).

## Highlights

- **One DGX Spark, tensor-parallel 1** — no second node required (unlike the official FP4 build, which needs TP2 across two Sparks).
- **DSpark K5 speculative decoding** with a K64 draft model (`MODE=dspark`, fixed K5, K64 draft).
- **`nvfp4_ds_mla`** compressed KV cache with the `B12X_MLA_SPARSE` attention/MoE backend.
- Tuned CUDA-graph capture (`[6,12,24]`) so concurrent decode stays on captured graphs instead of falling back to eager.
- Long-prefill fairness (decode-starvation guard) enabled natively.
- Two upstream kernel backports applied as read-only bind-mounts (see [Backports](#backports)).
- Weights download **fully locally** into `./hf-hub` (no remote machine involved). An optional LAN mode reuses one copy of the ~107 GB across machines via an SSHFS share (`REMOTE_HOST` — opt-in).

### Measured results (single boot, small samples — directional)

| Change | Before | After | Delta |
|---|---|---|---|
| cuDAG capture 6 → 24, agg decode 4× short ctx | 35.4 tok/s | **77.9 tok/s** | **+120%** |
| cuDAG capture 6 → 24, agg decode 4× long ctx | 22.5 tok/s | **43.2 tok/s** | **+92%** |
| KV pool tokens | 140,424 | **180,695** | **+26%** |

Methodology caveat: the decode numbers include time-to-first-token and are single warm-up runs on a shared desktop workload; treat as directional, not a benchmark.

## Requirements

- **Hardware:** one NVIDIA DGX Spark (GB10, SM121, ≥128 GiB unified memory), GPU passthrough to Docker via the NVIDIA Container Toolkit.
- **OS:** Linux aarch64 (DGX OS). The runtime image is **aarch64-only**.
- **Software:** Docker Engine + Compose v2, `curl`, and ~110+ GiB free local disk. `sshfs` + `fuse3` are only needed for the optional LAN sharing mode (auto-installed with sudo when missing; requires `user_allow_other` in `/etc/fuse.conf`).
- **Network:** internet access to HuggingFace for the one-time ~107 GB download. The download is fully local — no remote host required. (Optional LAN mode: set `REMOTE_HOST`/`REMOTE_USER`/`REMOTE_SHARE_DIR` to reuse a single copy across machines.) No HuggingFace login required; the repo and image are public. If you do hit HF rate limits or need a private repo, set the optional `HF_TOKEN` in `start.sh` (or via env / `.env`).

## Quick start

```bash
./start.sh      # DEFAULT / recommended: deep-context NVFP4 (334k, DSpark) — writes compose.yml
./start-180k.sh # ALTERNATIVE: 180k, up to 4 concurrent requests, stock 584-byte records — writes compose-180k.yaml
./start.sh --no-wait   # start without waiting
```

First boot is intentionally long: pulls the image, downloads ~107 GB of weights locally (into `./hf-hub`), coalesces TP4→TP1 losslessly, builds the K64 draft, and captures CUDA graphs. It is marked `healthy` only when the OpenAI-compatible endpoint responds.

> ℹ️ **No compose files are shipped in this repo** — `compose.yml` (from
> `start.sh`) and `compose-180k.yaml` (from `start-180k.sh`) are generated
> automatically by their launcher on the first run and rewritten on every
> launch. They are gitignored, since the real config lives in the launchers.
> To produce them without starting anything: `./start.sh compose-gen` (the
> 180k variant has no gen-only subcommand; its file is written when
> `./start-180k.sh` runs). Do not hand-edit them.

### Weights bootstrap — fully local

Everything downloads **on this machine**; no remote host is involved in the
default path:

- **Default:** `./start.sh` (and `./start-180k.sh`) auto-download on first
  boot into the local HF cache root `./hf-hub` and coalesce into
  `./data/tp1`. The losslessly coalesced serving checkpoint, the K64 draft,
  and the runtime caches all live on this machine.
- **Optional offline prep:** `./download.sh` performs the same
download+coalesce+verify standalone (also fully local). Once
  `./data/tp1/rank-sliced-tp1-manifest.json` exists, boot skips the
  download/coalesce step entirely, so a network-free runtime follows — the
  manifest (not an env knob) is the gate.
- **Optional LAN sharing:** set `REMOTE_HOST` (+ `REMOTE_USER` /
  `REMOTE_SHARE_DIR` / `MIA_MOUNT` / `HF_CACHE`) to reuse a single weight
  copy across machines instead of downloading locally — the spark3
  arrangement, preserved as an opt-in mode on `main` and unchanged on the
  `mia-shared-setup` branch.

## Choosing a launcher: `start.sh` (default) vs `start-180k.sh` (alternative)

Both launchers drive the **same** stack — identical pinned weights, image, lossless
to-TP1 coalescing, K64 DSpark draft, and OpenAI-compatible endpoint. They differ
only in the serving configuration they generate, and **you can only run one at a
time**: both use the same compose project name, so whichever launcher runs last
recreates the container in *its* config. The difference comes down to two
opposite goals:

- **`start.sh` — deep single-request context (the default).** Native **432-byte NVFP4**
  KV records give a **337,841-token KV pool** and `MAX_MODEL_LEN=334000`, so a
  single request can span a very long context. It runs `MAX_NUM_SEQS=1` (one
  request at a time) and uses `restart: on-failure:1` so a failed boot **stops
  after one failure** and can never death-spiral the host.
- **`start-180k.sh` — concurrency over depth.** The stock **584-byte FP8-compat**
  KV records (~181k-token pool) keep `MAX_MODEL_LEN=180000` but allow
  **`MAX_NUM_SEQS=4`** concurrent requests sharing that pool (4×~45k, 2×~90k,
  1×~180k). Legacy `restart: unless-stopped` policy. This is the original
  launcher, kept for compatibility.

| | `start.sh` (**default**) | `start-180k.sh` (alternative) |
|---|---|---|
| Goal | one very-long request | several concurrent shorter requests |
| KV record layout | native **432 B NVFP4** (`KV_RECORD=stock432`) | 584 B FP8-compat padded (stock semantics) |
| KV pool (validated) | **337,841 tokens** | ~181k tokens |
| `MAX_MODEL_LEN` default | **334000** | 180000 |
| `MAX_NUM_SEQS` default | 1 | 4 |
| Boot safety | `restart: on-failure:1` — single-clean-failure, never loops | `restart: unless-stopped` (legacy) |
| Compose | `compose.yml` + `COMPOSE_FILE` | `compose-180k.yaml` |
| Entrypoint | patched `entrypoint-256k.sh` (honors NVFP4 envs) | `entrypoint-no-download.sh` gates download |
| `AUTO_DOWNLOAD` | image default (always on) | env-gated (`AUTO_DOWNLOAD=0` needs `./download.sh`) |
| History | formerly `start-256k.sh` | formerly `start.sh` |

**Start with `start.sh`.** Its only hard requirement is **free host RAM ≥ 114.3 GiB at
launch** (0.94 × 121.63 GiB; this UMA machine shares the 121.63 GiB between GPU and
host) — stop the old container first and check `free -h` before launching. If a
boot ever fails (KV check or otherwise), it stops after one failure; lower
`MAX_MODEL_LEN` a notch and retry once — don't launch repeatedly while the host
is loaded. Choose `start-180k.sh` only if you specifically need concurrency; each
request is then capped at 180k and the pool is shared.

### Try it

```bash
curl -sS http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' -d '{
    "model": "deepseek-v4-flash-0731",
    "messages": [{"role":"user","content":"Write a correct Python function that returns the first n Fibonacci numbers."}],
    "temperature": 0, "max_completion_tokens": 256 }'
```

Served model name: `deepseek-v4-flash-0731`. API: `http://127.0.0.1:8888/v1` (OpenAI-compatible chat/responses endpoints, DSpark spec decoding active).

## Repository layout

| Path | Purpose |
|---|---|
| `start.sh` | **Default launcher** (deep-context NVFP4, 334k/1-seq, DSpark) — all tunables live here; **regenerates** `compose.yml` (do not edit that file directly) |
| `start-180k.sh` | **Alternative** launcher: 180k, 4 concurrent seqs, 584-byte records (legacy semantics); regenerates `compose-180k.yaml` |
| `compose.yml` | Generated by `start.sh` (default compose name); pinned image + mounts + runtime env |
| `compose-180k.yaml` | Generated by `start-180k.sh`; pinned image + mounts + runtime env |
| `image-patch/` | Read-only bind-mount overrides (coalescer + kernel backports) |
| `data/` | Serving checkpoint (`tp1/`), K64 draft, caches (on local disk) |
| `cache/` | Runtime JIT/kernel caches (CuTeDSL, TileLang, TRITON, vLLM) |

## Commands

| Command | Action |
|---|---|
| `./start.sh` | mount share + start + wait for `/health` |
| `./start.sh --no-wait` | start without waiting |
| `./start.sh mount` / `unmount` | manage the SSHFS share |
| `./start.sh logs` / `ps` / `status` | inspect runtime |
| `./start.sh stop` / `restart` / `down` | lifecycle (preserves `data/` + caches) |
| `./start.sh pull` | pull the pinned image now |
| `./start.sh help` | usage |
| ... `mount` / `unmount` | manage the optional SSHFS share (no-op in local mode) |

## Tunables (edit in `start.sh`)

| Variable | Default | Notes |
|---|---|---|
| `MAX_MODEL_LEN` | 334000 | `start.sh` default: ~1.1% under the validated 337,841-token NVFP4 pool; lower it if a boot ever fails the KV check (180k variant: 180000) |
| `MAX_NUM_SEQS` | 1 | `start.sh` default (single deep-context request; raise to share the pool: 2×169k, 3×113k, 4×84k) |
| `MAX_NUM_BATCHED_TOKENS` | 8224 | prefill budget |
| `GPU_MEMORY_UTILIZATION` | 0.94 | **max this host boots at** (see KV section) |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | 0 | removes the profiler's ~0.68 GiB graph over-reservation (real usage is 0.07 GiB) → grows KV |
| `LONG_PREFILL_TOKEN_THRESHOLD` | 1024 | caps prefill chunks; prevents decode starvation (0=off) |
| `MAX_NUM_PARTIAL_PREFILLS` | 0 | wired but currently a no-op on this fork |
| `MAX_CUDAGRAPH_CAPTURE_SIZE` | 24 | = seqs×(k+1); 0=image default |
| `CUDAGRAPH_CAPTURE_SIZES` | `6,12,24` | explicit capture list; 0=image default |
| `HF_TOKEN` | *(empty)* | optional HF token — normally **not** needed (repo + image are public); set it to avoid rate limits or reach a private repo. Also honored from the environment or a local `.env` (both launchers; the set-in-file knob is in `start.sh`) |
| `SERVED_MODEL_NAME` | `deepseek-v4-flash-0731` | id clients send as `model` |
| `MODE` | `dspark` | fixed K5 DSpark draft |
| `DSPARK_*` | — | draft size/experts knobs |
| `VERIFY_MODEL_CHECKSUMS` | 1 | 0 skips SAP-256 inventory |
| `DEFAULT_CHAT_TEMPLATE_KWARGS_THINKING` | `true` | server-side default thinking (client kwargs override) |
| `DEFAULT_CHAT_TEMPLATE_KWARGS_EFFORT` | `max` | server-side default effort (override to `high`/`low`/`false`) |
| `REMOTE_HOST` / `REMOTE_USER` / `REMOTE_SHARE_DIR` | *(empty)* / `mia` / `/home/mia/shared` | LAN-share mode, **opt-in** — set `REMOTE_HOST` to reuse one weight copy across machines; defaults target the spark3 shared folder (`10.0.0.1` / `mia` / `/home/mia/shared`) |
| `HF_CACHE` | `./hf-hub` (local) or `$MIA_MOUNT` (remote mode) | where weights download — always resolved absolute |
| `SERVING_PORT` | 8888 | OpenAI-compatible port |

Every tunable is an environment variable override: `GPU_MEMORY_UTILIZATION=0.94 ./start.sh`.

## About the KV cache

Two record layouts are switchable with `KV_RECORD` on `start.sh`:

- **`stock432` (default, fixed 2026-08-20):** native 432-byte NVFP4 records → **337,841-token pool** at util 0.94 (255,400-token context served, DSpark acceptance 0.65–0.92 / 0.44 / 0.29–0.46 / 0.19–0.42 / 0.12). The dual-cache prefill path had four NVFP4 bugs (fixed via `image-patch/sparkinfer/` bind mounts) — see the internal postmortem (kept local, not in this repo).
- **`padded`:** 584-byte FP8-compat records (stock semantics, ~270k pool at 256k) — the fallback.

The 180k/584 numbers below are the historical sweep for the `start-180k.sh` variant:

This host reports only ~114.5 GiB of 121.63 as free (the unified-memory display/desktop holds ~7 GiB), so `GPU_MEMORY_UTILIZATION` above **~0.940 fails to boot** — the vendor's recipe value 0.9465 does **not** start here. The numbers below are the historical sweep of the **584-byte FP8-compat layout used by `start-180k.sh`**; the 432-byte NVFP4 layout used by the default `start.sh` reaches **337,841 tokens** (see the intro). The `start-180k.sh` ceiling on this hardware:

| Config | KV pool | Notes |
|---|---:|---|
| 0.93 (stock) | ~142k tokens | initial |
| 0.936 + est=0 | ~165k tokens | graph reservation reclaimed |
| **0.940 + est=0 (sweep winner)** | **~181k tokens** | validated with a 130k prefill, no OOM |

Pool of 180k tokens ⇒ ~1.39 concurrent full-length (131k) requests. For more KV, options are structural (smaller weights / lower bpw, or a 2-node TP2 stack).

## About the EXL3/Trellis quantization

This build ships **EXL3 3.0 bpw** weights (MCG codebook / Trellis `TR3` tier) on a **REAP-pruned K216** checkpoint that retains **216 of 256 experts** per MoE scope. Size: ~99.5 GiB. Non-routed tensors (attention, embeddings, output head, mHC, compressor, indexer) stay FP8/BF16. Two independent things are happening, and it's worth keeping them separate:

- **REAP** decides *which* experts survive (quality impact roughly follows router top-k=6 coverage; low-saliency experts are dropped).
- **EXL3/Trellis** decides *how precisely* the surviving weights are stored (per-tensor importance weighting, codebook + trellis).

### What EXL3 is (and is not)

- EXL3 is a **QTIP-style trellis/codebook** format with per-tensor importance weighting — it uses **non-uniform bit allocation** and strong weight reconstruction. It is **not** a uniform 3-bit round-to-nearest quant and therefore is **not** comparable to classic GGUF K-quants or basic I-quants at the same average bpw.
- It supports **fractional bits-per-weight** (this build: 3.0), with output/head layers kept higher (head_bits 8 in the upstream ladder).
- Runtimes: ExLlamaV3 and this SparkInfer/vLLM fork only — EXL3 has **no** llama.cpp/Ollama path and does not map onto GGUF numeric tiers.

### EXL3 ↔ GGUF quality mapping (community consensus)

| EXL3 bpw | Typical GGUF quality equivalent | Notes |
|---|---:|---|
| 2.0–2.5 | IQ2_M / IQ3_XXS territory (usable) | EXL3 stays more coherent |
| **3.0** | **IQ4_XS / Q4_K_S**, often feels like Q4–Q5 | **Strongest advantage zone for EXL3** |
| 4.0 | Q4_K_M / Q4_K_L (sometimes better) | Early tests: EXL3 4.0 ≈ EXL2 5.0 / GGUF Q4_K |
| 5.0+ | Q5_K_M / Q6_K | Closer to parity |

At matched bits-per-weight, EXL3 and GGUF I-quants are roughly similar, but EXL3 pulls ahead at low bpw because of better error distribution and codebook design.

### For this specific model

Treat **EXL3 3.0 bpw ≈ Q4_K_M–Q5_K range in GGUF quality, often closer to Q5 in practice** for this model — noticeably better than a standard GGUF Q3 at similar size, in line with EXL3's reputation for punching above its bit rate. A user who tested this exact Spark recipe reported it feels about **Q5 GGUF quality** (previous Spark recipes used much lower-quality Q2/Q3 GGUF). Exact perceived quality still depends on the task — coding/agentic workloads were part of the calibration here.

## Client configuration

The server exposes an OpenAI-compatible API on `http://127.0.0.1:8888/v1`. Recommended settings for any client:

| Setting | Value | Notes |
|---|---|---|
| Base URL | `http://127.0.0.1:8888/v1` | |
| Model id | `deepseek-v4-flash-0731` | sent as `model` |
| Context window | up to 334000 (`start.sh` default) | actual ceiling is the KV pool: **337,841 tokens** (deep-context). For the `start-180k.sh` variant use 180000 |
| Max output tokens | e.g. 32768 | anything ≤ `MAX_MODEL_LEN` is accepted |
| Tokenizer | DSV4 (`deepseek_v4`) | enabled server-side |
| Reasoning | **thinking ON, effort `max` by default** | this is the server-side default; send `chat_template_kwargs` to override per request (thinking `false`, or `reasoning_effort` low/high/max) |
| Tool calling | supported (`deepseek_v4` parser, auto tool choice) | |

### Example — pi agent (`~/.pi/agent/models.json`)

The pi coding agent can target this server directly. Model config (this exact entry is already installed at `~/.pi/agent/models.json`):

```json
"deepseek-v4-flash-spark-local": {
  "baseUrl": "http://127.0.0.1:8888/v1",
  "apiKey": "dummy",
  "api": "openai-completions",
  "authHeader": false,
  "auth": "none",
  "models": [
    {
      "id": "deepseek-v4-flash-0731",
      "name": "DeepSeek V4 Flash 0731 Spark · DSpark · 334k (local Spark)",
      "reasoning": true,
      "input": ["text"],
      "contextWindow": 334000,
      "maxTokens": 32768,
      "thinkingLevelMap": {
        "minimal": null, "low": null, "medium": null,
        "high": "high", "max": "max"
      },
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "requiresReasoningContentOnAssistantMessages": true,
        "maxTokensField": "max_tokens",
        "thinkingFormat": "deepseek"
      }
    }
  ]
}
```

Then select `deepseek-v4-flash-0731` in pi.

## Backports

Two fixes from `local-inference-lab/b12x` (the current maintainer repo of the `sparkinfer`/b12x kernel stack) are applied onto the image's pinned kernel tree (`272a84bd`) as read-only bind-mounts in `image-patch/sparkinfer/`:

- **#150** — preallocate W4A16 route histograms for CUDA-graph capture.
- **#228** — keep graphed tiny-decode routes with inactive expert ids in range (prevents out-of-range reads on graph padding).

The 0xSero image is pinned and no newer build (with newer kernel commits) is published yet, which is why these are backported locally. Remove the `image-patch/sparkinfer/` mounts from `start.sh` to return to stock kernels.

## Credits & links

- Weights: [`0xSero/deepseek-v4-flash-0731-spark`](https://huggingface.co/0xSero/deepseek-v4-flash-0731-spark) (REAP-K216, EXL3 3.0 bpw, Trellis) and the upstream [`0xSero/DeepSeek-V4-Flash-0731-EXL3-3.0bpw`](https://huggingface.co/0xSero/DeepSeek-V4-Flash-0731-EXL3-3.0bpw)
- Runtime image: `ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer` (NVIDIA vLLM 26.02 base)
- Kernel stack: [`local-inference-lab/b12x`](https://github.com/local-inference-lab/b12x) (sparkinfer / formerly b12x)
- Design reference: [`MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) (2-node TP2 recipe; our speed/KV work derives from its methodology)

## License

The packaging/orchestration glue in this repository is licensed under the [MIT License](LICENSE). The runtime image, model weights, and upstream libraries are covered by their own licenses.
