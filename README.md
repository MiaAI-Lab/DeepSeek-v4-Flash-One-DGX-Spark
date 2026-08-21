<h1 align="center">DeepSeek-v4-Flash EXL3 on one DGX Spark</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://ko-fi.com/Z8Z3SPLOD" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://storage.ko-fi.com/cdn/kofi6.png?v=6" alt="Buy Me a Coffee at ko-fi.com" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

Single-node launcher for **DeepSeek V4 Flash 0731 (EXL3/ExLlamaV3)** with DSpark speculative decoding on one **NVIDIA DGX Spark** (GB10, SM121, 128 GiB unified memory).

Serves the `0xSero/deepseek-v4-flash-0731-spark` build (3.0 bpw EXL3) via the `sparkinfer` (formerly `b12x`) kernel stack — a complete, self-contained Docker recipe tuned for speed and KV-cache headroom on a single device.

> ⚙️ **Defaults changed (2026-08-21):** the launcher `start.sh` serves the **deep-context NVFP4 config** — `KV_RECORD=stock432` (native 432-byte records), `GPU_MEMORY_UTILIZATION=0.94`, `MAX_MODEL_LEN=384000`, `MAX_NUM_SEQS=1`, DSpark K5 healthy → **439,622-token KV pool** (a 370,104-token context stress-tested with exact needle recall — see [Stress test](#stress-test)). The NVFP4 dual-cache prefill bugs behind the old either/or are fixed — full story in the internal postmortem (kept local, not in this repo).

---

## Highlights

- **One DGX Spark, tensor-parallel 1** — no second node required (unlike the official FP4 build, which needs TP2 across two Sparks).
- **DSpark K5 speculative decoding** with a K64 draft model (`MODE=dspark`, fixed K5, K64 draft).
- **`nvfp4_ds_mla`** compressed KV cache with the `B12X_MLA_SPARSE` attention/MoE backend.
- Tuned CUDA-graph capture (`[6,12,24]`) so concurrent decode stays on captured graphs instead of falling back to eager.
- Long-prefill fairness (decode-starvation guard) enabled natively.
- Two upstream kernel backports applied as read-only bind-mounts (see [Backports](#backports)).
- Weights download **fully locally** into `./hf-hub` (no remote machine involved). An optional LAN mode reuses one copy of the ~107 GB across machines via an SSHFS share (`REMOTE_HOST` — opt-in).

---

## Measured results

| Metric | Value |
|---|---|
| Decode tok/s (structured) — `start.sh`, 330k context | **44–47 tok/s** |
| KV cache pool | **439,622 tokens** (boot-dependent — see [Stress test](#stress-test)) |
| Stress test — 320k & 370k context | exact needle recall, **0 preemptions** — see [Stress test](#stress-test) |

---

## Stress test

Two needle-in-a-haystack runs on 2026-08-21, both on fresh boots of the deep-context config (`MAX_NUM_SEQS=1`, `KV_RECORD=stock432`, util 0.94), each at ~96% of that boot's `MAX_MODEL_LEN`.

**Method (both runs):** a single large user prompt of random-word filler, generated so prefix caching cannot deduplicate it (`cached_tokens: 0` — every token fresh KV); a secret passphrase planted at token ~20; a recall question at the very end; `temperature 0`, thinking disabled via `chat_template_kwargs`, `max_completion_tokens 128`.

| | Run 1 | Run 2 |
|---|---|---|
| `MAX_MODEL_LEN` | 334,000 | 384,000 (now the default) |
| KV pool (that boot) | 402,334 tokens (1.20×) | 439,622 tokens (1.14×, cold boot) |
| Prompt size | 320,037 tokens (1.33 MB) | 370,104 tokens (1.54 MB) |
| Share of pool | ~80% | ~84% |
| Needle recall from token ~20 | ✅ exact — `XQ-7741-BLUE` | ✅ exact — `ZK-9931-AMBER` |
| `finish_reason` | `stop` (clean, not truncated) | `stop` (clean, not truncated) |
| Preemptions | **0** | **0** |
| End-to-end | 517 s — prefill ~630 tok/s effective | 594 s — prefill ~625 tok/s effective |
| Server after | healthy, `/health` 200, KV usage back to 0% | healthy, `/health` 200, KV usage back to 0% |

Prefill throughput decays with depth — ~1,024 tok/s at the start of a request, ~350–614 tok/s past 300k accumulated — so a full-length 384k prefill takes ~10 minutes end-to-end.

These runs are the deepest exercise to date of the NVFP4 dual-cache prefill path fixed in `image-patch/sparkinfer/` — the pre-fix kernel NaN'd on any prompt ≥ 7 tokens; these tests pushed 320k and 370k tokens through it with exact recall.

<details>
<summary><b>Why the pool size differs between boots</b></summary>

The pool is a leftover-derived number — (util budget) − weights − profiled activation peak − non-torch overhead — and it moves between boots. Observed on this host: **337,841** (2-seq boot) → **402,334 → 430,909 → 440,461** (334k/1-seq) → **439,622** (384k/1-seq, cold). Two effects dominate:

1. **The hybrid cache split.** The model keeps 128-token sliding-window layers alongside full-depth global layers, and the split of KV bytes shifts with `MAX_NUM_SEQS`. Single-sequence boots route more of the budget into the global cache that defines the reported pool (337,841 at 2 seqs → 402k+ at 1 seq from comparable bytes).
2. **Cold vs. warm JIT.** A boot that compiles fresh kernels during warmup leaves less memory free at KV-sizing time than a warm boot hitting the on-disk JIT caches (observed swing: ~0.6 GiB ≈ 28k tokens).

The worst boot observed still clears `MAX_MODEL_LEN` with ≥ 1.14× headroom; a truly bad boot trips the boot-time KV check and stops cleanly (`restart: on-failure:1`).
</details>

---

## Requirements

- **Hardware:** one NVIDIA DGX Spark (GB10, SM121, ≥128 GiB unified memory), GPU passthrough to Docker via the NVIDIA Container Toolkit.
- **OS:** Linux aarch64 (DGX OS). The runtime image is **aarch64-only**.
- **EarlyOOM:** disable it if present on the host (`sudo systemctl disable --now earlyoom`). The server intentionally holds ~94% of the 128 GiB unified memory, so a user-space OOM killer can't tell a healthy server from a leak — and may kill it mid-serve.
- **Software:** Docker Engine + Compose v2, `curl`, and ~110+ GiB free local disk. `sshfs` + `fuse3` are only needed for the optional LAN sharing mode (auto-installed with sudo when missing; requires `user_allow_other` in `/etc/fuse.conf`).
- **Network:** internet access to HuggingFace for the one-time ~107 GB download. The download is fully local — no remote host required. (Optional LAN mode: set `REMOTE_HOST`/`REMOTE_USER`/`REMOTE_SHARE_DIR` to reuse a single copy across machines.) No HuggingFace login required; the repo and image are public. If you do hit HF rate limits or need a private repo, set the optional `HF_TOKEN` in `start.sh` (or via env / `.env`).

---

## Quick start

```bash
./start.sh      # start: deep-context NVFP4 (384k, DSpark) — writes compose.yml
./start.sh --no-wait   # start without waiting
```

First boot is intentionally long: pulls the image, downloads ~107 GB of weights locally (into `./hf-hub`), coalesces TP4→TP1 losslessly, builds the K64 draft, and captures CUDA graphs. It is marked `healthy` only when the OpenAI-compatible endpoint responds.

> ℹ️ **No compose file is shipped in this repo** — `compose.yml` is generated
> by `start.sh` on the first run and rewritten on every launch. It is
> gitignored, since the real config lives in the launcher. To produce it
> without starting anything: `./start.sh compose-gen`. Do not hand-edit it.

### Weights bootstrap — fully local

Everything downloads **on this machine**; no remote host is involved in the
default path:

- **Default:** `./start.sh` auto-downloads on first boot into the local HF
  cache root `./hf-hub` and coalesces into `./data/tp1`. The losslessly
  coalesced serving checkpoint, the K64 draft, and the runtime caches all
  live on this machine.
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

---

## Launch notes

The only hard requirement is **free host RAM ≥ 114.3 GiB at launch** (0.94 ×
121.63 GiB; this UMA machine shares the 121.63 GiB between GPU and host) —
stop an old container first and check `free -h` before launching. If a boot
ever fails (KV check or otherwise), it stops after one failure (`restart:
on-failure:1` — it can never death-spiral the host); lower `MAX_MODEL_LEN` a
notch and retry once, don't launch repeatedly while the host is loaded.

Concurrency is a single env override away: `MAX_NUM_SEQS=4 ./start.sh`. Note
that the pool itself shrinks at higher sequence counts (the hybrid cache split
shifts — a 2-seq boot observed ~337k total, ≈169k per slot), so depth trades
against concurrency. The rest of the launcher is the deep-context config
described in this README.

### Try it

```bash
curl -sS http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' -d '{
    "model": "deepseek-v4-flash-0731",
    "messages": [{"role":"user","content":"Write a correct Python function that returns the first n Fibonacci numbers."}],
    "temperature": 0, "max_completion_tokens": 256 }'
```

Served model name: `deepseek-v4-flash-0731`. API: `http://127.0.0.1:8888/v1` (OpenAI-compatible chat/responses endpoints, DSpark spec decoding active).

---

## Repository layout

| Path | Purpose |
|---|---|
| `start.sh` | **Launcher** (deep-context NVFP4, 384k/1-seq, DSpark) — all tunables live here; **regenerates** `compose.yml` (do not edit that file directly) |
| `compose.yml` | Generated by `start.sh`; pinned image + mounts + runtime env |
| `image-patch/` | Read-only bind-mount overrides (coalescer + kernel backports) |
| `data/` | Serving checkpoint (`tp1/`), K64 draft, caches (on local disk) |
| `cache/` | Runtime JIT/kernel caches (CuTeDSL, TileLang, TRITON, vLLM) |

---

## Commands

| Command | Action |
|---|---|
| `./start.sh` | start (deep-context config) + wait for `/health` |
| `./start.sh --no-wait` | start without waiting |
| `./start.sh mount` / `unmount` | manage the optional SSHFS share (no-op in local mode) |
| `./start.sh logs` / `ps` / `status` | inspect runtime |
| `./start.sh stop` / `restart` / `down` | lifecycle (preserves `data/` + caches) |
| `./start.sh pull` | pull the pinned image now |
| `./start.sh help` | usage |

---

## Tunables (edit in `start.sh`)

| Variable | Default | Notes |
|---|---|---|
| `MAX_MODEL_LEN` | 384000 | ~13% under the worst-observed cold-boot pool (439,622 tokens); lower it if a boot ever fails the KV check |
| `MAX_NUM_SEQS` | 1 | `start.sh` default (single deep-context request; raise for concurrency — the pool shrinks with seq count via the hybrid cache split, e.g. 2 seqs ≈ 337k total) |
| `MAX_NUM_BATCHED_TOKENS` | 8224 | prefill budget |
| `GPU_MEMORY_UTILIZATION` | 0.94 | **max this host boots at** (see KV section) |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | 0 | removes the profiler's ~0.68 GiB graph over-reservation (real usage is 0.07 GiB) → grows KV |
| `LONG_PREFILL_TOKEN_THRESHOLD` | 1024 | caps prefill chunks; prevents decode starvation (0=off) |
| `MAX_NUM_PARTIAL_PREFILLS` | 0 | wired but currently a no-op on this fork |
| `MAX_CUDAGRAPH_CAPTURE_SIZE` | 24 | = seqs×(k+1); 0=image default |
| `CUDAGRAPH_CAPTURE_SIZES` | `6,12,24` | explicit capture list; 0=image default |
| `HF_TOKEN` | *(empty)* | optional HF token — normally **not** needed (repo + image are public); set it to avoid rate limits or reach a private repo. Also honored from the environment or a local `.env` (the set-in-file knob is in `start.sh`) |
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

---

<details>
<summary><b>About the KV cache</b></summary>

Two record layouts are switchable with `KV_RECORD` on `start.sh`:

- **`stock432` (default, fixed 2026-08-20):** native 432-byte NVFP4 records → **439,622-token pool** at util 0.94 on the 384k/1-seq config (370,104-token context stress-tested, DSpark acceptance 0.65–0.92 / 0.44 / 0.29–0.46 / 0.19–0.42 / 0.12). The dual-cache prefill path had four NVFP4 bugs (fixed via `image-patch/sparkinfer/` bind mounts) — see the internal postmortem (kept local, not in this repo).
- **`padded`:** 584-byte FP8-compat records (stock semantics, ~270k pool at 256k) — the fallback.

This host reports only ~114.5 GiB of 121.63 as free (the unified-memory display/desktop holds ~7 GiB), so `GPU_MEMORY_UTILIZATION` above **~0.940 fails to boot** — the vendor's recipe value 0.9465 does **not** start here. The numbers below are the historical sweep of the **584-byte FP8-compat padded layout** (`KV_RECORD=padded`); the default **432-byte NVFP4 layout** (`KV_RECORD=stock432`) reaches **~440k tokens** (439,622 observed cold on the 384k/1-seq config; see the intro). The padded-layout ceiling on this hardware:

| Config | KV pool | Notes |
|---|---:|---|
| 0.93 (stock) | ~142k tokens | initial |
| 0.936 + est=0 | ~165k tokens | graph reservation reclaimed |
| **0.940 + est=0 (sweep winner)** | **~181k tokens** | validated with a 130k prefill, no OOM |

For more KV, options are structural (smaller weights / lower bpw, or a 2-node TP2 stack).

</details>

---

<details>
<summary><b>About the EXL3/Trellis quantization</b></summary>

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

</details>

---

## Client configuration

The server exposes an OpenAI-compatible API on `http://127.0.0.1:8888/v1`. Recommended settings for any client:

| Setting | Value | Notes |
|---|---|---|
| Base URL | `http://127.0.0.1:8888/v1` | |
| Model id | `deepseek-v4-flash-0731` | sent as `model` |
| Context window | up to 384000 (`start.sh` default) | actual ceiling is the KV pool: **439,622 tokens** (boot-dependent) |
| Max output tokens | e.g. 32768 | anything ≤ `MAX_MODEL_LEN` is accepted |
| Tokenizer | DSV4 (`deepseek_v4`) | enabled server-side |
| Reasoning | **thinking ON, effort `max` by default** | this is the server-side default; send `chat_template_kwargs` to override per request (thinking `false`, or `reasoning_effort` low/high/max) |
| Tool calling | supported (`deepseek_v4` parser, auto tool choice) | |

<details>
<summary><b>Example — pi agent</b> (<code>~/.pi/agent/models.json</code>)</summary>

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
      "name": "DeepSeek V4 Flash 0731 Spark · DSpark · 384k (local Spark)",
      "reasoning": true,
      "input": ["text"],
      "contextWindow": 384000,
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

</details>

---

## Backports

Two fixes from `local-inference-lab/b12x` (the current maintainer repo of the `sparkinfer`/b12x kernel stack) are applied onto the image's pinned kernel tree (`272a84bd`) as read-only bind-mounts in `image-patch/sparkinfer/`:

- **#150** — preallocate W4A16 route histograms for CUDA-graph capture.
- **#228** — keep graphed tiny-decode routes with inactive expert ids in range (prevents out-of-range reads on graph padding).

The 0xSero image is pinned and no newer build (with newer kernel commits) is published yet, which is why these are backported locally. Remove the `image-patch/sparkinfer/` mounts from `start.sh` to return to stock kernels.

---

## Credits & links

- Weights: [`0xSero/deepseek-v4-flash-0731-spark`](https://huggingface.co/0xSero/deepseek-v4-flash-0731-spark) (REAP-K216, EXL3 3.0 bpw, Trellis) and the upstream [`0xSero/DeepSeek-V4-Flash-0731-EXL3-3.0bpw`](https://huggingface.co/0xSero/DeepSeek-V4-Flash-0731-EXL3-3.0bpw)
- Runtime image: `ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer` (NVIDIA vLLM 26.02 base)
- Kernel stack: [`local-inference-lab/b12x`](https://github.com/local-inference-lab/b12x) (sparkinfer / formerly b12x)
- Design reference: [`MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) (2-node TP2 recipe; our speed/KV work derives from its methodology)

---

## License

The packaging/orchestration glue in this repository is licensed under the [MIT License](LICENSE). The runtime image, model weights, and upstream libraries are covered by their own licenses.
