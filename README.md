# DeepSeek-v4-Flash-One-DGX-Spark - DwarfStar 4 Engine

<p align="center">
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

Thin, idempotent launcher scripts for running the **DeepSeek-V4-Flash** server built for the NVIDIA DGX Spark (GB10 / SM121) — the DwarfStar 4 (C/CUDA) engine that serves an OpenAI-compatible `/v1` API on `:8888`.

This is based on **antirez/ds4** (DwarfStar 4) and its DGX Spark fork:

- [antirez/ds4](https://github.com/antirez/ds4) — upstream DwarfStar 4 engine (MIT-licensed, C/CUDA)
- [Entrpi/ds4-on-spark](https://github.com/Entrpi/ds4-on-spark) — DGX Spark one-command install, benchmarks, and roofline analysis (what `start.sh` pulls)
- [Entrpi/ds4 (batched-serving)](https://github.com/Entrpi/ds4/tree/batched-serving) — the DGX-Spark-optimized CUDA perf fork used here

> **Note:** this repo does **not** use vLLM. `ds4-server` exposes the same `/v1` API that `vllm serve` does, but vLLM cannot read this repo's asymmetric GGUF, so the repo ships its own server. These scripts are just thin wrappers over that server's official installer.

## Requirements

- A NVIDIA DGX Spark (GB10 / SM121)
- Bash
- `curl`
- Disk space for the ~110 GiB GGUF weight set
- `ss` (for `stop.sh` / bind checks)

## Background

**DwarfStar 4** is a small, self-contained native inference engine optimized for DeepSeek V4 Flash, written by Salvatore Sanfilippo ([antirez](https://x.com/antirez)) — deliberately narrow, not a generic GGUF runner. [Entrpi](https://github.com/Entrpi) maintains a DGX-Spark-optimized CUDA perf fork of it, plus the [ds4-on-spark](https://github.com/Entrpi/ds4-on-spark) installer this repo wraps, which serves DeepSeek-V4-Flash entirely on-device on a GB10 / SM121 DGX Spark (RTX PRO 6000 / 5090-class `sm_120` also builds).

Thanks to Bleys Goodson ([@bleysg on X](https://x.com/bleysg)).

## Quick start

```bash
./start.sh    # full DSpark stack on 0.0.0.0:8888 (LAN-reachable)
```

First run does the heavy lifting: clones and builds the pinned fork, downloads the ~110 GiB GGUF set, smoke-tests, installs `ds4-serve`, and starts the server on `:8888`. Later runs fast-forward the clone to the pinned tag, skip GGUFs already on disk, and just start the server.

**Bind address:** `ds4-server` defaults to `127.0.0.1` and the upstream installer has no `--host` flag. This launcher therefore starts `ds4-serve` itself with `--host` (default **`0.0.0.0`**) so the API is reachable from other machines on the LAN. Use `HOST=127.0.0.1` for loopback-only.

If a previous run left the server bound only to localhost, `./start.sh` detects the mismatch against `HOST` and rebinds automatically.

## Usage

```bash
PORT=8889 ./start.sh            # different port
HOST=127.0.0.1 ./start.sh       # loopback-only
CTX=262144 ./start.sh           # smaller context budget  (KV ≈ 9.5 KiB/token)
./start.sh --no-dspark          # plain continuous decode (passes through)
```

Environment variables (all optional):

| Variable       | Default      | Meaning                                   |
|----------------|--------------|-------------------------------------------|
| `PORT`         | `8888`       | Server port                               |
| `HOST`         | `0.0.0.0`    | Bind address (`0.0.0.0` = all interfaces) |
| `CTX`          | `1000000`    | Context budget (KV ≈ 9.5 KiB/token)       |
| `DS4_SRC_DIR`  | `~/code/ds4` | Source directory for the pinned clone     |
| `DS4_GGUF_DIR` | `~/gguf`     | Weights directory                         |

## Stopping and restarting

```bash
./stop.sh                     # stop server on :8888 (wait until port is freed)
PORT=8889 ./stop.sh           # stop server on a different port

./stop.sh; ./start.sh         # force a clean restart
```

`start.sh` is idempotent: if the server is already answering on the port **with a bind that matches `HOST`**, it reports the running model and exits without touching anything.

## Checking status

```bash
# on the Spark
curl http://127.0.0.1:8888/v1/models
ss -tlnp | grep 8888          # expect 0.0.0.0:8888 (or *:8888)

# from another machine on the LAN
curl http://<spark-ip>:8888/v1/models
```

## Performance

![Performance on a single NVIDIA DGX Spark](bench.jpg)

Measured on a single NVIDIA DGX Spark (GB10 / SM121).

## Logs

Server logs go to `~/ds4-server.log`. Check there if the server doesn't come up:

```
!!! Not reachable yet — check $HOME/ds4-server.log
```

## Files

| File         | Purpose                                     |
|--------------|---------------------------------------------|
| `start.sh`   | Fetch installer, build, download weights, serve on `0.0.0.0:8888` |
| `stop.sh`    | Stop the `ds4-server` process              |
| `bench.jpg`  | Decode throughput benchmark (tok/s vs context) on a single DGX Spark |
