# Neuro-Symbolic Supervisory Control for Autonomous Mining Haulage (MCAF)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24354718.svg)](https://doi.org/10.5281/zenodo.24354718)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Target Venue](https://img.shields.io/badge/Journal-IEEE%20Access-00629B.svg)](https://ieeeaccess.ieee.org/)

**Official Source Code, Formal LTLf Artifacts, and Reproducibility Package for the *IEEE Access* Manuscript:**  
> *"Neuro-Symbolic Supervisory Control for Autonomous Mining Haulage: Integrating Local MCP-RAG, Low-Rank SLM Adaptation, and Compiled LTLf Runtime Shields"*  
> Breno H. N. Andrade and Ricardo A. R. Oliveira (2026).

---

## Overview

The **Mine-Continuum Agentic Framework (MCAF)** is a proof-of-concept neuro-symbolic supervisory control architecture for open-pit Autonomous Haulage Systems (AHS) operating across connected, degraded, and disconnected wireless regimes. MCAF decouples high-level contextual Standard Operating Procedure (SOP) arbitration from deterministic safety enforcement along an **Observe-Orient-Decide-Act (OODA)** loop:

1. **Cognitive Agentic Tier (General-Purpose Cores 0-2):** Combines an asymmetric **Hot/Cold Model Context Protocol (MCP)** retrieval server (`11` onboard local `hot` bulletins + `7` remote `cold` maintenance schedules indexed via `all-MiniLM-L6-v2`) with quantized `Qwen2.5` Small Language Models adapted via leak-free **Attention + SwiGLU MLP LoRA (`MLP-LoRA`)** and bounded by **logical timeout substitution**.
2. **Compiled Runtime Shield Tier (Isolated Core 3 / `PREEMPT_RT`):** Compiles five formal safety specifications written in Linear Temporal Logic over Finite Traces (LTLf) via `ltlf2dfa` + `MONA` (`11` DFA states, `17` symbolic BDD transition edges) into a zero-allocation, `mlockall`-pinned **ANSI C11** runtime guard (`c_guard/guard.c`).

---

## Key Verified Empirical Results

| Metric / Benchmark | Campaign Scope | Verified Result |
| :--- | :--- | :--- |
| **Isolated Decision Concordance (`MLP-LoRA Hybrid+RAG`)** | Desktop RTX 3050 (`5` seeds x `1,000` cycles, budget >= 3500 ms) | **94.3%** [93.7%, 94.8%] (+13.76 pp vs. `Prefilter-only`, p_Holm = 3.62e-4) |
| **Embedded Decision Concordance (Raspberry Pi 5)** | Arm Cortex-A76 (`500` cycles, >= 3500 ms / 500 ms) | **94.2%** (unconstrained) / **82.8%** (logical timeout fallback at 500 ms) |
| **Formal Shield Invariant Compliance** | All `635` experimental cells (`480,000` GPU + `12,500` Pi 5 cycles) | **100.0%** (zero violations; exhaustive proof over `3,456` combinations) |
| **Cross-Architecture Parity (`x86_64`/CUDA vs. ARM64)** | Cycle-paired embedded `MLP-LoRA` runs (`1,500` cycles) | **99.93%** action identity (`1,499/1,500` cycles) |
| **Compiled ANSI C11 Guard Latency (`PREEMPT_RT`)** | Raspberry Pi 5 Core 3 (`isolcpus=3`, `SCHED_FIFO 80`, `10^7` stressed calls) | **p50 = 56 ns**, **p99.99 = 93 ns**, **MOET = 9.32 us** (5.19x lower than idle CFS) |
| **Sub-1B GGUF Tensor Composition Discovery** | Binary header audit of `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` | **78.1%** (`132/169`) quantized weight tensors silently fall back to `Q5_0` |

---

## Repository Structure

```text
MCAF/
├── analysis/
│   ├── codegen_c_guard.py           # Lowers MONA-compiled DFAs into zero-allocation ANSI C11
│   ├── compile_shield.py            # Translates the 5 LTLf rules via ltlf2dfa + MONA
│   ├── eval_lora_and_behaviors.py   # Seed-level paired t-tests, Holm correction & McNemar 2x2 tables
│   └── verify_shield_totality.py    # Exhaustive 3,456-combination totality & non-empty fallback proof
├── c_guard/
│   ├── guard.c                      # Standalone zero-allocation ANSI C11 runtime shield
│   ├── guard.h                      # C11 shield API and sensor/verdict data structures
│   └── shield_tables.h              # Static MONA-compiled DFA transition & accepting bitmask tables
├── data/
│   └── raw/                         # 635 raw CSV logs (480 desktop GPU files + 155 Raspberry Pi 5 files)
├── mcaf/
│   ├── config.py                    # Single source of truth for thresholds, seeds, models, and schemas
│   ├── rag.py                       # Asymmetric Hot (11 local) / Cold (7 remote) MCP retriever
│   ├── shield.py                    # Python reference runtime shield executing MONA DFAs in O(1)
│   └── simulator.py                 # Paired AR(1) thermal telemetry simulator, ground-truth oracle & prefilter
├── models/
│   └── MANIFEST.sha256              # Cryptographic SHA-256 digests locking all 7 GGUF model & LoRA files
├── paper/
│   └── tables/                      # Modular LaTeX tables (.tex) included in the IEEE Access manuscript
├── run/
│   └── train_lora.py                # Leak-free RAFT-style MLP-LoRA training script (8 disjoint seeds)
├── specs/
│   └── compiled_shield.json         # Exported MONA DFA states, symbolic BDD edges & transition maps
├── tests/
│   └── test_etapa_10_c_guard.py     # Differential equivalence test suite (MONA vs. Python vs. C11)
├── mcaf_osf_replication_data.sha256 # SHA-256 checksum for the Zenodo archival bundle
├── mcaf_osf_replication_data.tar.gz # Immutable replication archive under DOI 10.5281/zenodo.24354718
└── README.md
