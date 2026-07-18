# Safety-Aware Embedded Agentic AI for Mining Edge Systems

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21428122.svg)](https://doi.org/10.5281/zenodo.21428122)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Replication Package & Source Code for the SBESC 2026 submission.**

This repository contains the executable Proof of Concept (PoC) and the experimental datasets evaluating the **Mine-Continuum Agentic Framework (MCAF)**. It demonstrates how probabilistic Small Language Models (SLMs) can be safely deployed on extreme edge hardware (e.g., Raspberry Pi 5) for industrial automation by bounding them with deterministic safety guardrails (LogicGuard) and strict grammatical constraints (GBNF).

## 🧠 Architecture Overview
To solve the latency-intelligence-safety trilemma in disconnected mining operations, this PoC implements a bimodal control flow:
1. **Deterministic Low-Latency Path:** Resolves critical physical safety rules ($< 5$ ms) and enforces DFA-based interlocks via `LogicGuard`.
2. **Soft Real-Time Cognitive Path:** Evaluates non-critical sensor telemetry using offline Retrieval-Augmented Generation (EACO-RAG) and a highly quantized 4-bit SLM (**Qwen2.5-0.5B**).

The system relies on the **Model Context Protocol (MCP)** to standardize the tool-calling interface between the cognitive agent and the underlying emulated PLCs.

## 📂 Repository Structure

```text
├── analysis/
│   ├── make_tables.py               # Generates LaTeX tables and macros directly from CSVs
│   └── sanity_check.py              # Validates data integrity before table generation
├── data/
│   ├── ht05_ablation.csv            # Quantization ablation results
│   ├── ht06_sensitivity.csv         # Criticality sensitivity ablation
│   └── ht07_main_5000.csv           # Main latency and conformity results
├── paper/                           # (Auto-generated) Contains .tex snippets for the article
├── prompts/
│   └── action.gbnf                  # Grammar constraint for the SLM
├── src/
│   ├── agent_sasc.py                # Main Agentic loop, Pre-filter, and Watchdog
│   ├── check_gpu.py                 # Hardware validation script
│   ├── check_model_pi.py            # Model loading verification
│   ├── config.py                    # Hyperparameters and model paths
│   ├── eaco_rag.py                  # Offline Vector Store retrieval logic
│   ├── logic_guard.py               # LTLf formal runtime monitors (Shields)
│   ├── mcp_server.py                # Synthetic telemetry and tool executor
│   ├── run_experiment.py            # Legacy orchestrator
│   ├── run_mrr_eval.py              # Script for RAG MRR evaluation
│   ├── run_overnight_sbesc.py       # Main production orchestrator for data collection
│   └── run_safety_eval.py           # Stress-test for LogicGuard MOET profiling
└── README.md
```

## 🚀 How to Replicate the Experiments

### 1. Requirements

This PoC is designed to run on a **Raspberry Pi 5 (16GB)** running standard Raspberry Pi OS (Debian) with Python 3.10+.

* `llama-cpp-python` (built for your specific hardware backend)
* `pandas`, `numpy`, `psutil`

### 2. Execution Pipeline

The replication follows a clear separation between data collection and data analysis. Ensure that the `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` model is correctly mapped in `src/config.py` before execution.

**Step 2.1: Data Collection**
Execute the overnight battery to run the full cycle array across all modes and ablations. This will populate the `data/` folder with the raw `.csv` traces.

```bash
python src/run_overnight_sbesc.py
```

**Step 2.2: Sanity Check & Table Generation**
Once the data is collected, run the analysis pipeline. This will process the logs, verify arithmetic consistency, and automatically generate the LaTeX codes (`.tex` files) for the tables and numerical macros used in the paper.

```bash
python analysis/sanity_check.py
python analysis/make_tables.py
```

*The final generated tables will be available in the `paper/` directory.*

## 📊 Data Auditability

The `data/` folder contains the exact CSV traces used to generate the empirical results of the paper. They log per-cycle latency decompositions ($L_{dec}$, $L_{LG}$, $L_{tot}$), decision routing, thermal conditions, and deterministic correctness against the ground-truth policy \pi^*. The data generation logic can be audited directly inside `src/run_overnight_sbesc.py`.

## 📜 Citation

If you use this dataset or framework in your research, please cite our SBESC 2026 paper:

```bibtex
@inproceedings{andrade2026sbesc,
  author    = {Andrade, Breno H. N. and Garrocho, Charles T. B. and Silva, Fernando A. M. and Oliveira, Ricardo A. R.},
  title     = {Safety-Aware Embedded Agentic AI for Mining Edge Systems: An Experimental Evaluation of MCP, LogicGuard, and Offline RAG},
  booktitle = {Proceedings of the XIV Brazilian Symposium on Computing Systems Engineering (SBESC)},
  year      = {2026},
  publisher = {IEEE}
}
```

## ⚖️ License

This project is licensed under the MIT License - see the LICENSE file for details.
