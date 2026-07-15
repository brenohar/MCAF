
# Safety-Aware Embedded Agentic AI for Mining Edge Systems

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21374219.svg)](https://doi.org/10.5281/zenodo.21374219)
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
├── data/
│   └── cycles_20260704_135854.csv   # The audited 600-cycle experimental log
├── prompts/
│   └── action.gbnf                  # Grammar constraint for the SLM
├── src/
│   ├── agent_sasc.py                # Main Agentic loop, Pre-filter, and Watchdog
│   ├── config.py                    # Hyperparameters and model paths
│   ├── eaco_rag.py                  # Offline Vector Store retrieval logic
│   ├── logic_guard.py               # LTLf formal runtime monitors (Shields)
│   ├── mcp_server.py                # Synthetic telemetry and tool executor
│   ├── run_experiment.py            # Orchestrator for the SBESC E1 evaluation
│   ├── run_mrr_eval.py              # Script for RAG MRR evaluation
│   └── run_safety_eval.py           # Stress-test for LogicGuard MOET profiling
└── README.md
```

## 🚀 How to Replicate the Experiments

### 1. Requirements
This PoC is designed to run on a **Raspberry Pi 5 (16GB)** running standard Raspberry Pi OS (Debian) with Python 3.10+.
* `llama-cpp-python` (built for your specific hardware backend)
* `pandas`, `numpy`, `psutil`

### 2. Running the Evaluation
To execute the full 600-cycle battery across all modes (Prefilter-only, Pure-SLM, Pure-SLM+GBNF, Hybrid) and network scenarios (Connected, Degraded, Isolated):

```bash
# Clone the repository
git clone [https://github.com/your-username/mcaf-poc.git](https://github.com/your-username/mcaf-poc.git)
cd mcaf-poc

# Run the orchestrator
python src/run_experiment.py
```
*Note: Ensure that the Qwen2.5-0.5B-Instruct-Q4_K_M.gguf model is correctly mapped in `src/config.py` before execution.*

## 📊 Data Auditability
The `data/` folder contains the exact CSV trace used to generate **Table IV** and **Table V** of the paper. It logs per-cycle latency decompositions ($L_{dec}$, $L_{LG}$, $L_{tot}$), decision routing, thermal conditions, and deterministic correctness against the ground-truth policy $\pi^*$.

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
