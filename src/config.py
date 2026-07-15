# src/config.py
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

CONFIG = {
    "model_path": str(BASE_DIR / "models" / "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"),
    "db_path": str(BASE_DIR / "data" / "chromadb"),
    "logs_dir": str(BASE_DIR / "logs"),

    # Configuração de inferência no Raspberry Pi 5 (CPU-only, 16 GB RAM)
    "n_gpu_layers": 0,          # sem offloading para GPU
    "n_ctx": 768,              # contexto conservador
    "n_batch": 128,             # batch menor para evitar picos de RAM
    "n_threads": 4,             # número de threads da CPU
    "top_k": 1,
    "max_tokens": 4,

    # Orçamentos de latência (ms)
    "budget_L_dec": 500,
    "budget_L_LG": 50,

    # Cenário: "connected", "intermittent", "disconnected"
    "current_scenario": "disconnected",
    "network_drop_prob": 0.4
}
