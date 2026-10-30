"""Single source of truth for engine, environment, and campaign parameters."""
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
SPECS_DIR = BASE_DIR / "specs"
DATA_DIR = BASE_DIR / "data" / "raw"
LOGS_DIR = BASE_DIR / "logs"

# Limiares fisicos formais (sincronizados entre Simulador, Gabarito, DFA e Manuais)
BRAKE_TEMP_MAX_C: float = 200.0
SPEED_MAX_KMH: float = 20.0
ENGINE_TEMP_CRIT_C: float = 105.0
ENGINE_TEMP_BAND_MIN_C: float = 90.0
MA_WINDOW_CYCLES: int = 5

# Matriz fixa de 20 equipamentos (6 com diretiva ativa nos documentos da Etapa 5)
ALL_EQUIPMENTS: Tuple[str, ...] = tuple(f"EQ-{i:02d}" for i in range(1, 21))
DIRECTIVE_EQUIPMENTS: Tuple[str, ...] = (
    "EQ-02", "EQ-05", "EQ-08", "EQ-11", "EQ-15", "EQ-19"
)

# Espacos de acoes: base (4 acoes cobrindo todas as corretivas) e tarefa rica (6 acoes)
ACTIONS_BASE: Tuple[str, ...] = (
    "continue_operation",
    "activate_cooling",
    "emergency_stop",
    "refuse_operation",
)

ACTIONS_RICH: Tuple[str, ...] = (
    "continue_operation",
    "activate_cooling",
    "emergency_stop",
    "refuse_operation",
    "schedule_maintenance",
    "reduce_speed",
)

CONTROLLERS: Tuple[str, ...] = (
    "Prefilter-only",
    "Pure-SLM",
    "Pure-SLM+GBNF",
    "Hybrid",
    "Hybrid+RAG",
)

SCENARIOS: Tuple[str, ...] = ("connected", "degraded", "isolated")
SEEDS: Tuple[int, ...] = (42, 101, 202, 303, 404)

MODEL_CATALOG = {
    ("0.5B", "Q4_K_M"): "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
    ("0.5B", "Q8_0"):   "Qwen2.5-0.5B-Instruct-Q8_0.gguf",
    ("1.5B", "Q4_K_M"): "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    ("1.5B", "Q8_0"):   "Qwen2.5-1.5B-Instruct-Q8_0.gguf",
    ("3B",   "Q4_K_M"): "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
    ("3B",   "Q8_0"):   "Qwen2.5-3B-Instruct-Q8_0.gguf",
}


@dataclass(frozen=True)
class EngineConfig:
    model_size: str = "0.5B"
    quantization: str = "Q4_K_M"
    n_ctx: int = 1024
    max_tokens: int = 16
    temperature: float = 0.0
    n_batch: int = 256
    n_threads: int = 4
    n_gpu_layers: int = 0
    rag_top_k: int = 2

    @property
    def model_path(self) -> Path:
        return MODELS_DIR / MODEL_CATALOG[(self.model_size, self.quantization)]


@dataclass(frozen=True)
class RunConfig:
    controller: str = "Hybrid+RAG"
    scenario: str = "isolated"
    seed: int = 42
    n_cycles: int = 1000
    watchdog_ms: float = 500.0
    watchdog_enabled: bool = True
    safe_default_action: str = "continue_operation"
    n_actions: int = 6              # 4 (base) ou 6 (riqueza da tarefa)
    band_mode: str = "directive"    # "trend" ou "directive"
    few_shot_k: int = 0             # 0, 1 ou 5 exemplos no prompt
    p_crit: float = 0.15
    p_auth: float = 0.10
    p_op_cmd: float = 0.12
    net_loss_prob: float = 0.20
    net_latency_ms: float = 150.0
    thermal_guard_max_c: float = 78.0

    @property
    def action_space(self) -> Tuple[str, ...]:
        return ACTIONS_RICH if self.n_actions == 6 else ACTIONS_BASE
