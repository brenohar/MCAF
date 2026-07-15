# src/run_experiment.py — comparação LLM puro vs modo híbrido
import random
import pandas as pd
from pathlib import Path
import csv
import uuid
import os
import psutil
import time
import datetime  # <-- NOVA BIBLIOTECA PARA O TIMESTAMP

from config import CONFIG
from agent_sasc import AgentSASC

# ---------------------------------------------------------------------------
# Configurações de Auditoria (Exigência do Revisor SBESC)
# ---------------------------------------------------------------------------
random.seed(42)
RUN_ID = str(uuid.uuid4())
SEED = 42

BASE_DIR = Path(__file__).parent.parent

# --- MELHORIA DE CONTROLE DE VERSÃO: TIMESTAMP NO NOME DO ARQUIVO ---
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = BASE_DIR / "data" / f"cycles_{TIMESTAMP}.csv"

LOG_HEADERS = [
    "run_id", "seed", "controller", "scenario", "cycle_idx",
    "t_obs", "t_orient", "t_dec", "t_LG",
    "L_dec_ms", "L_LG_ms", "L_tot_ms",
    "route", "proposed_action", "executed_action", "gt_action",
    "lg_blocked", "slm_raw_completion_ms", "cpu_temp_C", "rss_mb"
]

N_RUNS = 50

# Ajustado para os nomes em inglês exigidos pelo script de sanidade do revisor
SCENARIOS = [
    ("connected", True),
    ("degraded", False),
    ("isolated", False),
]

MODES = [
    "Prefilter-only",
    "Pure-SLM",
    "Pure-SLM+GBNF",
    "Hybrid",
]

ENGINE_TEMP_HISTORY = []

# ---------------------------------------------------------------------------
# Funções Auxiliares de Sistema e Log
# ---------------------------------------------------------------------------
def get_cpu_temp() -> float:
    """Lê a temperatura da CPU no Linux/Raspberry Pi."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return 0.0  # Retorna 0.0 se não estiver no Raspberry/Linux

def get_ram_usage_mb() -> float:
    """Lê a memória consumida pelo processo (RSS) em MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def init_csv_logger():
    """Garante que o arquivo CSV e o cabeçalho existam."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = LOG_FILE.is_file()
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(LOG_HEADERS)

def log_cycle(row_dict: dict):
    """Salva instantaneamente um ciclo no CSV (append-only)."""
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=LOG_HEADERS)
        writer.writerow(row_dict)

# ---------------------------------------------------------------------------
# Heurísticas e Roteamento (Ground Truth - Passo 1.5)
# ---------------------------------------------------------------------------
def expected_action(row: dict) -> str:
    cmd = row.get("operator_command", "continue_operation")
    auth = row.get("auth_token", False)

    # 1. Estados Críticos (Ordem de prioridade corrigida)
    if row["brake_temp_c"] > 200.0: return "emergency_stop"
    if row["speed_kmh"] > 0.0 and cmd == "R": return "emergency_stop"
    if row["engine_temp_c"] > 105.0: return "activate_cooling"
    if cmd == "Op_risk" and not auth: return "continue_operation"

    # 2. Estados Não-Críticos (A Faixa de Trade-off)
    if 90.0 <= row["engine_temp_c"] <= 105.0:
        if len(ENGINE_TEMP_HISTORY) >= 5:
            ma_current = sum(ENGINE_TEMP_HISTORY[-5:]) / 5.0
            ma_prev = sum(ENGINE_TEMP_HISTORY[-6:-1]) / 5.0 if len(ENGINE_TEMP_HISTORY) >= 6 else ma_current
            if ma_current > ma_prev:
                return "activate_cooling" 
    
    return "continue_operation"

def get_route(controller_mode: str, row: dict) -> str:
    cmd = row.get("operator_command", "continue_operation")
    auth = row.get("auth_token", False)

    if controller_mode in ["Hybrid", "Prefilter-only"]:
        if row["brake_temp_c"] > 200.0 or \
           (row["speed_kmh"] > 0.0 and cmd == "R") or \
           row["engine_temp_c"] > 105.0 or \
           (cmd == "Op_risk" and not auth):
            return "prefilter"
        return "fallback" if controller_mode == "Prefilter-only" else "slm"
    return "slm"


# ---------------------------------------------------------------------------
# Execução Principal por Modo
# ---------------------------------------------------------------------------

def run_mode(mode_name: str) -> list[dict]:
    print("\n" + "=" * 78)
    print(f"=== CONTROLLER: {mode_name.upper()} ===")
    print("=" * 78)

    agent = AgentSASC(
        model_path=CONFIG["model_path"],
        n_gpu_layers=CONFIG["n_gpu_layers"],
        n_ctx=CONFIG["n_ctx"],
        n_batch=CONFIG["n_batch"],
        n_threads=CONFIG["n_threads"],
        top_k=CONFIG["top_k"],
        max_tokens=CONFIG["max_tokens"],
        controller_mode=mode_name,   # Passamos o nome exato agora
    )

    print(f"  model_load_ms = {agent.latency_model_load_ms:.1f} ms\n")
    rows = []

    for scenario_name, connectivity in SCENARIOS:
        print(f"=== Cenário: {scenario_name} ===")
        agent.history.clear()
        agent.cycle = 0
        ENGINE_TEMP_HISTORY.clear() # <-- ADICIONE ESTA LINHA PARA LIMPAR O HISTÓRICO

        for i in range(N_RUNS):
            t_obs_abs = time.time()
            
            r = agent.run_cycle(connectivity=connectivity)
            
            ENGINE_TEMP_HISTORY.append(r.sensors["engine_temp_c"]) # <-- ADICIONE ESTA LINHA PARA GRAVAR A TEMP

            gt = expected_action(r.sensors)
            route = get_route(mode_name, r.sensors)
            
            # ... resto do código continua igual ...

            # Reconstrução dos timestamps absolutos exigidos na auditoria
            t_orient_abs = t_obs_abs + (r.latency_obs_ms / 1000.0)
            t_dec_abs = t_orient_abs + ((r.latency_orient_ms + r.latency_dec_ms) / 1000.0)
            t_LG_abs = t_dec_abs + (r.latency_lg_ms / 1000.0)

            # Lógica da ação executada final (considera bloqueios do LogicGuard/PLC)
            if r.allowed:
                exec_action = r.action
            else:
                exec_action = "emergency_stop" if r.plc_fallback else "continue_operation"

            # Tempo bruto do SLM (Inferência + Prompt + Parse)
            #slm_raw_ms = r.latency_prompt_ms + r.latency_infer_ms + r.latency_parse_ms
            slm_raw_ms = r.slm_raw_ms

            # Montagem da linha exata pro CSV exigido
            log_row = {
                "run_id": RUN_ID,
                "seed": SEED,
                "controller": mode_name,
                "scenario": scenario_name,
                "cycle_idx": r.cycle,
                "t_obs": round(t_obs_abs, 4),
                "t_orient": round(t_orient_abs, 4),
                "t_dec": round(t_dec_abs, 4),
                "t_LG": round(t_LG_abs, 4),
                "L_dec_ms": r.latency_dec_ms,
                "L_LG_ms": r.latency_lg_ms,
                "L_tot_ms": r.latency_tot_ms,
                "route": route,
                "proposed_action": r.action,
                "executed_action": exec_action,
                "gt_action": gt,
                "lg_blocked": not r.allowed,
                "slm_raw_completion_ms": round(slm_raw_ms, 3),
                "cpu_temp_C": round(get_cpu_temp(), 1),
                "rss_mb": round(get_ram_usage_mb(), 2)
            }
            
            # Gravação em disco IMEDIATA (garante os dados se o Raspberry Pi travar)
            log_cycle(log_row)

            # Mantém em memória apenas pro relatório do terminal
            row_terminal = log_row.copy()
            row_terminal["engine_temp_c"] = r.sensors["engine_temp_c"]
            row_terminal["brake_temp_c"] = r.sensors["brake_temp_c"]
            row_terminal["speed_kmh"] = r.sensors["speed_kmh"]
            row_terminal["coherent"] = (r.action == gt)
            rows.append(row_terminal)

            print(
                f"  [{i+1:02d}] {r.action:25s} | "
                f"src={route:10s} | "
                f"allowed={r.allowed} | "
                f"Ldec={r.latency_dec_ms:.1f}ms | "
                f"Ltot={r.latency_tot_ms:.1f}ms"
            )

        m = agent.metrics()
        print(f"\n  -- Médias do cenário {scenario_name} --")
        print(f"     obs={m['avg_obs_ms']}ms  Ldec={m['avg_dec_ms']}ms  Ltot={m['avg_tot_ms']}ms")
        print(f"     cache_hits={m['cache_hits']}  cache_misses={m['cache_misses']}  fallbacks={m['plc_fallbacks']}\n")

    return rows

def print_report(df: pd.DataFrame) -> None:
    for mode_name in df["controller"].unique():
        dmode = df[df["controller"] == mode_name]
        print("\n" + "#" * 78)
        print(f"### RELATÓRIO FINAL — {mode_name.upper()} ###")
        print("#" * 78)

        print("\n=== LATÊNCIAS (ms) ===")
        print(dmode.groupby("scenario")[["L_dec_ms", "L_LG_ms", "L_tot_ms"]].describe().round(4))

        print("\n=== ORIGEM DA DECISÃO ===")
        print(dmode.groupby(["scenario", "route"]).size().unstack(fill_value=0))

        print("\n=== CONFORMIDADE (Coherent) ===")
        coherence = dmode.groupby("scenario")["coherent"].mean().round(4)
        for scenario, rate in coherence.items():
            status = "✅" if rate >= 0.80 else "⚠️  REVISAR"
            print(f"  {scenario:12s}: {rate * 100:.1f}%  {status}")


if __name__ == "__main__":
    # Garante a biblioteca psutil
    try:
        import psutil
    except ImportError:
        print("ERRO: Instale a biblioteca psutil executando 'pip install psutil' antes de rodar.")
        exit(1)

    print(f"SASC: Iniciando auditoria para {Path(CONFIG['model_path']).name}...")
    init_csv_logger()

    all_rows = []
    for mode_name in MODES:
        all_rows.extend(run_mode(mode_name))

    df = pd.DataFrame(all_rows)

    print_report(df)
    print(f"\nCSV auditável salvo linha a linha em: {LOG_FILE}")
