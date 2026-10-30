"""Stage 8 Resumable Campaign Orchestrator with Thermal Guard and Fiscal Gate."""
import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

from llama_cpp import Llama

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.sanity_check import audit_csv_file  # noqa: E402
from mcaf.agent import AuditableAgent, read_board_temp_c  # noqa: E402
from mcaf.config import (  # noqa: E402
    CONTROLLERS,
    DATA_DIR,
    EngineConfig,
    LOG_COLUMNS,
    RunConfig,
    SCENARIOS,
    SEEDS,
)
from mcaf.rag import NeuralRetriever  # noqa: E402
from mcaf.simulator import SituationSimulator  # noqa: E402

CellSpec = Tuple[str, EngineConfig, RunConfig]


def _engine_for_device(
    size: str, quant: str, device: str, lora_path: str = ""
) -> EngineConfig:
    if device == "desktop_gpu":
        # 3B Q8_0 (~3.3 GB) usa 28 camadas na RTX 3050 4GB para evitar OOM no WSL2
        gpu_layers = 28 if (size == "3B" and quant == "Q8_0") else -1
        return EngineConfig(
            model_size=size,
            quantization=quant,
            n_threads=8,
            n_gpu_layers=gpu_layers,
            lora_path=lora_path,
        )
    if device == "desktop_cpu":
        return EngineConfig(
            model_size=size,
            quantization=quant,
            n_threads=8,
            n_gpu_layers=0,
            lora_path=lora_path,
        )
    if device == "pi_cpu":
        return EngineConfig(
            model_size=size,
            quantization=quant,
            n_threads=4,
            n_gpu_layers=0,
            lora_path=lora_path,
        )
    raise ValueError(f"Dispositivo desconhecido: {device}")


def build_campaign_cells(
    exp_name: str, device: str, n_cycles_override: int | None, lora_path: str = ""
) -> List[CellSpec]:
    sim_sleep = device == "pi_cpu"
    cells: List[CellSpec] = []

    if exp_name == "pilot_pi":
        n_cyc = n_cycles_override or 20
        for size in ("0.5B", "1.5B", "3B"):
            for quant in ("Q4_K_M", "Q8_0"):
                eng = _engine_for_device(size, quant, device)
                rcfg = RunConfig(
                    controller="Hybrid+RAG",
                    scenario="isolated",
                    seed=42,
                    n_cycles=n_cyc,
                    watchdog_enabled=False,
                    simulate_net_sleep=sim_sleep,
                )
                cells.append(("pilot_pi", eng, rcfg))
        return cells

    n_cyc = n_cycles_override or 1000

    if exp_name == "exp1_base":
        eng = _engine_for_device("0.5B", "Q4_K_M", device)
        for ctrl in CONTROLLERS:
            for scen in SCENARIOS:
                for seed in SEEDS:
                    rcfg = RunConfig(
                        controller=ctrl,
                        scenario=scen,
                        seed=seed,
                        n_cycles=n_cyc,
                        watchdog_enabled=False,
                        simulate_net_sleep=sim_sleep,
                    )
                    cells.append(("exp1_base", eng, rcfg))

    elif exp_name == "exp2_scale":
        for size in ("0.5B", "1.5B", "3B"):
            for quant in ("Q4_K_M", "Q8_0"):
                eng = _engine_for_device(size, quant, device)
                for ctrl in ("Pure-SLM", "Hybrid+RAG"):
                    for seed in SEEDS:
                        rcfg = RunConfig(
                            controller=ctrl,
                            scenario="isolated",
                            seed=seed,
                            n_cycles=n_cyc,
                            watchdog_enabled=False,
                            simulate_net_sleep=sim_sleep,
                        )
                        cells.append(("exp2_scale", eng, rcfg))

    elif exp_name == "exp3_fewshot":
        for size in ("0.5B", "1.5B"):
            eng = _engine_for_device(size, "Q4_K_M", device)
            for fs_k in (0, 1, 5):
                for gbnf_flag in (False, True):
                    if fs_k > 0 and gbnf_flag:
                        continue
                    sub = f"exp3_fewshot/fs{fs_k}_gbnf{int(gbnf_flag)}"
                    for ctrl in ("Pure-SLM", "Hybrid+RAG"):
                        for seed in SEEDS:
                            rcfg = RunConfig(
                                controller=ctrl,
                                scenario="isolated",
                                seed=seed,
                                n_cycles=n_cyc,
                                watchdog_enabled=False,
                                few_shot_k=fs_k,
                                force_gbnf=gbnf_flag,
                                simulate_net_sleep=sim_sleep,
                            )
                            cells.append((sub, eng, rcfg))

    elif exp_name == "exp3_lora":
        if not lora_path:
            raise ValueError("Informe --lora-path para rodar exp3_lora.")
        eng = _engine_for_device("0.5B", "Q4_K_M", device, lora_path=lora_path)
        for ctrl in ("Pure-SLM", "Hybrid+RAG"):
            for seed in SEEDS:
                rcfg = RunConfig(
                    controller=ctrl,
                    scenario="isolated",
                    seed=seed,
                    n_cycles=n_cyc,
                    watchdog_enabled=False,
                    simulate_net_sleep=sim_sleep,
                )
                cells.append(("exp3_lora", eng, rcfg))

    elif exp_name == "exp4_watchdog_live":
        eng = _engine_for_device("0.5B", "Q4_K_M", device)
        for seed in SEEDS:
            rcfg = RunConfig(
                controller="Hybrid+RAG",
                scenario="isolated",
                seed=seed,
                n_cycles=n_cyc,
                watchdog_ms=500.0,
                watchdog_enabled=True,
                simulate_net_sleep=sim_sleep,
            )
            cells.append(("exp4_watchdog_live", eng, rcfg))

    elif exp_name == "exp5_sensitivity":
        eng = _engine_for_device("0.5B", "Q4_K_M", device)
        sweeps = [
            ("loss_0.00", {"scenario": "degraded", "net_loss_prob": 0.00}),
            ("loss_0.10", {"scenario": "degraded", "net_loss_prob": 0.10}),
            ("loss_0.20", {"scenario": "degraded", "net_loss_prob": 0.20}),
            ("loss_0.40", {"scenario": "degraded", "net_loss_prob": 0.40}),
            ("lat_050ms", {"scenario": "degraded", "net_latency_ms": 50.0}),
            ("lat_300ms", {"scenario": "degraded", "net_latency_ms": 300.0}),
            ("lat_600ms", {"scenario": "degraded", "net_latency_ms": 600.0}),
            ("pcrit_0.01", {"scenario": "isolated", "p_crit": 0.01}),
            ("pcrit_0.05", {"scenario": "isolated", "p_crit": 0.05}),
            ("pcrit_0.25", {"scenario": "isolated", "p_crit": 0.25}),
            ("pauth_0.05", {"scenario": "isolated", "p_auth": 0.05}),
            ("pauth_0.25", {"scenario": "isolated", "p_auth": 0.25}),
        ]
        for tag, overrides in sweeps:
            for ctrl in ("Prefilter-only", "Hybrid+RAG"):
                for seed in SEEDS:
                    base_rcfg = RunConfig(
                        controller=ctrl,
                        seed=seed,
                        n_cycles=n_cyc,
                        watchdog_enabled=False,
                        simulate_net_sleep=sim_sleep,
                    )
                    rcfg = replace(base_rcfg, **overrides)
                    cells.append((f"exp5_sensitivity/{tag}", eng, rcfg))

    elif exp_name == "exp6_richness":
        for size in ("0.5B", "1.5B", "3B"):
            eng = _engine_for_device(size, "Q4_K_M", device)
            for n_act, b_mode in ((4, "trend"), (6, "directive")):
                for fs_k in (0, 5):
                    sub = f"exp6_richness/act{n_act}_{b_mode}_fs{fs_k}"
                    for ctrl in ("Prefilter-only", "Hybrid", "Hybrid+RAG"):
                        if ctrl == "Prefilter-only" and (size != "0.5B" or fs_k != 0):
                            continue
                        for seed in SEEDS:
                            rcfg = RunConfig(
                                controller=ctrl,
                                scenario="isolated",
                                seed=seed,
                                n_cycles=n_cyc,
                                n_actions=n_act,
                                band_mode=b_mode,
                                few_shot_k=fs_k,
                                watchdog_enabled=False,
                                simulate_net_sleep=sim_sleep,
                            )
                            cells.append((sub, eng, rcfg))
    else:
        raise ValueError(f"Experimento desconhecido: {exp_name}")

    return cells


def wait_for_thermal_headroom(max_temp_c: float, cool_target_c: float = 70.0) -> None:
    temp = read_board_temp_c()
    if temp >= max_temp_c:
        print(
            f"[THERMAL GUARD] Placa a {temp:.1f} C (>= {max_temp_c:.1f} C). "
            f"Pausando ate baixar para <= {cool_target_c:.1f} C...",
            flush=True,
        )
        while read_board_temp_c() > cool_target_c:
            time.sleep(15.0)
        print(f"[THERMAL GUARD] Temperatura normalizada ({read_board_temp_c():.1f} C).", flush=True)


def cell_output_path(subdir: str, eng: EngineConfig, rcfg: RunConfig, device: str) -> Path:
    wd_lbl = f"wd{int(rcfg.watchdog_ms)}" if rcfg.watchdog_enabled else "wdoff"
    fname = (
        f"{device}__{eng.model_tag}__{rcfg.controller}__{rcfg.scenario}"
        f"__b-{rcfg.band_mode}__a{rcfg.n_actions}__fs{rcfg.few_shot_k}"
        f"__{wd_lbl}__s{rcfg.seed}.csv"
    )
    return DATA_DIR / subdir / fname


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="Nome da campanha (ex: exp1_base)")
    ap.add_argument(
        "--device",
        choices=("desktop_gpu", "desktop_cpu", "pi_cpu"),
        default="desktop_gpu",
    )
    ap.add_argument("--cycles", type=int, default=None)
    ap.add_argument("--lora-path", type=str, default="")
    args = ap.parse_args()

    cells = build_campaign_cells(args.exp, args.device, args.cycles, args.lora_path)
    retriever = NeuralRetriever(top_k=2)
    loaded_llms: Dict[Tuple[str, str, int, str], Llama] = {}

    total = len(cells)
    print(f"=== ORQUESTRADOR ETAPA 8: {args.exp} ({args.device}) | {total} celulas ===", flush=True)

    for idx, (subdir, eng, rcfg) in enumerate(cells, start=1):
        out_csv = cell_output_path(subdir, eng, rcfg, args.device)
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        # Retomada automatica: pula se o arquivo final ja existe e passa no fiscal
        if out_csv.exists():
            if not audit_csv_file(out_csv, safe_default=rcfg.safe_default_action):
                print(f"[{idx}/{total}] [SKIP - JA VALIDADO] {out_csv.name}", flush=True)
                continue
            print(f"[{idx}/{total}] [RE-RUN - REPROVADO NO FISCAL] {out_csv.name}", flush=True)
            out_csv.unlink()

        wait_for_thermal_headroom(rcfg.thermal_guard_max_c)

        shared_llm = None
        if rcfg.controller != "Prefilter-only":
            llm_key = (eng.model_size, eng.quantization, eng.n_gpu_layers, eng.lora_path)
            if llm_key not in loaded_llms:
                loaded_llms.clear()  # Libera VRAM antes de subir proximo modelo
                print(f"[MODEL] Carregando {eng.model_tag} (gpu_layers={eng.n_gpu_layers})...", flush=True)
                loaded_llms[llm_key] = Llama(**eng.llama_init_kwargs())
            shared_llm = loaded_llms[llm_key]

        t0 = time.perf_counter()
        sim = SituationSimulator(rcfg)
        episode = sim.generate_episode()
        agent = AuditableAgent(
            engine_cfg=eng,
            run_cfg=rcfg,
            shared_llm=shared_llm,
            shared_retriever=retriever,
        )
        agent.reset_episode()

        tmp_csv = out_csv.with_suffix(".csv.tmp")
        with tmp_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(LOG_COLUMNS))
            writer.writeheader()
            for c_i, (sit, gt) in enumerate(episode):
                if c_i > 0 and c_i % 100 == 0:
                    wait_for_thermal_headroom(rcfg.thermal_guard_max_c)
                writer.writerow(agent.step(sit, gt))

        errs = audit_csv_file(tmp_csv, safe_default=rcfg.safe_default_action)
        if errs:
            tmp_csv.unlink(missing_ok=True)
            raise RuntimeError(f"Fiscal reprovou celula recem-gerada {out_csv.name}: {errs}")

        tmp_csv.replace(out_csv)
        dt = time.perf_counter() - t0
        print(f"[{idx}/{total}] [OK {dt:.1f}s] {out_csv.name}", flush=True)

    print(f"=== CAMPANHA {args.exp} CONCLUIDA COM 100% DE APROVACAO NO FISCAL ===", flush=True)


if __name__ == "__main__":
    main()
