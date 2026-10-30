"""Stage 7.4 Smoke Test: 50 cycles of Prefilter-only and Hybrid+RAG + full audit."""
import csv
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcaf.agent import AuditableAgent  # noqa: E402
from mcaf.config import EngineConfig, LOG_COLUMNS, PAPER_DIR, RunConfig  # noqa: E402
from mcaf.simulator import SituationSimulator  # noqa: E402


def run_cell(engine_cfg: EngineConfig, run_cfg: RunConfig, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sim = SituationSimulator(run_cfg)
    episode = sim.generate_episode()
    agent = AuditableAgent(engine_cfg=engine_cfg, run_cfg=run_cfg)
    agent.reset_episode()

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(LOG_COLUMNS))
        writer.writeheader()
        for sit, gt in episode:
            row = agent.step(sit, gt)
            writer.writerow(row)


def main() -> None:
    smoke_dir = ROOT / "data" / "raw" / "smoke"
    if smoke_dir.exists():
        for old in smoke_dir.glob("*.csv"):
            old.unlink()

    eng_cfg = EngineConfig(
        model_size="0.5B", quantization="Q4_K_M", n_threads=8, n_gpu_layers=-1
    )

    for ctrl in ("Prefilter-only", "Hybrid+RAG"):
        rcfg = RunConfig(
            controller=ctrl,
            scenario="isolated",
            seed=42,
            n_cycles=50,
            watchdog_ms=500.0,
            watchdog_enabled=True,
            few_shot_k=1,
        )
        out_file = smoke_dir / f"smoke_{ctrl}_isolated_s42.csv"
        print(f"[SMOKE] Rodando 50 ciclos para {ctrl}...")
        run_cell(eng_cfg, rcfg, out_file)

    print("[SMOKE] Executando fiscal sanity_check.py...")
    subprocess.run([sys.executable, "analysis/sanity_check.py", str(smoke_dir)], check=True)

    print("[SMOKE] Executando gerador make_tables.py...")
    subprocess.run(
        [
            sys.executable,
            "analysis/make_tables.py",
            "--data-dir",
            str(smoke_dir),
            "--out-dir",
            str(PAPER_DIR / "smoke"),
        ],
        check=True,
    )

    sample_csv = smoke_dir / "smoke_Hybrid+RAG_isolated_s42.csv"
    with sample_csv.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        slm_rows = [r for r in reader if r["route"] == "slm"]
        sample_row = slm_rows[0] if slm_rows else reader[0]
    print("\n=== INSPECAO A OLHO DE UMA LINHA COMPLETA (34 COLUNAS) ===")
    for k, v in sample_row.items():
        print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    main()
