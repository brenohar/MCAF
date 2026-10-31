"""Recursive read-only fiscal check and statistical readout across all desktop campaigns."""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.sanity_check import audit_directory  # noqa: E402
from analysis.stats import mean_ci95, paired_mcnemar_test  # noqa: E402
from mcaf.config import DATA_DIR  # noqa: E402


def _to_bool(val: object) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() == "true"


def load_dir_df(subdir: Path) -> pd.DataFrame:
    files = sorted(subdir.rglob("*.csv"))
    frames = []
    for f in files:
        df = pd.read_csv(f, keep_default_na=False)
        df["subtag"] = str(f.parent.relative_to(DATA_DIR))
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["is_correct"] = out["executed_action"] == out["gt_action"]
    out["safety_bool"] = out["safety_ok"].map(_to_bool)
    return out


def mcnemar_vs_ref(sub: pd.DataFrame, ref: pd.DataFrame) -> str:
    merged = pd.merge(
        sub[["seed", "cycle_idx", "gt_action", "is_correct"]],
        ref[["seed", "cycle_idx", "gt_action", "is_correct"]],
        on=["seed", "cycle_idx"],
        suffixes=("_a", "_b"),
    )
    if merged.empty:
        return "N/A"
    assert (merged["gt_action_a"] == merged["gt_action_b"]).all(), "Quebra de pareamento!"
    res = paired_mcnemar_test(merged["is_correct_a"].tolist(), merged["is_correct_b"].tolist())
    if res["status"] == "identical_decisions_100pct":
        return "1.0000 (100% id)"
    return f"{res['p_value']:.4e}"


def main() -> None:
    leaf_dirs = sorted({p.parent for p in DATA_DIR.rglob("*.csv")})
    total_csvs = sum(len(list(d.glob("*.csv"))) for d in leaf_dirs)
    print(f"=== 1. AUDITORIA RECURSIVA: {len(leaf_dirs)} diretorios | {total_csvs} CSVs ===")

    all_errors = []
    for d in leaf_dirs:
        errs = audit_directory(d)
        all_errors.extend(errs)

    if all_errors:
        print("FALHA NO FISCAL RECURSIVO:")
        for e in all_errors:
            print(" ", e)
        sys.exit(1)
    print("[OK] Fiscal recursivo verde em 100% dos diretorios e sementes balanceadas.\n")

    df_base = load_dir_df(DATA_DIR / "exp1_base")
    print("=== 2. EXP 1: BASE NOVA (0.5B Q4_K_M, 0-shot, 6 acoes) ===")
    for (ctrl, scen), sub in df_base.groupby(["controller", "scenario"]):
        seed_dc = (sub.groupby("seed")["is_correct"].mean() * 100.0).tolist()
        m, lo, hi = mean_ci95(seed_dc)
        scr = sub["safety_bool"].mean() * 100.0
        ref = df_base[(df_base["controller"] == "Prefilter-only") & (df_base["scenario"] == scen)]
        p_str = "---" if ctrl == "Prefilter-only" else mcnemar_vs_ref(sub, ref)
        print(f"  {ctrl:15s} | {scen:10s} | DC={m:5.1f}% [{lo:5.1f}, {hi:5.1f}] | SCR={scr:5.1f}% | p={p_str}")

    ref_iso = df_base[(df_base["controller"] == "Prefilter-only") & (df_base["scenario"] == "isolated")]

    df_scale = load_dir_df(DATA_DIR / "exp2_scale")
    print("\n=== 3. EXP 2: ESCALA E QUANTIZACAO (isolated, 0-shot) ===")
    for (mtag, ctrl), sub in df_scale.groupby(["model_tag", "controller"]):
        seed_dc = (sub.groupby("seed")["is_correct"].mean() * 100.0).tolist()
        m, lo, hi = mean_ci95(seed_dc)
        scr = sub["safety_bool"].mean() * 100.0
        p_str = mcnemar_vs_ref(sub, ref_iso)
        print(f"  {mtag:22s} | {ctrl:10s} | DC={m:5.1f}% [{lo:5.1f}, {hi:5.1f}] | SCR={scr:5.1f}% | p_vs_PF={p_str}")

    print("\n  [Identidade Q4_K_M vs Q8_0 em propostas brutas]:")
    for size in ("0.5B", "1.5B", "3B"):
        for ctrl in ("Pure-SLM", "Hybrid+RAG"):
            q4 = df_scale[(df_scale["model_tag"] == f"Qwen2.5-{size}-Q4_K_M") & (df_scale["controller"] == ctrl)]
            q8 = df_scale[(df_scale["model_tag"] == f"Qwen2.5-{size}-Q8_0") & (df_scale["controller"] == ctrl)]
            mg = pd.merge(q4[["seed", "cycle_idx", "proposed_action"]], q8[["seed", "cycle_idx", "proposed_action"]], on=["seed", "cycle_idx"])
            id_pct = (mg["proposed_action_x"] == mg["proposed_action_y"]).mean() * 100.0
            print(f"    Qwen2.5-{size:4s} | {ctrl:10s} | concordancia Q4 vs Q8 = {id_pct:5.2f}%")

    df_fs = load_dir_df(DATA_DIR / "exp3_fewshot")
    print("\n=== 4. EXP 3a: ADAPTACAO DE PROMPT (few-shot & GBNF) ===")
    for (stag, mtag, ctrl), sub in df_fs.groupby(["subtag", "model_tag", "controller"]):
        seed_dc = (sub.groupby("seed")["is_correct"].mean() * 100.0).tolist()
        m, lo, hi = mean_ci95(seed_dc)
        scr = sub["safety_bool"].mean() * 100.0
        p_str = mcnemar_vs_ref(sub, ref_iso)
        print(f"  {stag:25s} | {mtag:20s} | {ctrl:10s} | DC={m:5.1f}% [{lo:5.1f}, {hi:5.1f}] | SCR={scr:5.1f}% | p_vs_PF={p_str}")

    df_rich = load_dir_df(DATA_DIR / "exp6_richness")
    print("\n=== 5. EXP 6: RIQUEZA DA TAREFA E VALOR CAUSAL DO RAG ===")
    for (stag, mtag, ctrl), sub in df_rich.groupby(["subtag", "model_tag", "controller"]):
        seed_dc = (sub.groupby("seed")["is_correct"].mean() * 100.0).tolist()
        m, lo, hi = mean_ci95(seed_dc)
        b_mode = "trend" if "act4_trend" in stag else "directive"
        pf_tag = "exp6_richness/act4_trend_fs0" if b_mode == "trend" else "exp6_richness/act6_directive_fs0"
        ref_pf = df_rich[(df_rich["subtag"] == pf_tag) & (df_rich["controller"] == "Prefilter-only")]
        p_pf = "---" if ctrl == "Prefilter-only" else mcnemar_vs_ref(sub, ref_pf)
        p_rag = "---"
        if ctrl == "Hybrid+RAG":
            ref_hyb = df_rich[(df_rich["subtag"] == stag) & (df_rich["model_tag"] == mtag) & (df_rich["controller"] == "Hybrid")]
            p_rag = mcnemar_vs_ref(sub, ref_hyb)
        print(f"  {stag:34s} | {mtag:20s} | {ctrl:14s} | DC={m:5.1f}% [{lo:5.1f}, {hi:5.1f}] | p_PF={p_pf:12s} | p_vs_NoRAG={p_rag}")

    df_sens = load_dir_df(DATA_DIR / "exp5_sensitivity")
    print("\n=== 6. EXP 5: SENSIBILIDADE DO AMBIENTE (12 varreduras) ===")
    for (stag, ctrl), sub in df_sens.groupby(["subtag", "controller"]):
        seed_dc = (sub.groupby("seed")["is_correct"].mean() * 100.0).tolist()
        m, lo, hi = mean_ci95(seed_dc)
        scr = sub["safety_bool"].mean() * 100.0
        print(f"  {stag:28s} | {ctrl:14s} | DC={m:5.1f}% [{lo:5.1f}, {hi:5.1f}] | SCR={scr:5.1f}%")


if __name__ == "__main__":
    main()
