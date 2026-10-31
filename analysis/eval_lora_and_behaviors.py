"""Read-only statistical readout for exp3_lora and non-critical action distributions."""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.sanity_check import audit_directory  # noqa: E402
from analysis.stats import mean_ci95, paired_mcnemar_test  # noqa: E402
from mcaf.config import DATA_DIR  # noqa: E402


def _load(files) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(f, keep_default_na=False) for f in files], ignore_index=True)
    df["is_correct"] = df["executed_action"] == df["gt_action"]
    df["safety_bool"] = df["safety_ok"].map(lambda v: str(v).strip().lower() == "true")
    return df


def _p_str(sub: pd.DataFrame, ref: pd.DataFrame) -> str:
    mg = pd.merge(
        sub[["seed", "cycle_idx", "is_correct"]],
        ref[["seed", "cycle_idx", "is_correct"]],
        on=["seed", "cycle_idx"],
        suffixes=("_a", "_b"),
    )
    res = paired_mcnemar_test(mg["is_correct_a"].tolist(), mg["is_correct_b"].tolist())
    return "1.0000 (100% id)" if res["status"] == "identical_decisions_100pct" else f"{res['p_value']:.4e}"


def main() -> None:
    errs = audit_directory(DATA_DIR / "exp3_lora")
    if errs:
        print("ERRO FISCAL:", errs)
        sys.exit(1)
    print("=== 1. FISCAL VERDE EM exp3_lora (15 CSVs | 15.000 CICLOS) ===\n")

    df_lora = _load(sorted((DATA_DIR / "exp3_lora").glob("*.csv")))
    df_base = _load(sorted((DATA_DIR / "exp1_base").glob("*__isolated__*.csv")))
    pf_iso = df_base[df_base["controller"] == "Prefilter-only"]

    print("=== 2. DESEMPENHO Qwen2.5-0.5B-Q4_K_M+LoRA (isolated, 0-shot, 6 acoes) ===")
    for ctrl, sub in df_lora.groupby("controller"):
        seed_dc = (sub.groupby("seed")["is_correct"].mean() * 100.0).tolist()
        m, lo, hi = mean_ci95(seed_dc)
        scr = sub["safety_bool"].mean() * 100.0
        zs_ref = df_base[df_base["controller"] == ctrl]
        p_pf = _p_str(sub, pf_iso)
        p_zs = _p_str(sub, zs_ref)
        print(
            f"  {ctrl:12s} | DC={m:5.1f}% [{lo:5.1f}, {hi:5.1f}] | "
            f"SCR={scr:5.1f}% | p_vs_PF={p_pf:12s} | p_vs_0shot={p_zs}"
        )

    hyb_lora = df_lora[df_lora["controller"] == "Hybrid"]
    rag_lora = df_lora[df_lora["controller"] == "Hybrid+RAG"]
    print(f"\n  [Teste Causal RAG sob LoRA] Hybrid+RAG vs Hybrid (sem RAG): p={_p_str(rag_lora, hyb_lora)}")

    df_scale = _load(sorted((DATA_DIR / "exp2_scale").glob("*Q4_K_M*.csv")))
    df_comb = pd.concat([df_scale, df_lora], ignore_index=True)
    print("\n=== 3. DISTRIBUICAO DE ACOES PROPOSTAS EM ROTA SLM (NAO-CRITICOS) ===")
    for (mtag, ctrl), sub in df_comb[df_comb["route"] == "slm"].groupby(["model_tag", "controller"]):
        dist = (sub["proposed_action"].value_counts(normalize=True) * 100.0).round(1).to_dict()
        print(f"  {mtag:27s} | {ctrl:10s} | n={len(sub):4d} | {dist}")


if __name__ == "__main__":
    main()
