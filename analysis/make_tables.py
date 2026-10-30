"""Generates LaTeX tables and numerical macros directly from config.py and raw CSVs."""
import argparse
import json
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.stats import mean_ci95, paired_mcnemar_test  # noqa: E402
from mcaf.config import (  # noqa: E402
    BRAKE_TEMP_MAX_C,
    ENGINE_TEMP_CRIT_C,
    EngineConfig,
    PAPER_DIR,
    RunConfig,
    SPECS_DIR,
    SPEED_MAX_KMH,
)


def _tex_safe_macro_name(raw: str) -> str:
    return re.sub(r"[^A-Za-z]", "", raw)


def _to_bool(val: object) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() == "true"


def generate_config_table(out_dir: Path) -> str:
    eng = EngineConfig()
    run = RunConfig()
    shield_meta = json.loads((SPECS_DIR / "compiled_shield.json").read_text(encoding="utf-8"))

    tex = "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Default Experimental Configuration (Auto-Generated from \texttt{config.py})}",
        r"\label{tab:config}",
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"\textbf{Parameter} & \textbf{Value} \\",
        r"\midrule",
        f"Context Window ($n_{{\\mathrm{{ctx}}}}$) & {eng.n_ctx} tokens \\\\",
        f"Max Generated Tokens & {eng.max_tokens} tokens \\\\",
        f"Sampling Temperature & {eng.temperature:.1f} \\\\",
        f"Supervisory Watchdog Budget & {run.watchdog_ms:.0f} ms \\\\",
        f"Critical Thresholds ($T_{{\\mathrm{{brake}}}}, v, T_{{\\mathrm{{eng}}}}$) & "
        f"$>{BRAKE_TEMP_MAX_C:.0f}\\,^\\circ$C, $>{SPEED_MAX_KMH:.0f}$ km/h, $>{ENGINE_TEMP_CRIT_C:.0f}\\,^\\circ$C \\\\",
        f"Compiled $\\mathrm{{LTL}}_f$ Shield Totals & "
        f"{shield_meta['sum_individual_states']} states, {shield_meta['sum_symbolic_transitions']} symbolic edges \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    (out_dir / "table_config.tex").write_text(tex, encoding="utf-8")
    return tex


def generate_results_and_macros(data_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    generate_config_table(tables_dir)

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {data_dir}")

    df = pd.concat([pd.read_csv(f, keep_default_na=False) for f in csv_files], ignore_index=True)
    df["is_correct"] = df["executed_action"] == df["gt_action"]
    df["safety_bool"] = df["safety_ok"].map(_to_bool)
    df["wd_bool"] = df["watchdog_triggered"].map(_to_bool)
    df["t_tot_f"] = df["t_total_ms"].astype(float)
    df["dmr_raw"] = df["t_tot_f"] > 500.0

    shield_meta = json.loads((SPECS_DIR / "compiled_shield.json").read_text(encoding="utf-8"))
    macros = [
        "% Auto-generated LaTeX macros - DO NOT EDIT BY HAND",
        f"\\newcommand{{\\ShieldStatesTotal}}{{{shield_meta['sum_individual_states']}}}",
        f"\\newcommand{{\\ShieldEdgesTotal}}{{{shield_meta['sum_symbolic_transitions']}}}",
        f"\\newcommand{{\\ShieldProductBound}}{{{shield_meta['upper_bound_product_states']}}}",
    ]

    rows_tex = []
    baseline_ctrl = "Prefilter-only"

    for (ctrl, scen), sub in df.groupby(["controller", "scenario"]):
        seed_dc = sub.groupby("seed")["is_correct"].mean() * 100.0
        dc_mean, dc_lo, dc_hi = mean_ci95(seed_dc.tolist())
        scr_pct = float(sub["safety_bool"].mean() * 100.0)
        lat_mean = float(sub["t_tot_f"].mean())
        lat_p95 = float(sub["t_tot_f"].quantile(0.95))
        dmr_pct = float(sub["dmr_raw"].mean() * 100.0)
        wd_pct = float(sub["wd_bool"].mean() * 100.0)

        p_str = "---"
        if ctrl != baseline_ctrl:
            base_sub = df[(df["controller"] == baseline_ctrl) & (df["scenario"] == scen)]
            if not base_sub.empty:
                merged_pair = pd.merge(
                    sub[["seed", "cycle_idx", "is_correct"]],
                    base_sub[["seed", "cycle_idx", "is_correct"]],
                    on=["seed", "cycle_idx"],
                    suffixes=("_a", "_b"),
                )
                if not merged_pair.empty:
                    mc = paired_mcnemar_test(
                        merged_pair["is_correct_a"].tolist(),
                        merged_pair["is_correct_b"].tolist(),
                    )
                    p_str = f"{mc['p_value']:.4f}"

        rows_tex.append(
            f"{ctrl} & {scen} & {dc_mean:.1f} [{dc_lo:.1f}, {dc_hi:.1f}] & "
            f"{scr_pct:.1f} & {lat_mean:.2f} & {lat_p95:.2f} & "
            f"{dmr_pct:.1f} & {wd_pct:.1f} & {p_str} \\\\"
        )

        tag = _tex_safe_macro_name(f"{ctrl}{scen.capitalize()}")
        macros.append(f"\\newcommand{{\\DC{tag}}}{{{dc_mean:.1f}\\%}}")
        macros.append(f"\\newcommand{{\\SCR{tag}}}{{{scr_pct:.1f}\\%}}")
        macros.append(f"\\newcommand{{\\DMR{tag}}}{{{dmr_pct:.1f}\\%}}")
        macros.append(f"\\newcommand{{\\LatMean{tag}}}{{{lat_mean:.2f}}}")

    summary_tex = "\n".join([
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Controller Performance Summary and Paired McNemar Tests vs.\ Prefilter-Only}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"\textbf{Controller} & \textbf{Scenario} & \textbf{DC \% [95\% CI]} & "
        r"\textbf{SCR \%} & \textbf{$L_{\mathrm{tot}}$ $\mu$ (ms)} & "
        r"\textbf{$L_{\mathrm{tot}}$ p95 (ms)} & \textbf{DMR \%} & "
        r"\textbf{Watchdog \%} & \textbf{McNemar $p$} \\",
        r"\midrule",
        *rows_tex,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])
    (tables_dir / "table_results_summary.tex").write_text(summary_tex, encoding="utf-8")
    (out_dir / "macros_generated.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    print(f"[OK] Tabelas e macros geradas em: {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=PAPER_DIR)
    args = ap.parse_args()
    generate_results_and_macros(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
