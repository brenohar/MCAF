"""Read-only data integrity guardian (Stage 7.1). NEVER modifies input files."""
import argparse
from pathlib import Path
import sys
from typing import List

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcaf.config import DIRECTIVE_EQUIPMENTS, LOG_COLUMNS  # noqa: E402
from mcaf.simulator import (  # noqa: E402
    CycleSituation,
    compute_ground_truth,
    deterministic_prefilter,
)


def _to_bool(val: object) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() == "true"


def audit_csv_file(csv_path: Path, safe_default: str = "continue_operation") -> List[str]:
    errors: List[str] = []
    df = pd.read_csv(csv_path, keep_default_na=False)

    if tuple(df.columns) != LOG_COLUMNS:
        errors.append(
            f"[{csv_path.name}] Esquema invalido ({len(df.columns)} cols vs 34 esperadas)."
        )
        return errors

    if df.empty:
        errors.append(f"[{csv_path.name}] Arquivo vazio.")
        return errors

    t_parts = (
        df["t_obs_ms"].astype(float)
        + df["t_orient_ms"].astype(float)
        + df["t_infer_ms"].astype(float)
        + df["t_shield_ms"].astype(float)
        + df["t_dispatch_ms"].astype(float)
    )
    t_tot = df["t_total_ms"].astype(float)
    bad_time = df[t_tot + 1e-4 < t_parts]
    if not bad_time.empty:
        errors.append(
            f"[{csv_path.name}] {len(bad_time)} ciclos violam t_total_ms >= soma das partes."
        )

    rem_mask = df["remote_queried"].map(_to_bool)
    net_lat = df["net_latency_ms"].astype(float)
    bad_net = df[(~rem_mask) & (net_lat != 0.0)]
    if not bad_net.empty:
        errors.append(
            f"[{csv_path.name}] {len(bad_net)} ciclos sem consulta remota tem net_latency_ms != 0."
        )

    wd_mask = df["watchdog_triggered"].map(_to_bool)
    bad_wd = df[
        wd_mask
        & (
            (df["proposed_action"] != safe_default)
            | ((df["executed_action"] != safe_default) & (df["blocked_by"] == ""))
        )
    ]
    if not bad_wd.empty:
        errors.append(
            f"[{csv_path.name}] {len(bad_wd)} cortes de watchdog nao geraram '{safe_default}'."
        )

    gt_mismatch = 0
    pf_crit_mismatch = 0
    for _, row in df.iterrows():
        eq_id = str(row["equipment_id"])
        sit = CycleSituation(
            cycle_idx=int(row["cycle_idx"]),
            equipment_id=eq_id,
            has_directive=(eq_id in DIRECTIVE_EQUIPMENTS),
            engine_temp_c=float(row["engine_temp_c"]),
            brake_temp_c=float(row["brake_temp_c"]),
            speed_kmh=float(row["speed_kmh"]),
            operator_command=str(row["operator_command"]),
            auth_token=_to_bool(row["auth_token"]),
            ma5_current_c=0.0,
            ma5_prev_c=0.0,
            ma5_rising=_to_bool(row["ma5_rising"]),
            is_critical_gt=False,
        )
        recomp_gt = compute_ground_truth(sit, band_mode=str(row["band_mode"]))
        if recomp_gt != str(row["gt_action"]):
            gt_mismatch += 1

        pf_act = deterministic_prefilter(sit)
        if pf_act is not None:
            if pf_act != recomp_gt:
                pf_crit_mismatch += 1
            if str(row["controller"]) in ("Prefilter-only", "Hybrid", "Hybrid+RAG"):
                if str(row["executed_action"]) != recomp_gt:
                    pf_crit_mismatch += 1

    if gt_mismatch > 0:
        errors.append(
            f"[{csv_path.name}] {gt_mismatch} linhas divergem na recomputacao do gabarito."
        )
    if pf_crit_mismatch > 0:
        errors.append(
            f"[{csv_path.name}] {pf_crit_mismatch} falhas do filtro deterministico em ciclos criticos."
        )

    return errors


def audit_directory(target_dir: Path) -> List[str]:
    csv_files = sorted(target_dir.glob("*.csv"))
    if not csv_files:
        return [f"Nenhum arquivo CSV encontrado em {target_dir}"]

    all_errors: List[str] = []
    frames = []
    for f in csv_files:
        errs = audit_csv_file(f)
        all_errors.extend(errs)
        if not errs:
            frames.append(pd.read_csv(f, keep_default_na=False))

    if frames:
        merged = pd.concat(frames, ignore_index=True)
        grp = (
            merged.groupby(["model_tag", "controller", "scenario", "band_mode", "seed"])
            .size()
            .reset_index(name="n_cycles")
        )
        for cfg_key, sub in grp.groupby(["model_tag", "controller", "scenario", "band_mode"]):
            unique_counts = sub["n_cycles"].unique()
            if len(unique_counts) > 1:
                all_errors.append(
                    f"Contagem desigual de ciclos entre sementes em {cfg_key}: {unique_counts}"
                )

    return all_errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=Path, help="Arquivo CSV ou diretorio de CSVs")
    args = ap.parse_args()

    errors = (
        audit_csv_file(args.target)
        if args.target.is_file()
        else audit_directory(args.target)
    )
    if errors:
        print("=== FISCAL REPROVOU OS DADOS ===")
        for e in errors:
            print("  [ERRO]", e)
        sys.exit(1)

    print(f"=== FISCAL VERDE: 100% DOS TESTES DE INTEGRIDADE APROVADOS ({args.target}) ===")
    sys.exit(0)


if __name__ == "__main__":
    main()
