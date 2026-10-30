"""Post-hoc watchdog budget sweep (250-1000 ms) and cross-device identity check."""
import argparse
import json
from pathlib import Path
import sys
from typing import List

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.stats import mean_ci95  # noqa: E402
from mcaf.shield import RuntimeShield  # noqa: E402


def _to_bool(val: object) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() == "true"


def simulate_budget_on_open_loop(
    df: pd.DataFrame, budget_ms: float, safe_default: str = "continue_operation"
) -> dict:
    shield = RuntimeShield()
    audit_shield = RuntimeShield()

    seed_dcs: List[float] = []
    tot_wd = 0
    tot_safe = 0
    tot_rows = len(df)

    for _, sub in df.groupby(["controller", "scenario", "seed"], sort=False):
        shield.reset_episode()
        audit_shield.reset_episode()
        correct = 0
        sub_ordered = sub.sort_values("cycle_idx")
        for _, row in sub_ordered.iterrows():
            sit_dict = {
                "brake_temp_c": float(row["brake_temp_c"]),
                "speed_kmh": float(row["speed_kmh"]),
                "engine_temp_c": float(row["engine_temp_c"]),
                "operator_command": str(row["operator_command"]),
                "auth_token": _to_bool(row["auth_token"]),
            }
            t_cog = (
                float(row["t_obs_ms"])
                + float(row["t_orient_ms"])
                + float(row["t_infer_ms"])
            )
            ctrl = str(row["controller"])
            use_override = ctrl in ("Prefilter-only", "Hybrid", "Hybrid+RAG")

            if str(row["route"]) == "slm" and t_cog > budget_ms:
                prop = safe_default
                tot_wd += 1
            else:
                prop = str(row["proposed_action"])

            if use_override:
                v = shield.verify_and_enforce(sit_dict, prop)
                exec_act = v.executed_action
                tot_safe += 1
            else:
                exec_act = prop
                av = audit_shield.verify_and_enforce(sit_dict, exec_act)
                if not av.intervened:
                    tot_safe += 1

            if exec_act == str(row["gt_action"]):
                correct += 1

        seed_dcs.append(100.0 * correct / max(len(sub_ordered), 1))

    dc_mean, dc_lo, dc_hi = mean_ci95(seed_dcs)
    return {
        "budget_ms": budget_ms,
        "watchdog_rate_pct": round(100.0 * tot_wd / max(tot_rows, 1), 2),
        "dc_mean_pct": round(dc_mean, 2),
        "dc_ci95": [round(dc_lo, 2), round(dc_hi, 2)],
        "scr_pct": round(100.0 * tot_safe / max(tot_rows, 1), 2),
    }


def check_cross_device_identity(csv_a: Path, csv_b: Path) -> dict:
    dfa = pd.read_csv(csv_a, keep_default_na=False).sort_values("cycle_idx")
    dfb = pd.read_csv(csv_b, keep_default_na=False).sort_values("cycle_idx")
    merged = pd.merge(
        dfa[["seed", "cycle_idx", "proposed_action", "executed_action"]],
        dfb[["seed", "cycle_idx", "proposed_action", "executed_action"]],
        on=["seed", "cycle_idx"],
        suffixes=("_a", "_b"),
    )
    prop_id = 100.0 * (merged["proposed_action_a"] == merged["proposed_action_b"]).mean()
    exec_id = 100.0 * (merged["executed_action_a"] == merged["executed_action_b"]).mean()
    return {
        "n_compared_cycles": len(merged),
        "proposed_action_identity_pct": round(float(prop_id), 3),
        "executed_action_identity_pct": round(float(exec_id), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--open-loop-dir", type=Path, default=None)
    ap.add_argument("--compare-identity", nargs=2, type=Path, default=None)
    args = ap.parse_args()

    if args.open_loop_dir:
        files = sorted(args.open_loop_dir.glob("*.csv"))
        df = pd.concat([pd.read_csv(f, keep_default_na=False) for f in files], ignore_index=True)
        results = {}
        for ctrl, sub in df.groupby("controller"):
            results[str(ctrl)] = [
                simulate_budget_on_open_loop(sub, b)
                for b in (250.0, 350.0, 500.0, 750.0, 1000.0)
            ]
        print(json.dumps(results, indent=2))

    if args.compare_identity:
        res = check_cross_device_identity(args.compare_identity[0], args.compare_identity[1])
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
