"""Statistical inference engine: 95% CIs and paired exact McNemar test (Stage 7.3)."""
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy import stats


def mean_ci95(values: Sequence[float]) -> Tuple[float, float, float]:
    """Returns (mean, ci_low, ci_high) using Student's t distribution."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0, 0.0)
    mean = float(np.mean(arr))
    if n == 1:
        return (mean, mean, mean)
    sd = float(np.std(arr, ddof=1))
    if sd == 0.0:
        return (mean, mean, mean)
    half_w = float(stats.t.ppf(0.975, df=n - 1) * sd / np.sqrt(n))
    return (mean, mean - half_w, mean + half_w)


def paired_mcnemar_test(
    correct_a: Sequence[bool], correct_b: Sequence[bool]
) -> Dict[str, object]:
    """Exact paired McNemar test on cycle-aligned correctness vectors."""
    ya = np.asarray(correct_a, dtype=bool)
    yb = np.asarray(correct_b, dtype=bool)
    if ya.shape != yb.shape:
        raise ValueError(f"Vetores despareados: {ya.shape} vs {yb.shape}")

    n_total = int(len(ya))
    n10 = int(np.sum(ya & (~yb)))
    n01 = int(np.sum((~ya) & yb))
    discordant = n10 + n01
    identical_pct = 100.0 * float(np.sum(ya == yb)) / max(n_total, 1)

    if discordant == 0:
        return {
            "n_total": n_total,
            "n10": 0,
            "n01": 0,
            "p_value": 1.0,
            "identical_pct": round(identical_pct, 2),
            "status": "identical_decisions_100pct",
        }

    p_val = float(
        stats.binomtest(min(n10, n01), n=discordant, p=0.5, alternative="two-sided").pvalue
    )
    return {
        "n_total": n_total,
        "n10": n10,
        "n01": n01,
        "p_value": p_val,
        "identical_pct": round(identical_pct, 2),
        "status": "evaluated",
    }
