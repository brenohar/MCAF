"""Read-only breakdown of action distributions and failure modes across scales."""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mcaf.config import DATA_DIR  # noqa: E402


def main() -> None:
    files = sorted((DATA_DIR / "exp2_scale").glob("*Q4_K_M*.csv"))
    df = pd.concat([pd.read_csv(f, keep_default_na=False) for f in files], ignore_index=True)

    print("=== DISTRIBUICAO DE ACOES PROPOSTAS EM CICLOS NAO-CRITICOS (EXP 2, Q4_K_M) ===")
    non_crit = df[df["route"] == "slm"]
    for (mtag, ctrl), sub in non_crit.groupby(["model_tag", "controller"]):
        dist = (sub["proposed_action"].value_counts(normalize=True) * 100.0).round(1).to_dict()
        print(f"  {mtag:22s} | {ctrl:10s} | n={len(sub):4d} | {dist}")


if __name__ == "__main__":
    main()
