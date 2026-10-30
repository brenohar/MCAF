"""Exports blind 50-query annotation sheets and computes Cohen's kappa + MRR."""
import argparse
import csv
import json
from pathlib import Path
import sys

import pandas as pd
from statsmodels.stats.inter_rater import cohens_kappa

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcaf.config import RAG_EVAL_DIR  # noqa: E402
from mcaf.rag import CORPUS_DOCUMENTS, NeuralRetriever  # noqa: E402


def export_templates(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = out_dir / "corpus_18_docs_reference.csv"
    with corpus_file.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "partition", "category", "text"])
        w.writeheader()
        w.writerows(CORPUS_DOCUMENTS)

    sheet_file = out_dir / "queries_50_annotation_template.csv"
    if not sheet_file.exists():
        with sheet_file.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "query_id",
                "author_engineer",
                "query_text",
                "annotator_1_doc_id",
                "annotator_2_doc_id",
                "adjudicated_doc_id",
                "adjudication_notes",
            ])
            for idx in range(1, 51):
                eng = "Engineer_A" if idx <= 25 else "Engineer_B"
                w.writerow([f"Q{idx:02d}", eng, "", "", "", "", ""])
    print(f"[OK] Catalogo de 18 documentos exportado em: {corpus_file}")
    print(f"[OK] Planilha de 50 consultas exportada em:  {sheet_file}")


def evaluate_annotated_csv(csv_path: Path, top_k: int = 2) -> dict:
    df = pd.read_csv(csv_path, keep_default_na=False)
    required = {
        "query_id", "query_text", "annotator_1_doc_id",
        "annotator_2_doc_id", "adjudicated_doc_id"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes: {missing}")

    labels = sorted(set(df["annotator_1_doc_id"]) | set(df["annotator_2_doc_id"]))
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    n_cat = len(labels)
    conf_mat = [[0] * n_cat for _ in range(n_cat)]
    for a1, a2 in zip(df["annotator_1_doc_id"], df["annotator_2_doc_id"]):
        if not a1 or not a2:
            raise ValueError("Planilha contem anotacoes em branco.")
        conf_mat[label_to_idx[a1]][label_to_idx[a2]] += 1

    kappa_res = cohens_kappa(conf_mat)
    kappa_val = float(kappa_res.kappa)

    retriever = NeuralRetriever(top_k=top_k)
    rr_full, rr_hot, hit_full, hit_hot = [], [], [], []
    for _, row in df.iterrows():
        gt_doc = str(row["adjudicated_doc_id"]).strip()
        q_txt = str(row["query_text"]).strip()

        ranked_full = [d_id for d_id, _, _ in retriever.rank_documents(q_txt, use_full_store=True)]
        ranked_hot = [d_id for d_id, _, _ in retriever.rank_documents(q_txt, use_full_store=False)]

        rr_full.append(1.0 / (ranked_full.index(gt_doc) + 1) if gt_doc in ranked_full else 0.0)
        rr_hot.append(1.0 / (ranked_hot.index(gt_doc) + 1) if gt_doc in ranked_hot else 0.0)
        hit_full.append(1.0 if gt_doc in ranked_full[:top_k] else 0.0)
        hit_hot.append(1.0 if gt_doc in ranked_hot[:top_k] else 0.0)

    return {
        "n_queries": len(df),
        "cohens_kappa": round(kappa_val, 4),
        "mrr_full_store": round(sum(rr_full) / len(rr_full), 4),
        "mrr_hot_store": round(sum(rr_hot) / len(rr_hot), 4),
        "recall_at_k_full": round(sum(hit_full) / len(hit_full), 4),
        "recall_at_k_hot": round(sum(hit_hot) / len(hit_hot), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-templates", action="store_true")
    ap.add_argument("--eval-csv", type=Path, default=None)
    args = ap.parse_args()

    if args.init_templates:
        export_templates(RAG_EVAL_DIR)
    if args.eval_csv:
        res = evaluate_annotated_csv(args.eval_csv)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
