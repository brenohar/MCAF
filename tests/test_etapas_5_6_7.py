"""Automated verification gates for Stages 5, 6, and 7."""
import csv
from pathlib import Path
import subprocess

import pandas as pd

from analysis.eval_rag_kappa import evaluate_annotated_csv
from analysis.sanity_check import audit_csv_file
from analysis.stats import paired_mcnemar_test
from mcaf.agent import AuditableAgent
from mcaf.config import BASE_DIR, DIRECTIVE_EQUIPMENTS, EngineConfig, LOG_COLUMNS, RunConfig
from mcaf.rag import CORPUS_DOCUMENTS, NeuralRetriever
from mcaf.simulator import SituationSimulator


def test_etapa5_directive_recall_above_95pct_both_stores():
    """Para equipamentos com diretiva, devolve a diretiva em >= 95% nas duas bases."""
    retriever = NeuralRetriever(top_k=2)
    assert len(retriever.full_docs) == 18
    assert len(retriever.hot_docs) == 11

    eq_to_doc = {
        "EQ-02": "D01",
        "EQ-05": "D02",
        "EQ-08": "D03",
        "EQ-11": "D04",
        "EQ-15": "D05",
        "EQ-19": "D06",
    }

    sim = SituationSimulator(RunConfig(seed=42, n_cycles=300))
    hits_full, hits_hot, total_dir_cycles = 0, 0, 0
    for idx in range(300):
        sit = sim.next_situation(idx)
        if sit.equipment_id in DIRECTIVE_EQUIPMENTS:
            total_dir_cycles += 1
            q = retriever.build_cycle_query(sit.to_dict())
            expected_doc = eq_to_doc[sit.equipment_id]
            top2_full = [d for d, _, _ in retriever.rank_documents(q, use_full_store=True)[:2]]
            top2_hot = [d for d, _, _ in retriever.rank_documents(q, use_full_store=False)[:2]]
            if expected_doc in top2_full:
                hits_full += 1
            if expected_doc in top2_hot:
                hits_hot += 1

    assert total_dir_cycles >= 50
    assert (hits_full / total_dir_cycles) >= 0.95
    assert (hits_hot / total_dir_cycles) >= 0.95


def test_etapa5_kappa_and_annotation_evaluator(tmp_path: Path):
    """Valida calculo do indice kappa e metricas em planilha de 50 consultas."""
    fixture = tmp_path / "fixture_50.csv"
    docs = list(CORPUS_DOCUMENTS)
    with fixture.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "query_id", "author_engineer", "query_text",
            "annotator_1_doc_id", "annotator_2_doc_id",
            "adjudicated_doc_id", "adjudication_notes"
        ])
        for i in range(50):
            d = docs[i % len(docs)]
            a2 = d["id"] if i < 47 else docs[(i + 1) % len(docs)]["id"]
            w.writerow([
                f"Q{i+1:02d}",
                "Eng_A" if i < 25 else "Eng_B",
                d["text"],
                d["id"],
                a2,
                d["id"],
                "resolved" if a2 != d["id"] else "",
            ])
    res = evaluate_annotated_csv(fixture, top_k=2)
    assert res["n_queries"] == 50
    assert 0.85 <= res["cohens_kappa"] <= 1.0
    assert res["mrr_full_store"] >= 0.90


def test_etapa6_zero_cache_and_short_watchdog_triggers():
    """Garante zero 'cache' em mcaf/ e disparo do watchdog gerando acao padrao."""
    proc = subprocess.run(
        ["grep", "-rni", "cache", str(BASE_DIR / "mcaf"), "--exclude-dir=__pycache__"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1 and proc.stdout == "", f"Resquicio encontrado:\n{proc.stdout}"

    eng = EngineConfig(model_size="0.5B", quantization="Q4_K_M", n_threads=4, n_gpu_layers=0)
    rcfg = RunConfig(
        controller="Pure-SLM",
        scenario="isolated",
        seed=42,
        n_cycles=3,
        watchdog_ms=0.01,
        watchdog_enabled=True,
    )
    sim = SituationSimulator(rcfg)
    agent = AuditableAgent(engine_cfg=eng, run_cfg=rcfg)
    for sit, gt in sim.generate_episode():
        row = agent.step(sit, gt)
        assert len(row) == 34
        assert row["watchdog_triggered"] is True
        assert row["executed_action"] == rcfg.safe_default_action


def test_etapa7_fiscal_rejects_tampered_csv_and_mcnemar_handles_ties(tmp_path: Path):
    """Fiscal acusa arquivo adulterado sem edita-lo; McNemar trata 100% de identidade."""
    valid_csv = tmp_path / "valid_fixture.csv"
    eng = EngineConfig(model_size="0.5B", quantization="Q4_K_M", n_threads=2, n_gpu_layers=0)
    rcfg = RunConfig(controller="Prefilter-only", scenario="isolated", seed=42, n_cycles=10)
    sim = SituationSimulator(rcfg)
    agent = AuditableAgent(engine_cfg=eng, run_cfg=rcfg)
    with valid_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(LOG_COLUMNS))
        writer.writeheader()
        for sit, gt in sim.generate_episode():
            writer.writerow(agent.step(sit, gt))

    assert audit_csv_file(valid_csv) == []

    df = pd.read_csv(valid_csv, keep_default_na=False)
    tampered = tmp_path / "tampered.csv"
    df_bad = df.copy()
    df_bad.loc[0, "t_dispatch_ms"] = 10.0
    df_bad.loc[0, "t_total_ms"] = 0.01
    df_bad.to_csv(tampered, index=False)

    before_bytes = tampered.read_bytes()
    errs = audit_csv_file(tampered)
    after_bytes = tampered.read_bytes()

    assert len(errs) >= 1
    assert before_bytes == after_bytes, "Fiscal jamais pode alterar o arquivo auditado!"

    res_tie = paired_mcnemar_test([True, False, True], [True, False, True])
    assert res_tie["status"] == "identical_decisions_100pct"
    assert res_tie["p_value"] == 1.0
