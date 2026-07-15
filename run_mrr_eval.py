# src/run_mrr_eval.py — MRR com golden set adversarial (QP3 — nota 10)
import sys, csv
from datetime import datetime
sys.path.append('src')
from eaco_rag import EACO_RAG

# ─────────────────────────────────────────────────────────────────────────────
# GOLDEN SET — 3 categorias:
#   A) Positivas diretas     → RR esperado = 1.0 (âncoras do domínio)
#   B) Adversariais semânticas → RR esperado < 1.0 (ambíguas, cruzadas)
#   C) Fora do domínio       → RR esperado = 0.0 (não deve retornar nada relevante)
# ─────────────────────────────────────────────────────────────────────────────
GOLDEN_SET = [
    # ── A) Positivas diretas (4 queries) ─────────────────────────────────────
    {"query": "motor 115C acima do limite requer resfriamento imediato",
     "relevant": "motor", "category": "direta"},

    {"query": "freio 130C descida longa superaquecimento critico",
     "relevant": "freio", "category": "direta"},

    {"query": "procedimento de parada de emergencia veiculo",
     "relevant": "emergencia", "category": "direta"},

    {"query": "motor 108C subindo rampa carga maxima",
     "relevant": "motor", "category": "direta"},

    {"query": "temperatura freio 95C operacao normal monitorar",
     "relevant": "freio", "category": "direta"},

    {"query": "velocidade acima do limite rampa inclinada acionar freio",
     "relevant": "freio", "category": "direta"},

    {"query": "parada de emergencia acionamento imediato veiculo mineracao",
     "relevant": "emergencia", "category": "direta"},

    {"query": "motor 100C carga normal continuar operacao",
     "relevant": "motor", "category": "direta"},


    # ── B) Adversariais semânticas (4 queries) ────────────────────────────────
    # B1 — temperatura ambígua: motor e freio ambos elevados
    {"query": "temperatura elevada em multiplos sistemas 105C",
     "relevant": "motor", "category": "adversarial"},

    # B2 — consulta invertida: pede o que NÃO fazer
    {"query": "quando nao devo acionar o sistema de resfriamento",
     "relevant": "motor", "category": "adversarial"},

    # B3 — linguagem técnica diferente do índice
    {"query": "brake overheat descida prolongada reduce speed",
     "relevant": "freio", "category": "adversarial"},

    # B4 — consulta parcial sem contexto de sensor
    {"query": "acionar resfriamento",
     "relevant": "motor", "category": "adversarial"},

    {"query": "o que fazer quando o veiculo esta com multiplas alertas simultaneas",
     "relevant": "emergencia", "category": "adversarial"},

    {"query": "high brake temperature mining truck downhill operation",
     "relevant": "freio", "category": "adversarial"},

    {"query": "cooling system activation threshold exceeded",
     "relevant": "motor", "category": "adversarial"},

    {"query": "parar ou continuar operacao com temperatura limite",
     "relevant": "motor", "category": "adversarial"},

    # ── C) Fora do domínio (2 queries) ───────────────────────────────────────
    # C1 — domínio completamente diferente
    {"query": "qual o protocolo de comunicacao modbus tcp ip",
     "relevant": "modbus", "category": "fora_dominio"},

    # C2 — consulta vazia de contexto operacional
    {"query": "status do sistema operacional linux kernel version",
     "relevant": "linux", "category": "fora_dominio"},

    {"query": "configurar endereco ip servidor dhcp rede industrial",
     "relevant": "dhcp", "category": "fora_dominio"},

    {"query": "manutencao preventiva calibracao sensor pressao hidraulica",
     "relevant": "calibracao", "category": "fora_dominio"}, 
]

def compute_mrr(rag, connectivity: bool, top_k: int = 5) -> dict:
    reciprocal_ranks_in_scope = []
    details = []
    ood_flags = []

    for item in GOLDEN_SET:
        result = rag.retrieve(item["query"], connectivity=connectivity)
        is_ood = result.get("is_ood", False)
        ood_flags.append(is_ood)

        chunks = result.get("chunks", [result.get("context", "")])
        rr = 0.0

        # Só calcula RR para queries dentro do domínio
        if not is_ood:
            for rank, chunk in enumerate(chunks[:top_k], start=1):
                if item["relevant"].lower() in chunk.lower():
                    rr = 1.0 / rank
                    break
            reciprocal_ranks_in_scope.append(rr)

        details.append({
            "query":    item["query"][:60],
            "relevant": item["relevant"],
            "category": item["category"],
            "rr":       round(rr, 4),
            "hit":      rr > 0,
            "is_ood":   is_ood,
        })

    # MRR global (apenas in-scope) e por categoria
    mrr_global = round(
        sum(reciprocal_ranks_in_scope) / len(reciprocal_ranks_in_scope), 4
    ) if reciprocal_ranks_in_scope else 0.0

    cats = {}
    for d in details:
        cats.setdefault(d["category"], []).append(d["rr"])
    mrr_by_cat = {c: round(sum(v)/len(v), 4) for c, v in cats.items()}

    # Taxa de detecção OOD entre as queries 'fora_dominio'
    ood_total = sum(
        1 for item, flag in zip(GOLDEN_SET, ood_flags)
        if item["category"] == "fora_dominio"
    )
    ood_detected = sum(
        1 for item, flag in zip(GOLDEN_SET, ood_flags)
        if item["category"] == "fora_dominio" and flag
    )
    ood_rate = round(
        ood_detected / ood_total, 4
    ) if ood_total else 0.0

    return {
        "mrr":        mrr_global,
        "mrr_by_cat": mrr_by_cat,
        "n":          len(GOLDEN_SET),
        "details":    details,
        "ood_rate":   ood_rate,
    }

if __name__ == "__main__":
    rag = EACO_RAG(top_k=5)
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M")

    print(f"\n=== AVALIAÇÃO MRR — EACO-RAG | {ts} ===")
    print(
          f"Golden set: {len(GOLDEN_SET)} queries  "
          f"(4 diretas · 4 adversariais · 2 fora-domínio)\n")

    results = {}
    for label, conn in [("conectado", True), ("isolado", False)]:
        r = compute_mrr(rag, connectivity=conn)
        results[label] = r
        print(
            f"[{label.upper()}] MRR global = {r['mrr']}  |  "
            f"por categoria: {r['mrr_by_cat']}"
        )

        for d in r["details"]:
            hit = "✅" if d["hit"] else "❌"
            print(
                f"  {hit} [{d['category']:12s}] RR={d['rr']:.4f} | {d['query']}")

        print()

    # CSV
    outfile = f"poc_mrr_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    with open(outfile, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(

["scenario","category","query","relevant_keyword","reciprocal_rank","hit"]
        )
        for label, r in results.items():
            for d in r["details"]:
                w.writerow([
                    label, d["category"], d["query"],
                    d["relevant"], d["rr"], d["hit"]
                ])

    print("==================================================")
    for label in ["conectado", "isolado"]:
        print(
            f"  MRR {label:10s} (in-scope): {results[label]['mrr']}  "
            f"(critério QP3 ≥ 0.80) · OOD rate: {results[label]['ood_rate']}"
        )

    delta = round(results['conectado']['mrr'] - results['isolado']['mrr'], 4)
    print(f"  Delta Con-Iso : {delta}")

    vmin  = min(results['conectado']['mrr'], results['isolado']['mrr'])
    status_qp3 = "✅ QP3 APROVADO" if vmin >= 0.80 else "❌ ABAIXO DO CRITÉRIO"
    print(f"  Status QP3    : {status_qp3}")
