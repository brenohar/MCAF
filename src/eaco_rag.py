# src/eaco_rag.py — Módulo 3: EACO-RAG (Raciocínio sem Conectividade)
import time
import math
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# Base de conhecimento sintética — manuais de caminhão de mineração
# Estrutura: {id, texto, categoria, partition}
# ─────────────────────────────────────────────────────────────────────────────
KNOWLEDGE_BASE = [
    # ── HOT (críticos, alta frequência de acesso) ─────────────────────────
    {"id": "H01", "partition": "hot",
     "categoria": "freio",
     "texto": "Temperatura do freio acima de 500C indica superaquecimento critico. Acionar resfriamento imediato e reduzir velocidade. Inspecionar pastilhas apos resfriamento."},
    {"id": "H02", "partition": "hot",
     "categoria": "motor",
     "texto": "Motor acima de 105C requer ativacao imediata do sistema de resfriamento. Desligar cargas adicionais. Parar veiculo se temperatura nao cair em 60 segundos."},
    {"id": "H03", "partition": "hot",
     "categoria": "transmissao",
     "texto": "Nunca engatar marcha re com veiculo em movimento. Aguardar velocidade zero antes de trocar direcao. Falha nessa regra causa danos graves a transmissao."},
    {"id": "H04", "partition": "hot",
     "categoria": "emergencia",
     "texto": "Procedimento de parada de emergencia: 1) Acionar freio de servico. 2) Reduzir acelerador a zero. 3) Engatar freio de estacionamento. 4) Notificar central."},
    {"id": "H05", "partition": "hot",
     "categoria": "pneu",
     "texto": "Temperatura de pneu acima de 90C indica risco de estouro. Parar veiculo imediatamente em local seguro e aguardar resfriamento natural por 30 minutos."},
    # ── COLD (contexto amplo, acesso esporadico) ──────────────────────────
    {"id": "C01", "partition": "cold",
     "categoria": "manutencao",
     "texto": "Intervalo de troca de oleo do motor: 250 horas de operacao ou 6 meses. Usar oleo 15W-40 API CI-4 para motores diesel de grande porte em mineracao."},
    {"id": "C02", "partition": "cold",
     "categoria": "freio",
     "texto": "Inspecao das pastilhas de freio deve ocorrer a cada 500 horas. Espessura minima aceitavel 8mm. Substituir em pares para garantir balanceamento de frenagem."},
    {"id": "C03", "partition": "cold",
     "categoria": "motor",
     "texto": "Sistema de arrefecimento: verificar nivel de fluido diariamente. Trocar fluido refrigerante a cada 2000 horas. Inspecionar mangueiras e conexoes mensalmente."},
    {"id": "C04", "partition": "cold",
     "categoria": "eletrico",
     "texto": "Bateria de 24V requer carga completa antes de operacao em turno noturno. Tensao minima aceitavel 22.5V. Substituir se capacidade cair abaixo de 80% do nominal."},
    {"id": "C05", "partition": "cold",
     "categoria": "transmissao",
     "texto": "Fluido de transmissao automatica: inspecionar nivel a cada 125 horas. Trocar a cada 1000 horas ou se apresentar coloracao escura ou odor de queimado."},
    {"id": "C06", "partition": "cold",
     "categoria": "hidrau",
     "texto": "Sistema hidraulico: pressao nominal 250 bar. Inspecionar mangueiras por vazamentos diariamente. Filtro hidraulico deve ser trocado a cada 500 horas de operacao."},
    {"id": "C07", "partition": "cold",
     "categoria": "pneu",
     "texto": "Pressao dos pneus traseiros carregados: 850 kPa. Dianteiros: 700 kPa. Verificar pressao com pneu frio antes do inicio de cada turno operacional."},
]

# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF simplificado (sem dependências externas)
# ─────────────────────────────────────────────────────────────────────────────

import re

SYNONYM_MAP = {
    "brake": ["freio"],
    "brakes": ["freio"],
    "overheat": ["superaquecimento"],
    "overheating": ["superaquecimento"],
    "cooling": ["resfriamento"],
    "cool": ["resfriamento"],
    "temperature": ["temperatura"],
    "temp": ["temperatura"],
    "threshold": ["limite"],
    "limit": ["limite"],
    "exceeded": ["acima"],
    "exceed": ["acima"],
    "downhill": ["descida"],
    "slope": ["rampa"],
    "truck": ["veiculo"],
    "mining": ["mineracao"],
    "speed": ["velocidade"],
    "reduce": ["reduzir"],
    "stop": ["parar"],
    "shutdown": ["parar"],
    "alerts": ["alertas"],
    "warning": ["alerta"],
    "engine": ["motor"],
    "emergency": ["emergencia"],
}

def tokenize(text):
    text = text.lower()
    raw_tokens = re.findall(r"[a-zA-Z0-9_]+", text)

    normalized_tokens = []
    for token in raw_tokens:
        normalized_tokens.append(token)

        if token in SYNONYM_MAP:
            normalized_tokens.extend(SYNONYM_MAP[token])

    return normalized_tokens

def build_tfidf_index(docs):
    N = len(docs)
    tf_vectors  = []
    df          = defaultdict(int)

    for doc in docs:
        tokens = tokenize(doc["texto"] + " " + doc["categoria"])
        tf     = defaultdict(int)
        for t in tokens:
            tf[t] += 1

        total = max(len(tokens), 1)
        tf_norm = {t: c / total for t, c in tf.items()}
        tf_vectors.append(tf_norm)

        for t in set(tokens):
            df[t] += 1

    idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in df}

    tfidf_vectors = []
    for tf in tf_vectors:
        vec = {t: tf[t] * idf.get(t, 1) for t in tf}
        norm = math.sqrt(sum(v**2 for v in vec.values())) or 1.0
        tfidf_vectors.append({t: v / norm for t, v in vec.items()})

    return tfidf_vectors, idf

def cosine_similarity(vec_q, vec_d):
    common = set(vec_q) & set(vec_d)
    return sum(vec_q[t] * vec_d[t] for t in common)

def query_to_tfidf(query, idf):
    tokens = tokenize(query)
    tf     = defaultdict(int)
    for t in tokens:
        tf[t] += 1
    total  = max(len(tokens), 1)
    vec    = {t: (c / total) * idf.get(t, 1) for t, c in tf.items()}
    norm   = math.sqrt(sum(v**2 for v in vec.values())) or 1.0
    return {t: v / norm for t, v in vec.items()}

# ─────────────────────────────────────────────────────────────────────────────
# EACO-RAG
# ─────────────────────────────────────────────────────────────────────────────
class EACO_RAG:
    def __init__(
        self,
        top_k: int = 5,
        candidate_k: int = 8,
        min_score: float = 0.15,
        cold_weight_default: float = 0.45,
        cold_weight_critical: float = 0.15,
    ):
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.min_score = min_score
        self.cold_weight_default = cold_weight_default
        self.cold_weight_critical = cold_weight_critical

        self.docs = KNOWLEDGE_BASE
        self.hot_docs = [d for d in self.docs if d["partition"] == "hot"]
        self.cold_docs = [d for d in self.docs if d["partition"] == "cold"]

        self.tfidf_vecs, self.idf = build_tfidf_index(self.docs)
        self.hot_vecs = self.tfidf_vecs[:len(self.hot_docs)]
        self.cold_vecs = self.tfidf_vecs[len(self.hot_docs):]

    def _rerank_score(self, base_score: float, doc: dict, query_tokens: list[str]) -> float:
        bonus = 0.0
        qset = set(query_tokens)
        text = doc["texto"].lower()
        categoria = doc["categoria"]

        if doc["partition"] == "hot":
            bonus += 0.10

        if categoria in qset:
            bonus += 0.14

        if "resfriamento" in qset and categoria == "motor":
            bonus += 0.20

        if {"alertas", "alerta", "simultaneas", "simultanea", "multiplas"} & qset and categoria == "emergencia":
            bonus += 0.25

        if {"parada", "parar", "emergencia"} & qset and categoria == "emergencia":
            bonus += 0.18

        if {"temperatura", "limite", "acima", "superaquecimento"} & qset and categoria == "motor":
            bonus += 0.12

        if "freio" in qset and categoria == "freio":
            bonus += 0.12

        if "motor" in text and "resfriamento" in qset and categoria == "motor":
            bonus += 0.08

        return base_score + bonus

    def _select_diverse_topk(self, ranked_items: list[tuple[float, dict]]) -> list[tuple[float, dict]]:
        selected = []
        seen_ids = set()
        category_counts = defaultdict(int)

        for score, doc in ranked_items:
            if len(selected) >= self.top_k:
                break

            doc_id = doc["id"]
            category = doc["categoria"]

            if doc_id in seen_ids:
                continue

            if category_counts[category] >= 1 and len(selected) < self.top_k - 1:
                continue

            selected.append((score, doc))
            seen_ids.add(doc_id)
            category_counts[category] += 1

        if len(selected) < self.top_k:
            for score, doc in ranked_items:
                if len(selected) >= self.top_k:
                    break
                if doc["id"] in seen_ids:
                    continue
                selected.append((score, doc))
                seen_ids.add(doc["id"])

        return selected

    def retrieve(self, query: str, connectivity: bool = False) -> dict:
        t0 = time.perf_counter()
        q_vec = query_to_tfidf(query, self.idf)
        query_tokens = tokenize(query)
        query_terms = set(query_tokens)

        critical_terms = {
            "emergencia",
            "freio",
            "motor",
            "parada",
            "critico",
            "superaquecimento",
            "resfriamento",
            "limite",
            "temperatura",
            "acima",
            "parar",
        }

        maintenance_terms = {
            "manutencao",
            "inspecao",
            "troca",
            "fluido",
            "oleo",
            "filtro",
            "mangueiras",
            "pastilhas",
            "pressao",
            "calibracao",
        }

        is_critical_query = bool(query_terms & critical_terms)
        is_maintenance_query = bool(query_terms & maintenance_terms)

        if connectivity:
            if is_critical_query:
                use_cold = False
                cold_weight = 0.0
            else:
                use_cold = True
                cold_weight = self.cold_weight_default if is_maintenance_query else 0.20
        else:
            use_cold = False
            cold_weight = 0.0

        scores_hot = [
            (cosine_similarity(q_vec, v), self.hot_docs[i])
            for i, v in enumerate(self.hot_vecs)
        ]

        scores_cold = []
        if connectivity and use_cold:
            scores_cold = [
                (cosine_similarity(q_vec, v) * cold_weight, self.cold_docs[i])
                for i, v in enumerate(self.cold_vecs)
            ]

        all_scores = sorted(scores_hot + scores_cold, key=lambda x: x[0], reverse=True)
        candidates = all_scores[:self.candidate_k]

        reranked_scores = sorted(
            [
                (self._rerank_score(score, doc, query_tokens), doc)
                for score, doc in candidates
            ],
            key=lambda x: x[0],
            reverse=True
        )

        selected = reranked_scores[:self.top_k]

        latency_ms = round((time.perf_counter() - t0) * 1e3, 4)

        best_score = selected[0][0] if selected else 0.0
        avg_score = sum(score for score, _ in selected) / len(selected) if selected else 0.0
        support_count = sum(1 for score, _ in selected if score >= self.min_score)

        is_ood = (best_score < self.min_score) or (support_count == 0)

        chunks = [
            f"[DOC {i+1} | id={doc['id']} | cat={doc['categoria']} | part={doc['partition']} | score={score:.4f}] {doc['texto']}"
            for i, (score, doc) in enumerate(selected)
        ]

        context = "\n\n".join(chunks)

        return {
            "query": query,
            "connectivity": connectivity,
            "top_k": [(round(score, 4), doc["id"], doc["partition"]) for score, doc in selected],
            "best_score": round(best_score, 4),
            "avg_score": round(avg_score, 4),
            "support_count": support_count,
            "chunks": chunks,
            "context": context,
            "latency_ms": latency_ms,
            "source": "hot+cold" if connectivity else "hot_only",
            "is_ood": is_ood,
        }

    def mrr(self, queries_relevant: list) -> float:
        reciprocal_ranks = []
        for query, rel_id in queries_relevant:
            result = self.retrieve(query, connectivity=True)
            ids = [doc_id for _, doc_id, _ in result["top_k"]]
            if rel_id in ids:
                rank = ids.index(rel_id) + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        return round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4) if reciprocal_ranks else 0.0
