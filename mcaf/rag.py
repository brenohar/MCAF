"""Neural document retrieval (all-MiniLM-L6-v2) with asymmetric hot/cold stores."""
from dataclasses import dataclass
import random
import time
from typing import Dict, List, Tuple
import zlib

import numpy as np
from sentence_transformers import SentenceTransformer

from mcaf.config import EMBEDDING_MODEL_NAME, RunConfig


CORPUS_DOCUMENTS: Tuple[Dict[str, str], ...] = (
    {
        "id": "H01",
        "partition": "hot",
        "category": "brake_safety",
        "text": (
            "Critical Brake Thermal Procedure: Brake temperature exceeding 200.0 C "
            "indicates severe thermal saturation risk on haul ramps. Mandatory immediate "
            "action is emergency_stop. Never continue normal haulage above 200.0 C."
        ),
    },
    {
        "id": "H02",
        "partition": "hot",
        "category": "engine_thermal",
        "text": (
            "Critical Engine Overheating Procedure: Engine coolant temperature above "
            "105.0 C requires immediate execution of activate_cooling to prevent "
            "cylinder head warping and seizure."
        ),
    },
    {
        "id": "H03",
        "partition": "hot",
        "category": "drivetrain_gear",
        "text": (
            "Transmission Reverse Interlock: Operator command R (reverse gear) while "
            "vehicle speed is above 0.0 km/h causes catastrophic gearbox damage. "
            "Mandatory action is emergency_stop until speed reaches zero."
        ),
    },
    {
        "id": "H04",
        "partition": "hot",
        "category": "overspeed_control",
        "text": (
            "Haul Road Overspeed Interlock: Autonomous truck speed exceeding 20.0 km/h "
            "violates open-pit ramp safety limits. Mandatory immediate action is "
            "emergency_stop."
        ),
    },
    {
        "id": "H05",
        "partition": "hot",
        "category": "authorization_policy",
        "text": (
            "High-Risk Operation Authorization Policy: Any operator request Op_risk "
            "without a valid cryptographic auth_token (auth_token=False) must be "
            "rejected immediately using refuse_operation."
        ),
    },
    {
        "id": "D01",
        "partition": "hot",
        "category": "directive_EQ-02",
        "text": (
            "Maintenance directive bulletin for Truck EQ-02: rear wheel hub bearing wear "
            "flagged on EQ-02. When EQ-02 operates in intermediate engine temperature "
            "band (90.0 C to 105.0 C), required action for EQ-02 is schedule_maintenance."
        ),
    },
    {
        "id": "D02",
        "partition": "hot",
        "category": "directive_EQ-05",
        "text": (
            "Maintenance directive bulletin for Truck EQ-05: auxiliary pump cavitation "
            "flagged on EQ-05. When EQ-05 operates in intermediate engine temperature "
            "band (90.0 C to 105.0 C), required action for EQ-05 is schedule_maintenance."
        ),
    },
    {
        "id": "D03",
        "partition": "hot",
        "category": "directive_EQ-08",
        "text": (
            "Maintenance directive bulletin for Truck EQ-08: radiator core dust fouling "
            "flagged on EQ-08. When EQ-08 operates in intermediate engine temperature "
            "band (90.0 C to 105.0 C), required action for EQ-08 is schedule_maintenance."
        ),
    },
    {
        "id": "D04",
        "partition": "hot",
        "category": "directive_EQ-11",
        "text": (
            "Maintenance directive bulletin for Truck EQ-11: cooling fan valve hysteresis "
            "flagged on EQ-11. When EQ-11 operates in intermediate engine temperature "
            "band (90.0 C to 105.0 C), required action for EQ-11 is schedule_maintenance."
        ),
    },
    {
        "id": "D05",
        "partition": "hot",
        "category": "directive_EQ-15",
        "text": (
            "Maintenance directive bulletin for Truck EQ-15: exhaust manifold thermal alert "
            "on EQ-15. When EQ-15 operates in intermediate engine temperature "
            "band (90.0 C to 105.0 C), required action for EQ-15 is schedule_maintenance."
        ),
    },
    {
        "id": "D06",
        "partition": "hot",
        "category": "directive_EQ-19",
        "text": (
            "Maintenance directive bulletin for Truck EQ-19: turbocharger oil-return inspection "
            "on EQ-19. When EQ-19 operates in intermediate engine temperature "
            "band (90.0 C to 105.0 C), required action for EQ-19 is schedule_maintenance."
        ),
    },
    {
        "id": "C01",
        "partition": "cold",
        "category": "periodic_lubrication",
        "text": (
            "Engine Lubrication Schedule: Replace diesel engine oil every 250 operating "
            "hours or 6 months using API CI-4 15W-40 heavy-duty lubricant."
        ),
    },
    {
        "id": "C02",
        "partition": "cold",
        "category": "periodic_brake_lining",
        "text": (
            "Brake Pad Wear Inspection: Measure friction lining thickness every 500 hours. "
            "Minimum allowable pad thickness is 8 mm; replace axle pairs symmetrically."
        ),
    },
    {
        "id": "C03",
        "partition": "cold",
        "category": "periodic_coolant_flush",
        "text": (
            "Coolant Circuit Overhaul: Check expansion tank level before each shift and "
            "flush ethylene-glycol coolant completely every 2000 operating hours."
        ),
    },
    {
        "id": "C04",
        "partition": "cold",
        "category": "periodic_electrical",
        "text": (
            "24V Battery Bank Maintenance: Verify terminal voltage above 22.5 V prior "
            "to night-shift dispatch; replace cells below 80 percent rated capacity."
        ),
    },
    {
        "id": "C05",
        "partition": "cold",
        "category": "periodic_transmission_oil",
        "text": (
            "Planetary Transmission Fluid Service: Check sight-glass level every 125 hours "
            "and drain fluid every 1000 hours or upon burnt odor detection."
        ),
    },
    {
        "id": "C06",
        "partition": "cold",
        "category": "periodic_hydraulics",
        "text": (
            "Hoist Hydraulic System Specification: Nominal working pressure is 250 bar. "
            "Replace high-pressure return filters every 500 operating hours."
        ),
    },
    {
        "id": "C07",
        "partition": "cold",
        "category": "periodic_tires",
        "text": (
            "Haul Truck Tire Cold Inflation: Maintain 850 kPa on rear dual tires and "
            "700 kPa on front steer tires measured cold before shift start."
        ),
    },
)


@dataclass(frozen=True)
class RetrievalResult:
    doc_ids: Tuple[str, ...]
    context_text: str
    remote_queried: bool
    net_latency_ms: float


class NeuralRetriever:
    """Indexes hot (11 docs) and full (18 docs) stores once using all-MiniLM-L6-v2."""

    _shared_encoder: SentenceTransformer | None = None

    def __init__(self, top_k: int = 2):
        self.top_k = top_k
        if NeuralRetriever._shared_encoder is None:
            NeuralRetriever._shared_encoder = SentenceTransformer(
                EMBEDDING_MODEL_NAME, device="cpu"
            )
        self.encoder = NeuralRetriever._shared_encoder

        self.full_docs: List[Dict[str, str]] = list(CORPUS_DOCUMENTS)
        self.hot_docs: List[Dict[str, str]] = [
            d for d in CORPUS_DOCUMENTS if d["partition"] == "hot"
        ]

        full_texts = [f"{d['id']} {d['category']}: {d['text']}" for d in self.full_docs]
        hot_texts = [f"{d['id']} {d['category']}: {d['text']}" for d in self.hot_docs]

        self.full_emb = self.encoder.encode(
            full_texts, normalize_embeddings=True, show_progress_bar=False
        )
        self.hot_emb = self.encoder.encode(
            hot_texts, normalize_embeddings=True, show_progress_bar=False
        )

    @staticmethod
    def build_cycle_query(sit_dict: Dict[str, object]) -> str:
        eq_id = sit_dict["equipment_id"]
        return (
            f"Maintenance directive bulletin for Truck {eq_id}: "
            f"required action for {eq_id} in intermediate temperature band."
        )

    def rank_documents(self, query: str, use_full_store: bool) -> List[Tuple[str, float, str]]:
        q_vec = self.encoder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]
        docs = self.full_docs if use_full_store else self.hot_docs
        mat = self.full_emb if use_full_store else self.hot_emb
        scores = np.dot(mat, q_vec)
        order = np.argsort(-scores)
        return [(docs[i]["id"], float(scores[i]), docs[i]["text"]) for i in order]

    def retrieve_for_cycle(
        self, sit_dict: Dict[str, object], cfg: RunConfig
    ) -> RetrievalResult:
        query = self.build_cycle_query(sit_dict)
        cyc = int(sit_dict["cycle_idx"])

        if cfg.scenario == "isolated":
            ranked = self.rank_documents(query, use_full_store=False)[: self.top_k]
            ids = tuple(r[0] for r in ranked)
            ctx = "\n".join(f"[{r[0]}] {r[2]}" for r in ranked)
            return RetrievalResult(
                doc_ids=ids, context_text=ctx, remote_queried=False, net_latency_ms=0.0
            )

        net_seed = zlib.crc32(
            f"net|{cfg.scenario}|seed={cfg.seed}|cyc={cyc}".encode("utf-8")
        )
        net_rng = random.Random(net_seed)

        if cfg.scenario == "connected":
            net_ms = 0.0
            use_full = True
        else:
            net_ms = float(cfg.net_latency_ms)
            packet_lost = net_rng.random() < cfg.net_loss_prob
            use_full = not packet_lost

        if net_ms > 0.0 and cfg.simulate_net_sleep:
            time.sleep(net_ms / 1000.0)

        ranked = self.rank_documents(query, use_full_store=use_full)[: self.top_k]
        ids = tuple(r[0] for r in ranked)
        ctx = "\n".join(f"[{r[0]}] {r[2]}" for r in ranked)
        return RetrievalResult(
            doc_ids=ids, context_text=ctx, remote_queried=True, net_latency_ms=net_ms
        )
