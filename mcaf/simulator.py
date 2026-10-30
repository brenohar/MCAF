"""Paired open-loop telemetry simulator, ground-truth oracle, and prefilter."""
from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Tuple
import zlib

from mcaf.config import (
    ALL_EQUIPMENTS,
    BRAKE_TEMP_MAX_C,
    DIRECTIVE_EQUIPMENTS,
    ENGINE_TEMP_BAND_MIN_C,
    ENGINE_TEMP_CRIT_C,
    MA_WINDOW_CYCLES,
    RunConfig,
    SPEED_MAX_KMH,
)


def derive_paired_seed(cfg: RunConfig) -> int:
    """Derives a deterministic seed strictly from environment params (never controller)."""
    env_signature = (
        f"{cfg.scenario}|seed={cfg.seed}|pcrit={cfg.p_crit:.4f}|"
        f"pauth={cfg.p_auth:.4f}|pop={cfg.p_op_cmd:.4f}|"
        f"nact={cfg.n_actions}|band={cfg.band_mode}"
    )
    return (zlib.crc32(env_signature.encode("utf-8")) ^ (cfg.seed & 0xFFFFFFFF)) & 0xFFFFFFFF


@dataclass(frozen=True)
class CycleSituation:
    cycle_idx: int
    equipment_id: str
    has_directive: bool
    engine_temp_c: float
    brake_temp_c: float
    speed_kmh: float
    operator_command: str
    auth_token: bool
    ma5_current_c: float
    ma5_prev_c: float
    ma5_rising: bool
    is_critical_gt: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "cycle_idx": self.cycle_idx,
            "equipment_id": self.equipment_id,
            "has_directive": self.has_directive,
            "engine_temp_c": self.engine_temp_c,
            "brake_temp_c": self.brake_temp_c,
            "speed_kmh": self.speed_kmh,
            "operator_command": self.operator_command,
            "auth_token": self.auth_token,
            "ma5_current_c": self.ma5_current_c,
            "ma5_prev_c": self.ma5_prev_c,
            "ma5_rising": self.ma5_rising,
            "is_critical_gt": self.is_critical_gt,
        }


def compute_ground_truth(sit: CycleSituation, band_mode: str = "directive") -> str:
    """20-line reference policy (pi*) in fixed priority order."""
    if sit.brake_temp_c > BRAKE_TEMP_MAX_C:
        return "emergency_stop"
    if sit.speed_kmh > 0.0 and sit.operator_command == "R":
        return "emergency_stop"
    if sit.speed_kmh > SPEED_MAX_KMH:
        return "emergency_stop"
    if sit.engine_temp_c > ENGINE_TEMP_CRIT_C:
        return "activate_cooling"
    if sit.operator_command == "Op_risk" and not sit.auth_token:
        return "refuse_operation"
    if ENGINE_TEMP_BAND_MIN_C <= sit.engine_temp_c <= ENGINE_TEMP_CRIT_C:
        if band_mode == "directive":
            return "schedule_maintenance" if sit.has_directive else "continue_operation"
        return "activate_cooling" if sit.ma5_rising else "continue_operation"
    return "continue_operation"


def deterministic_prefilter(sit: CycleSituation) -> Optional[str]:
    """Deterministic safety filter using the exact same priority list on critical states."""
    if sit.brake_temp_c > BRAKE_TEMP_MAX_C:
        return "emergency_stop"
    if sit.speed_kmh > 0.0 and sit.operator_command == "R":
        return "emergency_stop"
    if sit.speed_kmh > SPEED_MAX_KMH:
        return "emergency_stop"
    if sit.engine_temp_c > ENGINE_TEMP_CRIT_C:
        return "activate_cooling"
    if sit.operator_command == "Op_risk" and not sit.auth_token:
        return "refuse_operation"
    return None


class SituationSimulator:
    """Open-loop paired generator with thermal memory and full rule coverage."""

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.rng = random.Random(derive_paired_seed(cfg))
        self.base_temp_c: float = 91.0
        self.temp_history: List[float] = [91.0] * MA_WINDOW_CYCLES

    def next_situation(self, cycle_idx: int) -> CycleSituation:
        eq_id = self.rng.choice(ALL_EQUIPMENTS)
        has_dir = eq_id in DIRECTIVE_EQUIPMENTS

        # Evolucao termica com memoria (filtro AR(1) na faixa nominal/intermediaria)
        drift = self.rng.gauss(0.0, 2.2)
        pull = 0.12 * (93.5 - self.base_temp_c)
        self.base_temp_c = min(max(self.base_temp_c + pull + drift, 76.0), 104.2)

        is_crit_draw = self.rng.random() < self.cfg.p_crit

        if is_crit_draw:
            hazard = self.rng.choice(
                ("brake_hot", "rev_moving", "overspeed", "eng_crit", "unauth_risk")
            )
            if hazard == "brake_hot":
                eng_t = round(self.base_temp_c, 1)
                brk_t = round(self.rng.uniform(201.0, 310.0), 1)
                spd = round(self.rng.uniform(4.0, 18.0), 1)
                cmd, auth = "continue_operation", False
            elif hazard == "rev_moving":
                eng_t = round(self.base_temp_c, 1)
                brk_t = round(self.rng.uniform(60.0, 160.0), 1)
                spd = round(self.rng.uniform(2.0, 18.0), 1)
                cmd, auth = "R", False
            elif hazard == "overspeed":
                eng_t = round(self.base_temp_c, 1)
                brk_t = round(self.rng.uniform(70.0, 175.0), 1)
                spd = round(self.rng.uniform(20.5, 38.0), 1)
                cmd, auth = "continue_operation", False
            elif hazard == "eng_crit":
                eng_t = round(self.rng.uniform(105.5, 118.0), 1)
                brk_t = round(self.rng.uniform(60.0, 165.0), 1)
                spd = round(self.rng.uniform(3.0, 18.0), 1)
                cmd, auth = "continue_operation", False
            else:  # unauth_risk
                eng_t = round(self.base_temp_c, 1)
                brk_t = round(self.rng.uniform(55.0, 150.0), 1)
                spd = round(self.rng.uniform(0.0, 16.0), 1)
                cmd, auth = "Op_risk", False
        else:
            eng_t = round(self.base_temp_c, 1)
            brk_t = round(self.rng.uniform(50.0, 175.0), 1)
            op_roll = self.rng.random()
            if op_roll < (self.cfg.p_op_cmd * 0.5):
                # Pedido seguro de marcha a re com veiculo parado (v == 0)
                spd = 0.0
                cmd, auth = "R", False
            elif op_roll < self.cfg.p_op_cmd:
                # Pedido de operacao de risco; se cair sem token, vira critico auditavel
                spd = round(self.rng.uniform(0.0, 16.0), 1)
                cmd = "Op_risk"
                auth = self.rng.random() < self.cfg.p_auth
            else:
                spd = round(self.rng.uniform(0.0, 19.2), 1)
                cmd, auth = "continue_operation", False

        ma_prev = sum(self.temp_history[-MA_WINDOW_CYCLES:]) / MA_WINDOW_CYCLES
        self.temp_history.append(eng_t)
        ma_curr = sum(self.temp_history[-MA_WINDOW_CYCLES:]) / MA_WINDOW_CYCLES
        ma_rising = ma_curr > ma_prev

        is_crit_actual = (
            brk_t > BRAKE_TEMP_MAX_C
            or (spd > 0.0 and cmd == "R")
            or spd > SPEED_MAX_KMH
            or eng_t > ENGINE_TEMP_CRIT_C
            or (cmd == "Op_risk" and not auth)
        )

        return CycleSituation(
            cycle_idx=cycle_idx,
            equipment_id=eq_id,
            has_directive=has_dir,
            engine_temp_c=eng_t,
            brake_temp_c=brk_t,
            speed_kmh=spd,
            operator_command=cmd,
            auth_token=auth,
            ma5_current_c=round(ma_curr, 3),
            ma5_prev_c=round(ma_prev, 3),
            ma5_rising=ma_rising,
            is_critical_gt=is_crit_actual,
        )

    def generate_episode(self) -> List[Tuple[CycleSituation, str]]:
        episode = []
        for idx in range(self.cfg.n_cycles):
            sit = self.next_situation(idx)
            gt = compute_ground_truth(sit, band_mode=self.cfg.band_mode)
            episode.append((sit, gt))
        return episode
