"""Runtime formal safety shield executing MONA-compiled DFAs in O(1)."""
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Dict, Optional

from mcaf.config import (
    BRAKE_TEMP_MAX_C,
    ENGINE_TEMP_CRIT_C,
    SPECS_DIR,
    SPEED_MAX_KMH,
)


@dataclass(frozen=True)
class ShieldVerdict:
    proposed_action: str
    executed_action: str
    intervened: bool
    blocked_by: str
    latency_ms: float


class RuntimeShield:
    """Executes compiled DFAs step-by-step and applies rule-specific overrides."""

    def __init__(self, artifact_path: Optional[Path] = None):
        path = artifact_path or (SPECS_DIR / "compiled_shield.json")
        if not path.exists():
            raise FileNotFoundError(
                f"Artefato {path} ausente. Rode 'python analysis/compile_shield.py' antes."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.rules = payload["rules"]
        self.states: Dict[str, str] = {r["rule_id"]: r["initial_state"] for r in self.rules}
        self._accepting: Dict[str, set[str]] = {
            r["rule_id"]: set(r["accepting_states"]) for r in self.rules
        }
        self._prev_auth: bool = False

    def reset_episode(self) -> None:
        self.states = {r["rule_id"]: r["initial_state"] for r in self.rules}
        self._prev_auth = False

    @staticmethod
    def _bit_key(rule_id: str, sit_dict: Dict[str, object], action: str) -> str:
        brk = float(sit_dict["brake_temp_c"])
        spd = float(sit_dict["speed_kmh"])
        eng = float(sit_dict["engine_temp_c"])
        cmd = str(sit_dict["operator_command"])
        auth = bool(sit_dict["auth_token"])

        if rule_id == "phi_brake":
            b0 = brk > BRAKE_TEMP_MAX_C
            b1 = action == "emergency_stop"
            return ("1" if b0 else "0") + ("1" if b1 else "0")

        if rule_id == "phi_gear":
            b0 = spd > 0.0 and cmd == "R"
            b1 = spd > 0.0
            b2 = action == "emergency_stop"
            b3 = action == "engage_reverse"
            return "".join("1" if b else "0" for b in (b0, b1, b2, b3))

        if rule_id == "phi_speed":
            b0 = spd > SPEED_MAX_KMH
            b1 = action == "emergency_stop"
            return ("1" if b0 else "0") + ("1" if b1 else "0")

        if rule_id == "phi_temp":
            b0 = eng > ENGINE_TEMP_CRIT_C
            b1 = action in ("activate_cooling", "emergency_stop")
            return ("1" if b0 else "0") + ("1" if b1 else "0")

        if rule_id == "phi_auth":
            op_risk_viol = (action == "execute_risk_op") or (
                cmd == "Op_risk"
                and not auth
                and action not in ("refuse_operation", "emergency_stop", "activate_cooling")
            )
            return ("1" if op_risk_viol else "0") + ("1" if auth else "0")

        raise KeyError(f"Regra desconhecida: {rule_id}")

    def verify_and_enforce(self, sit_dict: Dict[str, object], proposed_action: str) -> ShieldVerdict:
        t0 = time.perf_counter()

        # Se um token de autorizacao anterior expirou, renova a janela de phi_auth
        curr_auth = bool(sit_dict["auth_token"])
        if self._prev_auth and not curr_auth:
            for r in self.rules:
                if r["rule_id"] == "phi_auth":
                    self.states["phi_auth"] = r["initial_state"]
        self._prev_auth = curr_auth

        exec_action = proposed_action
        blocked_by = ""

        # Avalia os 5 automatos na ordem formal de prioridade
        for r in self.rules:
            rid = r["rule_id"]
            cur_q = self.states[rid]
            key = self._bit_key(rid, sit_dict, exec_action)
            nxt_q = r["transitions"][cur_q][key]

            if nxt_q not in self._accepting[rid]:
                if not blocked_by:
                    blocked_by = rid
                exec_action = r["corrective_action"]

        # Avanca o estado real dos 5 automatos com a acao efetivamente executada
        for r in self.rules:
            rid = r["rule_id"]
            cur_q = self.states[rid]
            key = self._bit_key(rid, sit_dict, exec_action)
            nxt_q = r["transitions"][cur_q][key]
            self.states[rid] = nxt_q

        elapsed_ms = (time.perf_counter() - t0) * 1e3
        return ShieldVerdict(
            proposed_action=proposed_action,
            executed_action=exec_action,
            intervened=(blocked_by != ""),
            blocked_by=blocked_by,
            latency_ms=elapsed_ms,
        )
