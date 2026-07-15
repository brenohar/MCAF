# src/logic_guard.py — Módulo 2: LogicGuard (Verificação Formal em Execução)
import time
from dataclasses import dataclass, field
from typing import Optional

SINK = "SINK"   # estado absorvente = ação BLOQUEADA

# ─────────────────────────────────────────────────────────────────────────────
# DFAs de Segurança — Espelham exatamente a Tabela I (Artigo SBESC)
# ─────────────────────────────────────────────────────────────────────────────

class DFA_Gear:
    """
    Gear Interlock (φ_gear): □((v > 0) → ¬R)
    Enforced Action: emergency_stop
    """
    def transition(self, action: str, sensors: dict) -> str:
        cmd = sensors.get("operator_command", "")
        v = sensors.get("speed_kmh", 0.0)
        
        # Se operador pediu Ré com veículo andando, a ÚNICA ação admissível é frear.
        if cmd == "R" and v > 0.0:
            return "SAFE" if action == "emergency_stop" else SINK
        return "SAFE"

class DFA_Thermal:
    def transition(self, action: str, sensors: dict) -> str:
        t_eng = sensors.get("engine_temp_c", 0.0)
        # Se motor superaqueceu, cooling é mandatório, MAS emergência também é segura
        if t_eng > 105.0:
            return "SAFE" if action in ["activate_cooling", "emergency_stop"] else SINK
        return "SAFE"

class DFA_Authorization:
    def transition(self, action: str, sensors: dict) -> str:
        cmd = sensors.get("operator_command", "")
        auth = sensors.get("auth_token", False)
        # Se operador pede risco sem token, as ações padrão do sistema continuam seguras
        # pois nenhuma delas executa o risco solicitado.
        if cmd == "Op_risk" and not auth:
            return "SAFE" if action in ["continue_operation", "activate_cooling", "emergency_stop"] else SINK
        return "SAFE"

class DFA_Brake:
    """
    Brake Interlock (φ_brake): □((T_brake > 200) → ◇E_stop)
    Enforced Action: emergency_stop
    """
    def transition(self, action: str, sensors: dict) -> str:
        t_brake = sensors.get("brake_temp_c", 0.0)
        
        # Se o freio superaqueceu, a ÚNICA ação admissível é frear/parar.
        if t_brake > 200.0:
            return "SAFE" if action == "emergency_stop" else SINK
        return "SAFE"


# ─────────────────────────────────────────────────────────────────────────────
# LogicGuard — Orquestrador dos DFAs
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class VerdictResult:
    allowed:     bool
    latency_ms:  float
    blocked_by:  Optional[str] = None
    dfa_states:  dict = field(default_factory=dict)

class LogicGuard:
    def __init__(self):
        self.dfas = {
            "DFA_Gear":           DFA_Gear(),
            "DFA_Thermal":        DFA_Thermal(),
            "DFA_Authorization":  DFA_Authorization(),
            "DFA_Brake":          DFA_Brake(),
        }
        self.block_count  = 0
        self.total_count  = 0

    def check(self, action: str, sensors: dict) -> VerdictResult:
        t0 = time.perf_counter()
        self.total_count += 1

        dfa_states  = {}
        blocked_by  = None

        # Valida a ação proposta contra todos os 4 DFAs
        for name, dfa in self.dfas.items():
            state = dfa.transition(action, sensors)
            dfa_states[name] = state
            
            # Se algum DFA bloquear e for o primeiro a fazer isso, registra
            if state == SINK and blocked_by is None:
                blocked_by = name

        allowed = blocked_by is None
        latency_ms = round((time.perf_counter() - t0) * 1e3, 4)

        if not allowed:
            self.block_count += 1

        return VerdictResult(
            allowed=allowed,
            latency_ms=latency_ms,
            blocked_by=blocked_by,
            dfa_states=dfa_states
        )

    @property
    def block_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return round(self.block_count / self.total_count, 4)
