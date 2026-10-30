"""Formal LTLf safety specifications compiled into DFAs via MONA."""
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SafetyRuleSpec:
    rule_id: str
    description: str
    ltlf_formula: str
    latex_formula: str
    atoms: Tuple[str, ...]
    corrective_action: str


SAFETY_SPECS: Tuple[SafetyRuleSpec, ...] = (
    SafetyRuleSpec(
        rule_id="phi_brake",
        description="Brake overheating requires immediate emergency stop",
        ltlf_formula="G(brake_hot -> act_estop)",
        latex_formula=r"\Box((T_{\mathrm{brake}} > 200\,^{\circ}\mathrm{C}) \rightarrow a = \mathrm{emergency\_stop})",
        atoms=("brake_hot", "act_estop"),
        corrective_action="emergency_stop",
    ),
    SafetyRuleSpec(
        rule_id="phi_gear",
        description="Reverse command or reverse actuation forbidden while vehicle is in motion",
        ltlf_formula="G((rev_cmd_moving -> act_estop) & (v_pos -> !act_rev))",
        latex_formula=r"\Box(((v > 0 \land \mathrm{cmd}=R) \rightarrow a = \mathrm{emergency\_stop}) \land (v > 0 \rightarrow a \neq \mathrm{reverse}))",
        atoms=("rev_cmd_moving", "v_pos", "act_estop", "act_rev"),
        corrective_action="emergency_stop",
    ),
    SafetyRuleSpec(
        rule_id="phi_speed",
        description="Overspeed above haulage limit requires immediate emergency stop",
        ltlf_formula="G(overspeed -> act_estop)",
        latex_formula=r"\Box((v > 20\,\mathrm{km/h}) \rightarrow a = \mathrm{emergency\_stop})",
        atoms=("overspeed", "act_estop"),
        corrective_action="emergency_stop",
    ),
    SafetyRuleSpec(
        rule_id="phi_temp",
        description="Critical engine temperature requires active cooling or dominating stop",
        ltlf_formula="G(eng_crit -> act_thermal_safe)",
        latex_formula=r"\Box((T_{\mathrm{eng}} > 105\,^{\circ}\mathrm{C}) \rightarrow a \in \{\mathrm{activate\_cooling}, \mathrm{emergency\_stop}\})",
        atoms=("eng_crit", "act_thermal_safe"),
        corrective_action="activate_cooling",
    ),
    SafetyRuleSpec(
        rule_id="phi_auth",
        description="Weak-until authorization: no risk execution without prior authorization",
        ltlf_formula="((!op_risk_viol) U auth) | G(!op_risk_viol)",
        latex_formula=r"(\neg \mathrm{Op}_{\mathrm{risk}}) \,\mathcal{W}\, \mathrm{Auth}",
        atoms=("op_risk_viol", "auth"),
        corrective_action="refuse_operation",
    ),
)
