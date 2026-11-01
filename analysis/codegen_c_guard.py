"""Generates zero-allocation ANSI C lookup tables directly from specs/compiled_shield.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "specs" / "compiled_shield.json"
OUT_HDR = ROOT / "c_guard" / "shield_tables.h"

ACTION_ENUM = {
    "continue_operation": "ACT_CONTINUE",
    "activate_cooling": "ACT_COOLING",
    "emergency_stop": "ACT_ESTOP",
    "refuse_operation": "ACT_REFUSE",
    "schedule_maintenance": "ACT_MAINT",
    "reduce_speed": "ACT_REDUCE_SPEED",
    "engage_reverse": "ACT_REVERSE",
    "execute_risk_op": "ACT_EXEC_RISK",
}


def main() -> None:
    payload = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    rules = payload["rules"]

    lines = [
        "/* AUTO-GENERATED FROM specs/compiled_shield.json (MONA DFA) - DO NOT EDIT */",
        "#ifndef MCAF_SHIELD_TABLES_H",
        "#define MCAF_SHIELD_TABLES_H",
        "",
        "#include <stdint.h>",
        "",
        "#define N_SHIELD_RULES 5",
        "#define MAX_DFA_STATES 4",
        "#define MAX_VALUATIONS 16",
        "",
        "typedef enum {",
        "    ACT_CONTINUE = 0,",
        "    ACT_COOLING = 1,",
        "    ACT_ESTOP = 2,",
        "    ACT_REFUSE = 3,",
        "    ACT_MAINT = 4,",
        "    ACT_REDUCE_SPEED = 5,",
        "    ACT_REVERSE = 6,",
        "    ACT_EXEC_RISK = 7",
        "} mcaf_action_t;",
        "",
        "typedef enum {",
        "    CMD_CONTINUE = 0,",
        "    CMD_REVERSE = 1,",
        "    CMD_OP_RISK = 2",
        "} mcaf_cmd_t;",
        "",
        "typedef struct {",
        "    const char *rule_id;",
        "    uint8_t initial_state;",
        "    uint32_t accepting_mask;",
        "    mcaf_action_t corrective_action;",
        "    uint8_t transitions[MAX_DFA_STATES][MAX_VALUATIONS];",
        "} dfa_rule_table_t;",
        "",
        "static const dfa_rule_table_t MONA_COMPILED_RULES[N_SHIELD_RULES] = {",
    ]

    for r in rules:
        st_list = r["states"]
        st_map = {s: idx for idx, s in enumerate(st_list)}
        init_idx = st_map[r["initial_state"]]
        acc_mask = 0
        for acc_s in r["accepting_states"]:
            acc_mask |= (1 << st_map[acc_s])
        corr_c = ACTION_ENUM[r["corrective_action"]]
        n_vals = 1 << len(r["atoms"])

        lines.append("    {")
        lines.append(f'        .rule_id = "{r["rule_id"]}",')
        lines.append(f"        .initial_state = {init_idx}u,")
        lines.append(f"        .accepting_mask = 0x{acc_mask:02X}u,")
        lines.append(f"        .corrective_action = {corr_c},")
        lines.append("        .transitions = {")
        for s_str in st_list:
            row_vals = []
            for v_idx in range(n_vals):
                bkey = format(v_idx, f"0{len(r['atoms'])}b")
                dst_idx = st_map[r["transitions"][s_str][bkey]]
                row_vals.append(f"{dst_idx}u")
            pad_vals = ["0u"] * (16 - n_vals)
            lines.append("            { " + ", ".join(row_vals + pad_vals) + " },")
        lines.append("        }")
        lines.append("    },")

    lines.extend([
        "};",
        "",
        "#endif /* MCAF_SHIELD_TABLES_H */",
        "",
    ])
    OUT_HDR.parent.mkdir(parents=True, exist_ok=True)
    OUT_HDR.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Tabelas C geradas a partir do MONA em: {OUT_HDR}")


if __name__ == "__main__":
    main()
