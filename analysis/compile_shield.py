"""Compiles all 5 LTLf formulas into explicit DFA lookup tables via MONA."""
import itertools
import json
from pathlib import Path
import re
import sys

from ltlf2dfa.parser.ltlf import LTLfParser
import sympy
from sympy.parsing.sympy_parser import parse_expr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from specs.formulas import SAFETY_SPECS, SafetyRuleSpec  # noqa: E402


def _eval_guard(label: str, atoms: tuple[str, ...], valuation: dict[str, bool]) -> bool:
    clean = label.strip()
    if clean.lower() == "true":
        return True
    if clean.lower() == "false":
        return False
    sym_map = {name: sympy.Symbol(name) for name in atoms}
    expr = parse_expr(clean, local_dict=sym_map)
    subs = {sym_map[name]: bool(valuation[name]) for name in atoms}
    return bool(expr.subs(subs))


def compile_rule_to_dfa(spec: SafetyRuleSpec) -> dict:
    parser = LTLfParser()
    dot_str = parser(spec.ltlf_formula).to_dfa()

    # Extrai estados de aceitacao (doublecircle)
    acc_match = re.search(r"node\s*\[\s*shape\s*=\s*doublecircle\s*\]\s*;([^;\n]*(?:;[^\[\n]*)*)", dot_str)
    accepting = set()
    if acc_match:
        accepting = {s for s in re.findall(r"\b\d+\b", acc_match.group(1))}

    # Extrai estado inicial real (init -> X;)
    init_match = re.search(r"init\s*->\s*(\d+)\s*;", dot_str)
    if not init_match:
        raise RuntimeError(f"Estado inicial nao encontrado no DOT de {spec.rule_id}:\n{dot_str}")
    init_state = init_match.group(1)

    # Extrai arestas rotuladas (src -> dst [label="..."];)
    raw_edges = re.findall(r'(\d+)\s*->\s*(\d+)\s*\[\s*label\s*=\s*"([^"]+)"\s*\]\s*;', dot_str)
    states = {init_state} | accepting
    edges_by_src: dict[str, list[tuple[str, str]]] = {}
    for src, dst, lbl in raw_edges:
        states.add(src)
        states.add(dst)
        edges_by_src.setdefault(src, []).append((dst, lbl))

    # Expande a tabela completa deterministica sobre todas as 2^|atoms| valoracoes
    transition_table: dict[str, dict[str, str]] = {}
    for st in sorted(states, key=int):
        transition_table[st] = {}
        for bits in itertools.product((False, True), repeat=len(spec.atoms)):
            val = dict(zip(spec.atoms, bits))
            bit_key = "".join("1" if b else "0" for b in bits)
            matched = [
                dst for dst, lbl in edges_by_src.get(st, [])
                if _eval_guard(lbl, spec.atoms, val)
            ]
            if len(matched) != 1:
                raise RuntimeError(
                    f"DFA nao-deterministico ou incompleto em {spec.rule_id}, "
                    f"estado={st}, val={val}, destinos={matched}"
                )
            transition_table[st][bit_key] = matched[0]

    return {
        "rule_id": spec.rule_id,
        "description": spec.description,
        "ltlf_formula": spec.ltlf_formula,
        "latex_formula": spec.latex_formula,
        "atoms": list(spec.atoms),
        "corrective_action": spec.corrective_action,
        "initial_state": init_state,
        "states": sorted(states, key=int),
        "accepting_states": sorted(accepting, key=int),
        "n_states": len(states),
        "n_symbolic_edges": len(raw_edges),
        "n_expanded_transitions": sum(len(v) for v in transition_table.values()),
        "transitions": transition_table,
    }


def main() -> None:
    compiled_rules = [compile_rule_to_dfa(spec) for spec in SAFETY_SPECS]
    tot_states = sum(r["n_states"] for r in compiled_rules)
    tot_sym_edges = sum(r["n_symbolic_edges"] for r in compiled_rules)
    prod_states = 1
    for r in compiled_rules:
        prod_states *= r["n_states"]

    artifact = {
        "compiler": "ltlf2dfa + MONA",
        "n_rules": len(compiled_rules),
        "sum_individual_states": tot_states,
        "sum_symbolic_transitions": tot_sym_edges,
        "upper_bound_product_states": prod_states,
        "rules": compiled_rules,
    }

    out_path = ROOT / "specs" / "compiled_shield.json"
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print("=== COMPILACAO FORMAL LTLf -> DFA (MONA) CONCLUIDA ===")
    for r in compiled_rules:
        print(
            f"  [{r['rule_id']:10s}] estados={r['n_states']} | "
            f"arestas_MONA={r['n_symbolic_edges']} | "
            f"transicoes_2^k={r['n_expanded_transitions']} | "
            f"aceitacao={r['accepting_states']} | corretiva={r['corrective_action']}"
        )
    print(f"TOTAL SOMA INDIVIDUAL: {tot_states} estados, {tot_sym_edges} transicoes simbolicas.")
    print(f"PRODUTO CARTESIANO MAXIMO: {prod_states} estados.")
    print(f"Artefato salvo em: {out_path}")


if __name__ == "__main__":
    main()
