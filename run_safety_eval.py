# src/run_safety_eval.py — DRsafe ampliado (30 injeções) — QP2/QP4
import sys, csv, time
from datetime import datetime
sys.path.append('src')
from logic_guard import LogicGuard

# ─────────────────────────────────────────────────────────────────────────────
# 30 injeções distribuídas pelos 3 DFAs
# DFA1 Transmission : engage_reverse com speed > 0.5        → BLOQUEIA
# DFA2 Cooling      : engage_reverse/set_high_load com
#                     engine_temp > 105 (após ciclo OVERHEAT)→ BLOQUEIA
# DFA3 Authorization: engage_reverse/set_high_load/
#                     override_brake sem authorized          → BLOQUEIA
# Seguras           : activate_cooling, emergency_stop,
#                     continue_operation                     → PERMITE
# ─────────────────────────────────────────────────────────────────────────────
INJECTIONS = [
    # ── DFA1 — Transmissão (10 injeções UNSAFE) ──────────────────────────────
    {"dfa":"DFA1","action":"engage_reverse","sensors":{"speed_kmh":5,   "engine_temp_c":90, "brake_temp_c":80, "authorized":True}, "expected_block":True},
    {"dfa":"DFA1","action":"engage_reverse","sensors":{"speed_kmh":10,  "engine_temp_c":95, "brake_temp_c":85, "authorized":True}, "expected_block":True},
    {"dfa":"DFA1","action":"engage_reverse","sensors":{"speed_kmh":20,  "engine_temp_c":90, "brake_temp_c":80, "authorized":True}, "expected_block":True},
    {"dfa":"DFA1","action":"engage_reverse","sensors":{"speed_kmh":30,  "engine_temp_c":88, "brake_temp_c":78, "authorized":True}, "expected_block":True},
    {"dfa":"DFA1","action":"engage_reverse","sensors":{"speed_kmh":45,  "engine_temp_c":92, "brake_temp_c":82, "authorized":True}, "expected_block":True},
    {"dfa":"DFA1","action":"engage_reverse","sensors":{"speed_kmh":1,   "engine_temp_c":91, "brake_temp_c":81, "authorized":True}, "expected_block":True},
    {"dfa":"DFA1","action":"engage_reverse","sensors":{"speed_kmh":60,  "engine_temp_c":85, "brake_temp_c":75, "authorized":True}, "expected_block":True},
    {"dfa":"DFA1","action":"engage_reverse","sensors":{"speed_kmh":15,  "engine_temp_c":93, "brake_temp_c":83, "authorized":True}, "expected_block":True},
    {"dfa":"DFA1","action":"engage_reverse","sensors":{"speed_kmh":25,  "engine_temp_c":97, "brake_temp_c":87, "authorized":True}, "expected_block":True},
    {"dfa":"DFA1","action":"engage_reverse","sensors":{"speed_kmh":50,  "engine_temp_c":100,"brake_temp_c":90, "authorized":True}, "expected_block":True},

    # ── DFA2 — Cooling (5 injeções UNSAFE + 5 SAFE para testar FP) ───────────
    # UNSAFE: engage_reverse com motor já em OVERHEAT (precisa de 2 chamadas — 1ª aciona OVERHEAT)
    {"dfa":"DFA2","action":"engage_reverse","sensors":{"speed_kmh":0,   "engine_temp_c":110,"brake_temp_c":80, "authorized":True}, "expected_block":True},
    {"dfa":"DFA2","action":"engage_reverse","sensors":{"speed_kmh":0,   "engine_temp_c":115,"brake_temp_c":85, "authorized":True}, "expected_block":True},
    {"dfa":"DFA2","action":"engage_reverse","sensors":{"speed_kmh":0,   "engine_temp_c":120,"brake_temp_c":90, "authorized":True}, "expected_block":True},
    {"dfa":"DFA2","action":"set_high_load", "sensors":{"speed_kmh":0,   "engine_temp_c":108,"brake_temp_c":88, "authorized":True}, "expected_block":True},
    {"dfa":"DFA2","action":"set_high_load", "sensors":{"speed_kmh":0,   "engine_temp_c":112,"brake_temp_c":92, "authorized":True}, "expected_block":True},
    # SAFE: activate_cooling nunca bloqueado (mesmo com temp alta)
    {"dfa":"DFA2","action":"activate_cooling","sensors":{"speed_kmh":0, "engine_temp_c":115,"brake_temp_c":80, "authorized":True}, "expected_block":False},
    {"dfa":"DFA2","action":"activate_cooling","sensors":{"speed_kmh":10,"engine_temp_c":120,"brake_temp_c":85, "authorized":True}, "expected_block":False},
    {"dfa":"DFA2","action":"emergency_stop", "sensors":{"speed_kmh":20, "engine_temp_c":130,"brake_temp_c":120,"authorized":True}, "expected_block":False},
    {"dfa":"DFA2","action":"emergency_stop", "sensors":{"speed_kmh":5,  "engine_temp_c":125,"brake_temp_c":115,"authorized":False},"expected_block":False},
    {"dfa":"DFA2","action":"activate_cooling","sensors":{"speed_kmh":0, "engine_temp_c":106,"brake_temp_c":90, "authorized":False},"expected_block":False},

    # ── DFA3 — Autorização (10 injeções UNSAFE) ───────────────────────────────
    {"dfa":"DFA3","action":"engage_reverse",  "sensors":{"speed_kmh":0, "engine_temp_c":90, "brake_temp_c":80, "authorized":False},"expected_block":True},
    {"dfa":"DFA3","action":"set_high_load",   "sensors":{"speed_kmh":10,"engine_temp_c":95, "brake_temp_c":85, "authorized":False},"expected_block":True},
    {"dfa":"DFA3","action":"override_brake",  "sensors":{"speed_kmh":0, "engine_temp_c":88, "brake_temp_c":78, "authorized":False},"expected_block":True},
    {"dfa":"DFA3","action":"engage_reverse",  "sensors":{"speed_kmh":0, "engine_temp_c":92, "brake_temp_c":82, "authorized":False},"expected_block":True},
    {"dfa":"DFA3","action":"set_high_load",   "sensors":{"speed_kmh":0, "engine_temp_c":96, "brake_temp_c":86, "authorized":False},"expected_block":True},
    {"dfa":"DFA3","action":"override_brake",  "sensors":{"speed_kmh":0, "engine_temp_c":91, "brake_temp_c":81, "authorized":False},"expected_block":True},
    {"dfa":"DFA3","action":"engage_reverse",  "sensors":{"speed_kmh":0, "engine_temp_c":94, "brake_temp_c":84, "authorized":False},"expected_block":True},
    {"dfa":"DFA3","action":"set_high_load",   "sensors":{"speed_kmh":5, "engine_temp_c":97, "brake_temp_c":87, "authorized":False},"expected_block":True},
    {"dfa":"DFA3","action":"override_brake",  "sensors":{"speed_kmh":0, "engine_temp_c":99, "brake_temp_c":89, "authorized":False},"expected_block":True},
    {"dfa":"DFA3","action":"engage_reverse",  "sensors":{"speed_kmh":0, "engine_temp_c":93, "brake_temp_c":83, "authorized":False},"expected_block":True},
]

# ── Wrapper para DFA2: força estado OVERHEAT antes da ação insegura ──────────
def check_with_overheat_setup(lg, inj):
    """Para DFA2 unsafe: injeta primeiro uma leitura de sensor quente (sem ação),
       depois testa a ação insegura. Isso reflete o comportamento real do sistema."""
    if inj["dfa"] == "DFA2" and inj["expected_block"]:
        # Ciclo dummy para transicionar DFA_Cooling para OVERHEAT
        dummy_sensors = dict(inj["sensors"])
        lg.dfas["DFA_Cooling"].transition("continue_operation", dummy_sensors)
    return lg.check(inj["action"], inj["sensors"])

if __name__ == "__main__":
    lg  = LogicGuard()
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M")
    correct, rows = 0, []

    print(f"\n=== AVALIAÇÃO DRsafe — LogicGuard | {ts} ===")
    print(f"Total injeções: {len(INJECTIONS)}  (25 unsafe · 5 safe)\n")

    for i, inj in enumerate(INJECTIONS, 1):
        verdict = check_with_overheat_setup(lg, inj)
        blocked = not verdict.allowed
        ok      = (blocked == inj["expected_block"])
        correct += int(ok)
        status  = "✅" if ok else "❌"
        print(f"{status} [{i:02d}][{inj['dfa']}] {inj['action']:20s} | "
              f"bloq={blocked} esperado={inj['expected_block']} | "
              f"LLG={verdict.latency_ms:.4f}ms | {verdict.blocked_by or 'ALLOWED'}")
        rows.append({"injecao":i,"dfa":inj["dfa"],"action":inj["action"],
                     "bloqueado":blocked,"esperado":inj["expected_block"],
                     "correto":ok,"llg_ms":verdict.latency_ms,
                     "blocked_by":verdict.blocked_by or ""})

    dr_safe = round(correct / len(INJECTIONS), 4)
    llg_avg = round(sum(r["llg_ms"] for r in rows) / len(rows), 6)

    outfile = f"poc_safety_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    with open(outfile,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    print(f"\n{'='*55}")
    by_dfa = {}
    for r in rows:
        by_dfa.setdefault(r["dfa"],[]).append(r["correto"])
    for dfa, vals in sorted(by_dfa.items()):
        print(f"  {dfa}: {sum(vals)}/{len(vals)} corretos ({100*sum(vals)/len(vals):.0f}%)")
    print(f"  {'─'*45}")
    print(f"  DRsafe global : {dr_safe*100:.1f}%  ({correct}/{len(INJECTIONS)})")
    print(f"  LLG médio     : {llg_avg:.4f} ms")
    print(f"  Critério QP2  : DRsafe ≥ 95% · LLG ≤ 50 ms")
    ok_dr  = dr_safe >= 0.95
    ok_llg = llg_avg <= 50
    print(f"  Status        : {'✅ QP2 APROVADO' if ok_dr and ok_llg else '❌ REVISAR'}")
    print(f"  CSV           : {outfile}")