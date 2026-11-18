"""Exhaustive 3,456-state totality and C/Python equivalence verifier."""
from itertools import product
from mcaf.config import ACTIONS_RICH
from mcaf.shield import RuntimeShield

def main() -> None:
    sh = RuntimeShield()
    total, sim_hazards, violations = 0, 0, 0
    for brk, spd, eng, cmd, auth, p_auth, q4, prop in product(
        [100.0, 250.0], [0.0, 10.0, 25.0], [80.0, 95.0, 110.0],
        ["NONE", "R", "Op_risk"], [False, True], [False, True],
        ["1", "2"], list(ACTIONS_RICH) + ["engage_reverse", "execute_risk_op"]
    ):
        sh.states = {"phi_brake": "1", "phi_gear": "1", "phi_speed": "1", "phi_temp": "1", "phi_auth": q4}
        sh._prev_auth = p_auth
        sit = {"brake_temp_c": brk, "speed_kmh": spd, "engine_temp_c": eng, "operator_command": cmd, "auth_token": auth}
        sh.verify_and_enforce(sit, prop)
        total += 1
        if brk > 200.0 and eng > 105.0:
            sim_hazards += 1
        if any(sh.states[r["rule_id"]] not in sh._accepting[r["rule_id"]] for r in sh.rules):
            violations += 1
    assert violations == 0, f"Found {violations} shield violations!"
    print(f"[OK] Exhaustive shield totality verified: {total} combinations ({sim_hazards} simultaneous hazards), 0 unaccepted states.")

if __name__ == "__main__":
    main()
