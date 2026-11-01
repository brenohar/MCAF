"""Stage 10 verification: 100% cycle-by-cycle equivalence between Python and ANSI C shield."""
import json
import random
import subprocess
import sys

from mcaf.config import ACTIONS_RICH, BASE_DIR, RunConfig
from mcaf.shield import RuntimeShield
from mcaf.simulator import SituationSimulator

ACT_TO_INT = {
    "continue_operation": 0,
    "activate_cooling": 1,
    "emergency_stop": 2,
    "refuse_operation": 3,
    "schedule_maintenance": 4,
    "reduce_speed": 5,
}
INT_TO_ACT = {v: k for k, v in ACT_TO_INT.items()}
CMD_TO_INT = {"continue_operation": 0, "R": 1, "Op_risk": 2}
IDX_TO_RULE = {-1: "", 0: "phi_brake", 1: "phi_gear", 2: "phi_speed", 3: "phi_temp", 4: "phi_auth"}


def test_etapa10_c_shield_100pct_equivalent_to_python_and_fast():
    subprocess.run([sys.executable, "analysis/codegen_c_guard.py"], cwd=BASE_DIR, check=True)
    bin_path = BASE_DIR / "c_guard" / "bench"
    subprocess.run(
        [
            "gcc", "-std=c11", "-O3", "-Wall", "-Wextra",
            "c_guard/guard.c", "c_guard/bench.c", "-o", str(bin_path),
        ],
        cwd=BASE_DIR,
        check=True,
    )

    sim = SituationSimulator(RunConfig(seed=42, n_cycles=5000, p_crit=0.25))
    py_shield = RuntimeShield()
    rng = random.Random(2026)

    stdin_lines = []
    expected = []
    for idx in range(5000):
        sit = sim.next_situation(idx)
        prop = rng.choice(ACTIONS_RICH)
        v_py = py_shield.verify_and_enforce(sit.to_dict(), prop)
        expected.append((v_py.executed_action, v_py.blocked_by))
        stdin_lines.append(
            f"{sit.brake_temp_c} {sit.speed_kmh} {sit.engine_temp_c} "
            f"{CMD_TO_INT[sit.operator_command]} {1 if sit.auth_token else 0} {ACT_TO_INT[prop]}"
        )

    proc = subprocess.run(
        [str(bin_path), "--stdin-verify"],
        input="\n".join(stdin_lines) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    out_lines = [ln.strip() for ln in proc.stdout.strip().splitlines() if ln.strip()]
    assert len(out_lines) == 5000
    for i, ln in enumerate(out_lines):
        act_i, blk_i = map(int, ln.split())
        assert (INT_TO_ACT[act_i], IDX_TO_RULE[blk_i]) == expected[i], f"Divergencia no ciclo {i}"

    bench_proc = subprocess.run(
        [str(bin_path), "--iters", "1000000"],
        capture_output=True,
        text=True,
        check=True,
    )
    metrics = json.loads(bench_proc.stdout)
    assert metrics["iterations"] == 1000000
    assert metrics["p99_ns"] < 50000
