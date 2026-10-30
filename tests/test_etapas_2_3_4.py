"""Automated verification gates for Stages 2, 3, and 4."""
from mcaf.config import RunConfig
from mcaf.shield import RuntimeShield
from mcaf.simulator import SituationSimulator, compute_ground_truth, deterministic_prefilter


def test_etapa3_paired_across_controllers():
    """(a) Dois controladores com a mesma semente veem a mesma sequencia exata."""
    cfg_a = RunConfig(controller="Pure-SLM", scenario="degraded", seed=42, n_cycles=200)
    cfg_b = RunConfig(controller="Hybrid+RAG", scenario="degraded", seed=42, n_cycles=200)
    ep_a = SituationSimulator(cfg_a).generate_episode()
    ep_b = SituationSimulator(cfg_b).generate_episode()
    assert ep_a == ep_b


def test_etapa3_reproducible_across_runs():
    """(b) A mesma sequencia sai identica em duas execucoes separadas."""
    cfg = RunConfig(controller="Hybrid", scenario="isolated", seed=303, n_cycles=200)
    ep_1 = SituationSimulator(cfg).generate_episode()
    ep_2 = SituationSimulator(cfg).generate_episode()
    assert ep_1 == ep_2


def test_etapa3_prefilter_100pct_concordance_10k():
    """(c) Em 10 mil situacoes, pre-filtro concorda com o gabarito em 100% dos criticos."""
    cfg = RunConfig(controller="Hybrid", scenario="isolated", seed=42, n_cycles=10000)
    sim = SituationSimulator(cfg)
    critical_count = 0
    for idx in range(10000):
        sit = sim.next_situation(idx)
        gt = compute_ground_truth(sit, band_mode=cfg.band_mode)
        pf = deterministic_prefilter(sit)
        if sit.is_critical_gt:
            critical_count += 1
            assert pf == gt, f"Divergencia no ciclo {idx}: pf={pf} vs gt={gt}"
        else:
            assert pf is None
    assert critical_count > 1000


def test_etapa4_four_desk_tests_and_weak_until():
    """Testes de mesa automaticos por regra + validacao do Weak Until sem autorizacao."""
    shield = RuntimeShield()

    # 1. Freio a 210 C propondo seguir operando -> emergency_stop (phi_brake)
    v1 = shield.verify_and_enforce(
        {"brake_temp_c": 210.0, "speed_kmh": 10.0, "engine_temp_c": 88.0,
         "operator_command": "continue_operation", "auth_token": False},
        "continue_operation",
    )
    assert v1.intervened and v1.executed_action == "emergency_stop" and v1.blocked_by == "phi_brake"

    # 2a. Marcha a re em movimento -> emergency_stop (phi_gear)
    v2a = shield.verify_and_enforce(
        {"brake_temp_c": 90.0, "speed_kmh": 8.5, "engine_temp_c": 88.0,
         "operator_command": "R", "auth_token": False},
        "continue_operation",
    )
    assert v2a.intervened and v2a.executed_action == "emergency_stop" and v2a.blocked_by == "phi_gear"

    # 2b. Excesso de velocidade (26 km/h > 20 km/h) -> emergency_stop (phi_speed)
    v2b = shield.verify_and_enforce(
        {"brake_temp_c": 90.0, "speed_kmh": 26.0, "engine_temp_c": 88.0,
         "operator_command": "continue_operation", "auth_token": False},
        "continue_operation",
    )
    assert v2b.intervened and v2b.executed_action == "emergency_stop" and v2b.blocked_by == "phi_speed"

    # 3. Motor critico (112 C) propondo seguir operando -> activate_cooling (phi_temp)
    v3 = shield.verify_and_enforce(
        {"brake_temp_c": 90.0, "speed_kmh": 10.0, "engine_temp_c": 112.0,
         "operator_command": "continue_operation", "auth_token": False},
        "continue_operation",
    )
    assert v3.intervened and v3.executed_action == "activate_cooling" and v3.blocked_by == "phi_temp"

    # 4. Operacao de risco sem autorizacao -> refuse_operation (phi_auth)
    v4 = shield.verify_and_enforce(
        {"brake_temp_c": 90.0, "speed_kmh": 10.0, "engine_temp_c": 88.0,
         "operator_command": "Op_risk", "auth_token": False},
        "continue_operation",
    )
    assert v4.intervened and v4.executed_action == "refuse_operation" and v4.blocked_by == "phi_auth"

    # 5. Weak Until: 500 ciclos nominais onde auth_token NUNCA ocorre -> zero falsos positivos
    shield.reset_episode()
    for _ in range(500):
        vn = shield.verify_and_enforce(
            {"brake_temp_c": 85.0, "speed_kmh": 12.0, "engine_temp_c": 89.0,
             "operator_command": "continue_operation", "auth_token": False},
            "continue_operation",
        )
        assert not vn.intervened and vn.executed_action == "continue_operation"
