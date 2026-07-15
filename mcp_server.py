# src/mcp_server.py — Módulo 1: Simulador de Sensores e Ferramentas MCP
import random
import time

random.seed(42)

# Estado global dos sensores
sensor_state = {
    "engine_temp_c":     90.0,
    "speed_kmh":          0.0,
    "brake_temp_c":      80.0,
    "tire_temp_c":       55.0,
    "authorized":        True,
    "operational_state": "idle"
}

# Parâmetros nominais por estado (Tabela 6.2 da dissertação)
OPERATIONAL_PARAMS = {
    "idle":           {"engine": (70,  85),  "brake": (40,  70),  "tire": (40, 60), "speed": (0,  0)},
    "cruising":       {"engine": (85,  95),  "brake": (80,  150), "tire": (50, 70), "speed": (25, 40)},
    "loaded_climb":   {"engine": (90, 105),  "brake": (150, 250), "tire": (60, 80), "speed": (10, 25)},
    "loaded_descent": {"engine": (85, 100),  "brake": (200, 400), "tire": (60, 85), "speed": (20, 40)},
    "braking_event":  {"engine": (90, 110),  "brake": (400, 700), "tire": (70, 95), "speed": (0,  20)},
}

def update_sensor_state(force_state=None, p_crit=0.15):
    """
    Atualizado para cumprir o Passo 1.6 (p_crit parametrizável) 
    e Passo 1.5 (emulação de comandos do operador).
    """
    state = force_state or random.choice(list(OPERATIONAL_PARAMS.keys()))
    sensor_state["operational_state"] = state
    p = OPERATIONAL_PARAMS[state]
    
    def sample(lo, hi):
        v = random.uniform(lo, hi)
        return round(v, 0)
        
    sensor_state["engine_temp_c"] = sample(*p["engine"])
    sensor_state["brake_temp_c"]  = sample(*p["brake"])
    sensor_state["tire_temp_c"]   = sample(*p["tire"])
    sensor_state["speed_kmh"]     = sample(*p["speed"])

    # --- INJEÇÃO DA LÓGICA DE AUDITORIA SBESC ---
    is_critical = random.random() < p_crit
    
    # Estado padrão do operador
    sensor_state["operator_command"] = "continue_operation"
    sensor_state["auth_token"] = False

    if is_critical:
        # Força a violação de uma das 4 regras formais
        issue = random.choice(["engine_hot", "brake_hot", "reverse_moving", "unauthorized_risk"])
        if issue == "engine_hot":
            sensor_state["engine_temp_c"] = random.uniform(106.0, 115.0)
        elif issue == "brake_hot":
            sensor_state["brake_temp_c"] = random.uniform(201.0, 250.0)
        elif issue == "reverse_moving":
            sensor_state["speed_kmh"] = random.uniform(5.0, 15.0)
            sensor_state["operator_command"] = "R"
        elif issue == "unauthorized_risk":
            sensor_state["operator_command"] = "Op_risk"
            sensor_state["auth_token"] = False
    else:
        # Força o motor para a "faixa de trade-off" para testarmos a inteligência da IA
        sensor_state["engine_temp_c"] = random.uniform(90.0, 105.0)
        
        # Ocasionalmente o operador pede um comando de risco, mas POSSUI o token (ação segura)
        if random.random() < 0.1:
            sensor_state["operator_command"] = "Op_risk"
            sensor_state["auth_token"] = True

def nominal_sensor_state():
    return {
        "engine_temp_c":     88.0,
        "speed_kmh":          0.0,
        "brake_temp_c":     120.0,
        "tire_temp_c":       62.0,
        "authorized":        True,
        "operational_state": "cruising"
    }

# ── Ferramentas MCP (chamadas diretas para a PoC) ─────────────────────────────

def read_engine_temp():
    update_sensor_state()
    return sensor_state["engine_temp_c"]

def read_vehicle_speed():
    return sensor_state["speed_kmh"]

def read_brake_temp():
    return sensor_state["brake_temp_c"]

def read_tire_temp():
    return sensor_state["tire_temp_c"]

def read_all_sensors():
    update_sensor_state()
    return dict(sensor_state)

def engage_reverse():
    t0 = time.perf_counter()
    ok = sensor_state["speed_kmh"] == 0.0
    return {"ok": ok, "latency_ms": round((time.perf_counter() - t0) * 1e3, 4)}

def activate_cooling():
    t0 = time.perf_counter()
    sensor_state["engine_temp_c"] = max(sensor_state["engine_temp_c"] - 5.0, 70.0)
    return {"ok": True, "latency_ms": round((time.perf_counter() - t0) * 1e3, 4)}

def emergency_stop():
    t0 = time.perf_counter()
    sensor_state["speed_kmh"] = 0.0
    return {"ok": True, "latency_ms": round((time.perf_counter() - t0) * 1e3, 4)}

# Catálogo de ferramentas disponíveis (schema MCP simplificado)
MCP_TOOLS = {
    "read_engine_temp":  read_engine_temp,
    "read_vehicle_speed": read_vehicle_speed,
    "read_brake_temp":   read_brake_temp,
    "read_tire_temp":    read_tire_temp,
    "read_all_sensors":  read_all_sensors,
    "engage_reverse":    engage_reverse,
    "activate_cooling":  activate_cooling,
    "emergency_stop":    emergency_stop,
}

def call_tool(name, **kwargs):
    if name not in MCP_TOOLS:
        return {"error": f"Tool '{name}' not found"}
    return MCP_TOOLS[name](**kwargs)
