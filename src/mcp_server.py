import random
import time

random.seed(42)

# Variável global controlada pelo script de teste (run_overnight_sbesc.py)
GLOBAL_P_CRIT = 0.15

sensor_state = {
    "engine_temp_c":     90.0,
    "speed_kmh":          0.0,
    "brake_temp_c":      80.0,
    "tire_temp_c":       55.0,
    "authorized":        True,
    "operational_state": "idle"
}

def set_p_crit(val: float):
    global GLOBAL_P_CRIT
    GLOBAL_P_CRIT = val

def update_sensor_state(force_state=None):
    # HT-17: p_crit agora controla ESTRITAMENTE a taxa de estados críticos
    is_critical = random.random() < GLOBAL_P_CRIT
    
    if is_critical:
        # Força uma anomalia que vai engatilhar o LogicGuard/Prefilter
        anomaly_type = random.choice(["overtemp", "speeding", "brake_fail"])
        if anomaly_type == "overtemp":
            sensor_state["engine_temp_c"] = random.uniform(106.0, 115.0) # > 105
            sensor_state["speed_kmh"] = random.uniform(0.0, 15.0)
            sensor_state["brake_temp_c"] = random.uniform(50.0, 100.0)
        elif anomaly_type == "speeding":
            sensor_state["engine_temp_c"] = random.uniform(80.0, 100.0)
            sensor_state["speed_kmh"] = random.uniform(21.0, 40.0) # > 20
            sensor_state["brake_temp_c"] = random.uniform(50.0, 100.0)
        else: # brake_fail
            sensor_state["engine_temp_c"] = random.uniform(80.0, 100.0)
            sensor_state["speed_kmh"] = random.uniform(1.0, 15.0) 
            sensor_state["brake_temp_c"] = random.uniform(201.0, 300.0) # > 200 <-- AJUSTADO AQUI
        sensor_state["operational_state"] = "critical_anomaly"

    else:
        # Gera um estado ESTRITAMENTE SEGURO (não aciona o prefiltro)
        sensor_state["engine_temp_c"] = random.uniform(70.0, 104.0)
        sensor_state["speed_kmh"] = random.uniform(0.0, 19.0)
        sensor_state["brake_temp_c"] = random.uniform(40.0, 149.0)
        sensor_state["operational_state"] = "nominal"
        
    sensor_state["engine_temp_c"] = round(sensor_state["engine_temp_c"], 1)
    sensor_state["speed_kmh"] = round(sensor_state["speed_kmh"], 1)
    sensor_state["brake_temp_c"] = round(sensor_state["brake_temp_c"], 1)


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
