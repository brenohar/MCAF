# src/agent_sasc.py - Modulo 4: Agente SASC + Ciclo OODA + Fallback CLP
import time
import sys
import concurrent.futures  # <-- Import adicionado para o Watchdog
from dataclasses import dataclass
from llama_cpp import Llama, LlamaGrammar

sys.path.append('src')
from mcp_server import read_all_sensors, call_tool
from logic_guard import LogicGuard
from eaco_rag import EACO_RAG

# ---------------------------------------------------------------------------
# CLP Emulado - fallback deterministico (Secao 6.5.1 da dissertacao)
# ---------------------------------------------------------------------------
class PLC_Emulated:
    def __init__(self):
        self.active = False
        self.activations = 0

    def activate(self, reason: str) -> dict:
        self.active = True
        self.activations += 1
        return {
            "plc_active": True,
            "reason": reason,
            "action": "emergency_stop",
            "activations": self.activations,
        }

    def deactivate(self):
        self.active = False

# ---------------------------------------------------------------------------
# Resultado de um ciclo OODA — com subtempos por etapa
# ---------------------------------------------------------------------------
@dataclass
class OODAResult:
    cycle: int
    sensors: dict
    context: str
    raw_response: str
    action: str
    allowed: bool
    blocked_by: str = ""
    plc_fallback: bool = False
    latency_obs_ms: float = 0.0
    latency_orient_ms: float = 0.0
    latency_prompt_ms: float = 0.0
    latency_infer_ms: float = 0.0
    latency_parse_ms: float = 0.0
    latency_dec_ms: float = 0.0
    slm_raw_ms: float = 0.0       # <-- NOVA MÉTRICA: Tempo bruto exigido pelo revisor
    latency_lg_ms: float = 0.0
    latency_act_ms: float = 0.0
    latency_fallback_ms: float = 0.0
    latency_tot_ms: float = 0.0
    delta_mcp_ms: float = 0.0

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
AVAILABLE_ACTIONS = [
    "engage_reverse",
    "activate_cooling",
    "emergency_stop",
    "continue_operation",
]

MAX_RETRIES = 2
SAFETY_BUDGET = 0.80  # reservado para possível uso de orçamento temporal

# ---------------------------------------------------------------------------
# Agente SASC
# ---------------------------------------------------------------------------
class AgentSASC:
    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int,
        n_ctx: int,
        n_batch: int,
        n_threads: int,
        top_k: int,
        max_tokens: int,
        controller_mode: str = "Hybrid", # "Pure-SLM", "Pure-SLM+GBNF", "Hybrid", "Prefilter-only"
    ):
        print(f"SASC: Carregando modelo no modo {controller_mode}...")
        t0 = time.perf_counter()
        
        self.controller_mode = controller_mode
        self.use_deterministic_prefilter = controller_mode in ["Hybrid", "Prefilter-only"]
        self.use_gbnf = controller_mode in ["Pure-SLM+GBNF", "Hybrid"]

        # Só sobe o modelo na RAM se o controlador usar IA
        if controller_mode != "Prefilter-only":
            self.llm = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_threads=n_threads,
                verbose=False,
            )
            self.latency_model_load_ms = round((time.perf_counter() - t0) * 1e3, 4)
        else:
            self.llm = None
            self.latency_model_load_ms = 0.0

        self.rag = EACO_RAG(top_k=top_k)
        self.lg = LogicGuard()
        self.plc = PLC_Emulated()

        self.cycle = 0
        self.history = []
        self.max_tokens = max_tokens

        # Gramática GBNF exigida pelo revisor (Passo 1.3)
        # Salve também uma cópia exata deste texto em 'prompts/action.gbnf' no repositório.
        gbnf_string = r"""
        root   ::= action
        action ::= "continue_operation" | "activate_cooling" | "emergency_stop"
        """
        self.action_grammar = LlamaGrammar.from_string(gbnf_string)

        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        print(f"SASC: Pronto. model_load_ms={self.latency_model_load_ms}")


    # -- OBSERVAR -------------------------------------------------------------
    def _observe(self) -> tuple[dict, float]:
        t0 = time.perf_counter()
        sensors = read_all_sensors()
        lat = round((time.perf_counter() - t0) * 1e3, 4)
        return sensors, lat

    # -- ORIENTAR -------------------------------------------------------------
    def _orient(self, sensors: dict, connectivity: bool = False) -> tuple[dict, float]:
        t0 = time.perf_counter()
        query = (
            f"motor {sensors['engine_temp_c']}C "
            f"freio {sensors['brake_temp_c']}C "
            f"velocidade {sensors['speed_kmh']}kmh "
            f"estado {sensors['operational_state']}"
        )
        result = self.rag.retrieve(query, connectivity=connectivity)
        lat = round((time.perf_counter() - t0) * 1e3, 4)
        if not isinstance(result, dict):
            result = {"context": str(result), "is_ood": False}
        return result, lat

    # -- DECIDIR (Com Watchdog de 500ms) --------------------------------------
    def _decide(
        self, sensors: dict, context: str, is_ood: bool = False
    ) -> tuple[str, str, dict]:
        
        # Se for o baseline determinístico puro, não acorda a IA. 
        # Dá o fallback direto (isso avalia o valor real do SLM no Hybrid)
        if self.controller_mode == "Prefilter-only":
            timings = {"prompt_ms": 0.0, "infer_ms": 0.0, "parse_ms": 0.0, "decide_ms": 0.0, "slm_raw_ms": 0.0}
            return "continue_operation", "STATIC_FALLBACK", timings

        key = (
            round(sensors["engine_temp_c"], 1),
            round(sensors["brake_temp_c"], 1),
            round(sensors["speed_kmh"], 2),
        )

        if key in self.cache:
            self.cache_hits += 1
            action, raw, timings = self.cache[key]
            return action, raw, timings
        else:
            self.cache_misses += 1

        t_p0 = time.perf_counter()
        
        # Prompt fixado e alinhado com as 4 regras formais
        prompt = (
            "You are a safety monitor. Based on the sensor readings, output exactly one of these actions: "
            "continue_operation, activate_cooling, emergency_stop.\n"
            "- operator wants 'Op_risk' without 'auth_token' => continue_operation\n"
            "- operator wants 'R' while speed > 0 => emergency_stop\n"
            "- engine > 105 => activate_cooling\n"
            "- brake > 200 => emergency_stop\n"
            "- otherwise => continue_operation\n"
            f"Sensors: engine={sensors['engine_temp_c']:.1f}, brake={sensors['brake_temp_c']:.1f}, "
            f"speed={sensors['speed_kmh']:.2f}, operator_command={sensors.get('operator_command', 'none')}, auth_token={sensors.get('auth_token', False)}\n"
            "Action:"


        )

        t_p1 = time.perf_counter()

        def _invoke_llm():
            # Parâmetros de amostragem fixos e reprodutíveis (Passo 1.3)
            kwargs = {
                "prompt": prompt,
                "max_tokens": self.max_tokens,
                "temperature": 0.0,
                "top_k": 40,
                "top_p": 0.95,
                "stop": ["\n"]
            }
            if self.use_gbnf:
                kwargs["grammar"] = self.action_grammar
                
            return self.llm(**kwargs)

        action = "continue_operation"
        raw = ""
        slm_raw_time = 0.0

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_invoke_llm)
            try:
                # WATCHDOG de 500ms
                out = future.result(timeout=0.500)
                t_infer = time.perf_counter()
                
                raw = out["choices"][0]["text"].strip()
                
                # Parsing seguro das saídas permitidas
                if raw in ["activate_cooling", "emergency_stop", "continue_operation"]:
                    action = raw
                else:
                    action = "continue_operation"
                    raw = f"HALLUCINATION::{raw}"
                
                slm_raw_time = (t_infer - t_p1) * 1000

            except concurrent.futures.TimeoutError:
                t_infer = time.perf_counter()
                action = "continue_operation" 
                raw = "TIMEOUT::watchdog_500ms"
                
                future.result() 
                t_real_end = time.perf_counter()
                slm_raw_time = (t_real_end - t_p1) * 1000

        t_parse = time.perf_counter()

        timings = {
            "prompt_ms": round((t_p1 - t_p0) * 1e3, 4),
            "infer_ms": round((t_infer - t_p1) * 1e3, 4),
            "parse_ms": round((t_parse - t_infer) * 1e3, 4),
            "decide_ms": round((t_parse - t_p0) * 1e3, 4),
            "slm_raw_ms": round(slm_raw_time, 4)
        }

        self.cache[key] = (action, raw, timings)
        return action, raw, timings


    # -- PRÉ-FILTRO DETERMINÍSTICO --------------------------------------------
    def _deterministic_decision(self, sensors: dict) -> str | None:
        cmd = sensors.get("operator_command", "continue_operation")
        auth = sensors.get("auth_token", False)

        # ORDEM DE PRIORIDADE: Ameaça à vida/física primeiro!
        if sensors["brake_temp_c"] > 200.0:
            return "emergency_stop"
        if sensors["speed_kmh"] > 0.0 and cmd == "R":
            return "emergency_stop"
        if sensors["speed_kmh"] > 20.0:  # <-- REGRA ADICIONADA AQUI
            return "emergency_stop"
        if sensors["engine_temp_c"] > 105.0:
            return "activate_cooling"
        
        # Ameaça de procedimento (sem risco físico imediato) depois
        if cmd == "Op_risk" and not auth:
            return "continue_operation" 
            
        return None

    # -- AGIR (LogicGuard + tool call MCP) ------------------------------------
    def _act(
        self, action: str, sensors: dict
    ) -> tuple[bool, str, float, float, float]:
        t0 = time.perf_counter()
        verdict = self.lg.check(action, sensors)
        t1 = time.perf_counter()

        delta_mcp = 0.0
        if verdict.allowed:
            t_mcp = time.perf_counter()
            if action != "continue_operation":
                call_tool(action)
            delta_mcp = round((time.perf_counter() - t_mcp) * 1e3, 4)

        t2 = time.perf_counter()
        return (
            verdict.allowed,
            verdict.blocked_by or "",
            round((t1 - t0) * 1e3, 4),
            delta_mcp,
            round((t2 - t0) * 1e3, 4),
        )

    # -- CICLO OODA COMPLETO --------------------------------------------------
    def run_cycle(self, connectivity: bool = False) -> OODAResult:
        self.cycle += 1
        t_total = time.perf_counter()

        sensors, lat_obs = self._observe()
        pre_action = self._deterministic_decision(sensors)

        if self.use_deterministic_prefilter and pre_action is not None:
            context = ""
            is_ood = False
            lat_orient = 0.0
            action = pre_action
            raw = pre_action
            dec_stats = {
                "prompt_ms": 0.0,
                "infer_ms": 0.0,
                "parse_ms": 0.0,
                "decide_ms": 0.0,
                "slm_raw_ms": 0.0,
            }
        else:
            rag_result, lat_orient = self._orient(sensors, connectivity)
            context = rag_result.get("context", "")
            is_ood = rag_result.get("is_ood", False)
            action, raw, dec_stats = self._decide(sensors, context, is_ood)

        allowed, blocked_by, lat_lg, delta_mcp, lat_act = self._act(action, sensors)

        retries = 0
        plc_active = False
        lat_fallback = 0.0

        while not allowed and retries < MAX_RETRIES:
            retries += 1
            action, raw, dec_retry = self._decide(sensors, context, is_ood)
            allowed, blocked_by, lat_lg, delta_mcp, lat_act = self._act(action, sensors)
            dec_stats["prompt_ms"] += dec_retry["prompt_ms"]
            dec_stats["infer_ms"] += dec_retry["infer_ms"]
            dec_stats["parse_ms"] += dec_retry["parse_ms"]
            dec_stats["decide_ms"] += dec_retry["decide_ms"]
            dec_stats["slm_raw_ms"] += dec_retry["slm_raw_ms"]

        if not allowed:
            t_fb = time.perf_counter()
            self.plc.activate(reason=f"blocked:{blocked_by} retries:{retries}")
            plc_active = True
            lat_fallback = round((time.perf_counter() - t_fb) * 1e3, 4)

        lat_tot = round((time.perf_counter() - t_total) * 1e3, 4)

        result = OODAResult(
            cycle=self.cycle,
            sensors=sensors,
            context=context[:120] + "..." if len(context) > 120 else context,
            raw_response=raw,
            action=action,
            allowed=allowed,
            blocked_by=blocked_by,
            plc_fallback=plc_active,
            latency_obs_ms=lat_obs,
            latency_orient_ms=lat_orient,
            latency_prompt_ms=dec_stats["prompt_ms"],
            latency_infer_ms=dec_stats["infer_ms"],
            latency_parse_ms=dec_stats["parse_ms"],
            latency_dec_ms=dec_stats["decide_ms"],
            slm_raw_ms=dec_stats["slm_raw_ms"],
            latency_lg_ms=lat_lg,
            latency_act_ms=lat_act,
            latency_fallback_ms=lat_fallback,
            latency_tot_ms=lat_tot,
            delta_mcp_ms=delta_mcp,
        )
        self.history.append(result)
        return result

    # -- METRICAS CONSOLIDADAS ------------------------------------------------
    def metrics(self) -> dict:
        total = len(self.history)
        blocked = sum(1 for r in self.history if not r.allowed)
        fallback = sum(1 for r in self.history if r.plc_fallback)

        def avg(attr: str) -> float:
            return round(sum(getattr(r, attr) for r in self.history) / max(total, 1), 4)

        return {
            "total_cycles": total,
            "blocked": blocked,
            "block_rate": round(blocked / max(total, 1), 4),
            "plc_fallbacks": fallback,
            "avg_obs_ms": avg("latency_obs_ms"),
            "avg_orient_ms": avg("latency_orient_ms"),
            "avg_prompt_ms": avg("latency_prompt_ms"),
            "avg_infer_ms": avg("latency_infer_ms"),
            "avg_parse_ms": avg("latency_parse_ms"),
            "avg_dec_ms": avg("latency_dec_ms"),
            "avg_lg_ms": avg("latency_lg_ms"),
            "avg_act_ms": avg("latency_act_ms"),
            "avg_fallback_ms": avg("latency_fallback_ms"),
            "avg_tot_ms": avg("latency_tot_ms"),
            "model_load_ms": self.latency_model_load_ms,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }
