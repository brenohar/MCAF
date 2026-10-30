"""Auditable single-inference supervisory agent with watchdog and 34-column logging."""
import math
import os
import time
from typing import Dict, Optional

from llama_cpp import Llama, LlamaGrammar, StoppingCriteriaList
import psutil

from mcaf.config import EngineConfig, LOG_COLUMNS, RunConfig
from mcaf.dispatch import TypedInProcessDispatcher
from mcaf.prompts import build_chatml_prompt, build_gbnf_grammar_str, parse_action_from_raw
from mcaf.rag import NeuralRetriever
from mcaf.shield import RuntimeShield
from mcaf.simulator import CycleSituation, deterministic_prefilter


def read_board_temp_c() -> float:
    thermal_file = "/sys/class/thermal/thermal_zone0/temp"
    try:
        with open(thermal_file, "r", encoding="utf-8") as f:
            return round(float(f.read().strip()) / 1000.0, 2)
    except OSError:
        return 0.0


class AuditableAgent:
    """Executes one timed OODA cycle without response reuse, recording all 34 columns."""

    def __init__(
        self,
        engine_cfg: EngineConfig,
        run_cfg: RunConfig,
        shared_llm: Optional[Llama] = None,
        shared_retriever: Optional[NeuralRetriever] = None,
    ):
        self.engine_cfg = engine_cfg
        self.run_cfg = run_cfg
        self.use_prefilter = run_cfg.controller in ("Prefilter-only", "Hybrid", "Hybrid+RAG")
        self.use_slm = run_cfg.controller != "Prefilter-only"
        self.use_rag = run_cfg.controller == "Hybrid+RAG"
        self.use_shield_override = run_cfg.controller in ("Prefilter-only", "Hybrid", "Hybrid+RAG")
        self.use_gbnf = (run_cfg.controller == "Pure-SLM+GBNF") or run_cfg.force_gbnf

        if self.use_slm:
            self.llm = shared_llm or Llama(**engine_cfg.llama_init_kwargs())
        else:
            self.llm = None

        if self.use_rag:
            self.retriever = shared_retriever or NeuralRetriever(top_k=engine_cfg.rag_top_k)
        else:
            self.retriever = None

        self.grammar = (
            LlamaGrammar.from_string(build_gbnf_grammar_str(run_cfg.action_space))
            if self.use_gbnf
            else None
        )
        self.shield = RuntimeShield()
        self.audit_shield = RuntimeShield()
        self.dispatcher = TypedInProcessDispatcher(run_cfg.action_space)
        self._proc = psutil.Process(os.getpid())
        self._proc.cpu_percent(interval=None)

    def reset_episode(self) -> None:
        self.shield.reset_episode()
        self.audit_shield.reset_episode()

    def step(self, sit: CycleSituation, gt_action: str) -> Dict[str, object]:
        t_cycle_start_ns = time.perf_counter_ns()

        t0 = time.perf_counter_ns()
        sit_dict = sit.to_dict()
        pf_action = deterministic_prefilter(sit) if self.use_prefilter else None
        t1 = time.perf_counter_ns()
        t_obs_ms = (t1 - t0) / 1e6

        t2_start = time.perf_counter_ns()
        remote_queried = False
        net_latency_ms = 0.0
        retrieved_docs_str = ""
        context_text = ""

        if pf_action is None and self.use_rag and self.retriever is not None:
            ret = self.retriever.retrieve_for_cycle(sit_dict, self.run_cfg)
            remote_queried = ret.remote_queried
            net_latency_ms = ret.net_latency_ms
            retrieved_docs_str = ";".join(ret.doc_ids)
            context_text = ret.context_text
        t2_end = time.perf_counter_ns()
        t_orient_ms = (t2_end - t2_start) / 1e6

        t3_start = time.perf_counter_ns()
        raw_response = ""
        watchdog_triggered = False

        if pf_action is not None:
            route = "prefilter"
            proposed_action = pf_action
        elif not self.use_slm:
            route = "fallback"
            proposed_action = self.run_cfg.safe_default_action
        else:
            route = "slm"
            prompt = build_chatml_prompt(
                sit_dict=sit_dict,
                context_text=context_text,
                allowed_actions=self.run_cfg.action_space,
                few_shot_k=self.run_cfg.few_shot_k,
                band_mode=self.run_cfg.band_mode,
            )

            gen_kwargs = self.engine_cfg.llama_gen_kwargs()
            gen_kwargs["stop"] = ["<|im_end|>", "\n"]
            if self.grammar is not None:
                gen_kwargs["grammar"] = self.grammar

            if self.run_cfg.watchdog_enabled:
                budget_ns = int(self.run_cfg.watchdog_ms * 1e6)

                def _abort_on_deadline(_input_ids, _scores) -> bool:
                    return (time.perf_counter_ns() - t_cycle_start_ns) >= budget_ns

                gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_abort_on_deadline])

            assert self.llm is not None
            out = self.llm(prompt, **gen_kwargs)
            raw_text = str(out["choices"][0]["text"])
            raw_response = raw_text.replace("\r", " ").replace("\n", "\\n").strip()

            cog_elapsed_ms = (time.perf_counter_ns() - t_cycle_start_ns) / 1e6
            if self.run_cfg.watchdog_enabled and cog_elapsed_ms > self.run_cfg.watchdog_ms:
                watchdog_triggered = True
                proposed_action = self.run_cfg.safe_default_action
            else:
                proposed_action = parse_action_from_raw(
                    raw_text,
                    self.run_cfg.action_space,
                    default=self.run_cfg.safe_default_action,
                )
        t3_end = time.perf_counter_ns()
        t_infer_ms = (t3_end - t3_start) / 1e6

        t4_start = time.perf_counter_ns()
        if self.use_shield_override:
            verdict = self.shield.verify_and_enforce(sit_dict, proposed_action)
            executed_action = verdict.executed_action
            blocked_by = verdict.blocked_by
            safety_ok = True
        else:
            executed_action = proposed_action
            audit_v = self.audit_shield.verify_and_enforce(sit_dict, executed_action)
            blocked_by = ""
            safety_ok = not audit_v.intervened
        t4_end = time.perf_counter_ns()
        t_shield_ms = (t4_end - t4_start) / 1e6

        t5_start = time.perf_counter_ns()
        self.dispatcher.dispatch(
            action=executed_action,
            equipment_id=sit.equipment_id,
            cycle_idx=sit.cycle_idx,
        )
        t5_end = time.perf_counter_ns()
        t_dispatch_ms = (t5_end - t5_start) / 1e6

        t_total_ms = (time.perf_counter_ns() - t_cycle_start_ns) / 1e6

        cpu_pct = round(self._proc.cpu_percent(interval=None), 2)
        rss_mb = round(self._proc.memory_info().rss / (1024.0 * 1024.0), 2)
        board_temp_c = read_board_temp_c()

        row = {
            "model_tag": self.engine_cfg.model_tag,
            "controller": self.run_cfg.controller,
            "scenario": self.run_cfg.scenario,
            "seed": self.run_cfg.seed,
            "band_mode": self.run_cfg.band_mode,
            "cycle_idx": sit.cycle_idx,
            "equipment_id": sit.equipment_id,
            "engine_temp_c": sit.engine_temp_c,
            "brake_temp_c": sit.brake_temp_c,
            "speed_kmh": sit.speed_kmh,
            "operator_command": sit.operator_command,
            "auth_token": sit.auth_token,
            "ma5_rising": sit.ma5_rising,
            "route": route,
            "remote_queried": remote_queried,
            "retrieved_docs": retrieved_docs_str,
            "raw_response": raw_response,
            "proposed_action": proposed_action,
            "executed_action": executed_action,
            "gt_action": gt_action,
            "watchdog_triggered": watchdog_triggered,
            "blocked_by": blocked_by,
            "safety_ok": safety_ok,
            "net_latency_ms": round(net_latency_ms, 6),
            "t_obs_ms": round(t_obs_ms, 6),
            "t_orient_ms": round(t_orient_ms, 6),
            "t_infer_ms": round(t_infer_ms, 6),
            "t_shield_ms": round(t_shield_ms, 6),
            "t_dispatch_ms": round(t_dispatch_ms, 6),
            "t_total_ms": round(t_total_ms, 6),
            "cpu_pct": cpu_pct,
            "rss_mb": rss_mb,
            "board_temp_c": board_temp_c,
            "energy_j": math.nan,
        }
        assert tuple(row.keys()) == LOG_COLUMNS
        return row
