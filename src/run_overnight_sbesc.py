import time
import random
import pandas as pd
from agent_sasc import AgentSASC
import mcp_server  

# --- Configurações da Rodada SBESC "Forte" ---
CYCLES_PER_RUN = 1000
SEEDS = [42, 101, 202, 303, 404] 
P_CRITS = [0.01, 0.05, 0.10, 0.15, 0.25] 
QUANTS = ["Q2_K", "Q4_K_M", "Q8_0"] 

# Configurações de Hardware/Inference do Raspberry Pi 5
LLAMA_PARAMS = {
    "n_gpu_layers": 0,    
    "n_ctx": 1024,        
    "n_batch": 512,       
    "n_threads": 4,       
    "top_k": 40,
    "max_tokens": 128     
}

def run_experiment(mode_name, seed, p_crit=0.15, quant="Q4_K_M"):
    print(f"Executando {mode_name} | Seed: {seed} | P_crit: {p_crit} | Quant: {quant}")
    
    random.seed(seed)
    model_path = f"models/Qwen2.5-0.5B-Instruct-{quant}.gguf"
    
    # Truque elegante para forçar o mcp_server a usar o p_crit do nosso loop
    original_update = mcp_server.update_sensor_state
    def custom_update(force_state=None, **kwargs):
        original_update(force_state=force_state, p_crit=p_crit)
    mcp_server.update_sensor_state = custom_update
    
    agent = AgentSASC(
        model_path=model_path,
        controller_mode=mode_name,
        **LLAMA_PARAMS
    )
    
    results = []
    for i in range(CYCLES_PER_RUN):
        # A MÁGICA ACONTECE AQUI: Um único comando roda o ciclo OODA completo
        res = agent.run_cycle(connectivity=True)
        
        # Descobre quem tomou a decisão olhando para o tempo de prompt
        route = "prefilter" if res.latency_prompt_ms == 0.0 and res.slm_raw_ms == 0.0 else "slm"
        
        results.append({
            "seed": seed, 
            "controller": mode_name, 
            "p_crit": p_crit, 
            "quant": quant,
            "L_dec_ms": res.latency_dec_ms, 
            "L_LG_ms": res.latency_lg_ms, 
            "slm_raw_completion_ms": res.slm_raw_ms,
            "L_tot_ms": res.latency_tot_ms,
            "route": route, 
            "executed_action": res.action if res.allowed else "emergency_stop", 
            "lg_blocked": not res.allowed
        })
    
    # Restaura a função original para não afetar as próximas rodadas
    mcp_server.update_sensor_state = original_update
    del agent
    return pd.DataFrame(results)

if __name__ == "__main__":
    print("Iniciando Bateria Overnight SBESC 2026...")
    
#    print("\n--- FASE 1: Rodada Principal (5.000 ciclos) ---")
#    main_results = []
#    for mode in ["Hybrid", "Pure-SLM", "Prefilter-only", "Pure-SLM+GBNF"]:
#        for s in SEEDS:
#            if mode == "Prefilter-only" and s != 42: continue
#            df = run_experiment(mode, s)
#            main_results.append(df)

#    pd.concat(main_results).to_csv("data/ht07_main_5000.csv", index=False)
    
#    print("\n--- FASE 2: Sensibilidade p_crit ---")
#    sens_results = [run_experiment("Hybrid", 42, p_crit=p) for p in P_CRITS]
#    pd.concat(sens_results).to_csv("data/ht06_sensitivity.csv", index=False)
    
    print("\n--- FASE 3: Ablação de Quantização ---")
    abl_results = [run_experiment("Pure-SLM", 42, quant=q) for q in QUANTS]
    pd.concat(abl_results).to_csv("data/ht05_ablation.csv", index=False)
    
    print("\nTodos os testes concluídos! Acorde e preencha o LaTeX amanhã.")
