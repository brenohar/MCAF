import pandas as pd
import pathlib
import os

print("Gerando Tabelas Automáticas (HT-20)...")

# Ajuste os nomes conforme os arquivos gerados pela sua bateria de testes
try:
    df_main = pd.read_csv("data/ht07_main_5000.csv")
    df_sens = pd.read_csv("data/ht06_sensitivity.csv")
    df_abl = pd.read_csv("data/ht05_ablation.csv")
except FileNotFoundError:
    print("ERRO: CSVs não encontrados na pasta data/. Rode as baterias primeiro.")
    exit(1)

# Cria a pasta paper se não existir
os.makedirs("paper", exist_ok=True)

# 1. TABELA V - Latência (Desagregada por Cenário - Exigência HT-13)
g = df_main.groupby(["controller", "route"]) # "route" ou "scenario", ajuste conforme o seu log
tab_v = pd.DataFrame({
    "Ldec_mu": g.L_dec_ms.mean(),
    "Ldec_sd": g.L_dec_ms.std(),
    "Ldec_p95": g.L_dec_ms.quantile(.95),
    "LLG_max": g.L_LG_ms.max(),
    "Ltot_mu": g.L_tot_ms.mean(),
    "Ltot_sd": g.L_tot_ms.std(),
    "Ltot_max": g.L_tot_ms.max(),
    "DMR": g.L_tot_ms.apply(lambda s: (s > 500).mean() * 100)
}).round(1)

pathlib.Path("paper/tabV.tex").write_text(tab_v.to_latex())

# 2. TABELA VI - Conformidade e Acurácia (Mesmo log, mesma fonte - Exigência HT-14)
g_conf = df_main.groupby("controller")
tab_vi = pd.DataFrame({
    "SCR": g_conf.apply(lambda d: d.lg_blocked.eq(False).mean() * 100),
    "DC": g_conf.apply(lambda d: (d.executed_action == d.gt_action).mean() * 100), # Ajuste gt_action
    "SLM_Routed": g_conf.route.apply(lambda s: s.eq("slm").mean() * 100)
}).round(1)

pathlib.Path("paper/tabVI.tex").write_text(tab_vi.to_latex())

# 3. MACROS (Variáveis de Texto Automáticas)
macros = {
    "Nciclos": len(df_main),
    "LLGmax": round(df_main.L_LG_ms.max(), 3),
    "MCPoverhead": round((df_main.L_tot_ms - df_main.L_dec_ms - df_main.L_LG_ms).mean(), 3),
    "DMRpure": round(tab_v.loc[("Pure-SLM", slice(None)), "DMR"].mean(), 1)
}

macro_text = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items())
pathlib.Path("paper/macros.tex").write_text(macro_text)

print("SUCESSO: Tabelas e macros exportadas para a pasta paper/.")
