from pathlib import Path
from llama_cpp import Llama

# Diretório base do projeto (~/mcaf-poc)
BASE_DIR = Path(__file__).resolve().parent.parent

# Caminho do modelo Gemma 2B (GGUF) no Raspberry Pi 5
MODEL_PATH = str(BASE_DIR / "models" / "gemma-2b-it-Q4_0.gguf")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=512,       # pode reduzir depois se quiser testar desempenho
    n_gpu_layers=0,   # CPU-only no Raspberry Pi 5
    verbose=False,
)

out = llm(
    "Explique em poucas linhas o que é o framework MCAF.\n",
    max_tokens=64,
)

print(out["choices"][0]["text"])
