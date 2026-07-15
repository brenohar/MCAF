# check_gpu.py
from llama_cpp import Llama
import llama_cpp

print(f"llama_cpp versão: {llama_cpp.__version__}")

# Tenta carregar 1 camada na GPU — se CUDA disponível, aparece no log
llm = Llama(
    model_path="../models/Phi-3.5-mini-instruct-Q4_K_M.gguf",
    n_gpu_layers=1,
    n_ctx=512,
    verbose=True
)
print("Carregamento concluído.")