"""Leak-free LoRA fine-tuning on disjoint seeds with episode-randomized directives."""
import argparse
from dataclasses import replace
from pathlib import Path
import random
import shutil
import subprocess
import sys
from typing import Dict, List, Tuple

import numpy as np
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llama_cpp import Llama  # noqa: E402
from mcaf.config import (  # noqa: E402
    ACTIONS_RICH,
    ALL_EQUIPMENTS,
    EngineConfig,
    MODELS_DIR,
    RunConfig,
    SEEDS,
)
from mcaf.prompts import build_chatml_prompt  # noqa: E402
from mcaf.rag import CORPUS_DOCUMENTS, NeuralRetriever  # noqa: E402
from mcaf.simulator import SituationSimulator, compute_ground_truth  # noqa: E402

HF_BASE_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_SEEDS = (9101, 9102, 9103, 9104, 9105, 9106)
CONV_SCRIPT = Path.home() / "llama-cpp-conv" / "convert_lora_to_gguf.py"


def build_randomized_hot_docs(directive_eqs: Tuple[str, ...]) -> List[Dict[str, str]]:
    """Builds 11 hot docs where D01-D06 target the episode-sampled trucks (zero ID leakage)."""
    base_hot = [d for d in CORPUS_DOCUMENTS if d["id"].startswith("H")]
    faults = (
        "rear wheel hub bearing wear",
        "auxiliary pump cavitation",
        "radiator core dust fouling",
        "cooling fan valve hysteresis",
        "exhaust manifold thermal alert",
        "turbocharger oil-return inspection",
    )
    dyn_directives = []
    for idx, (eq_id, fault_desc) in enumerate(zip(directive_eqs, faults), start=1):
        dyn_directives.append({
            "id": f"D0{idx}",
            "partition": "hot",
            "category": f"directive_{eq_id}",
            "text": (
                f"Maintenance directive bulletin for Truck {eq_id}: {fault_desc} "
                f"flagged on {eq_id}. When {eq_id} operates in intermediate engine "
                f"temperature band (90.0 C to 105.0 C), required action for {eq_id} "
                f"is schedule_maintenance."
            ),
        })
    return base_hot + dyn_directives


def generate_leak_free_dataset() -> List[Tuple[str, str]]:
    assert set(TRAIN_SEEDS).isdisjoint(set(SEEDS)), "Vazamento de semente detectado!"
    retriever = NeuralRetriever(top_k=2)

    pos_dir, neg_rag, no_rag_nom, crit_cases = [], [], [], []

    for t_seed in TRAIN_SEEDS:
        rng = random.Random(t_seed)
        ep_dirs = tuple(sorted(rng.sample(list(ALL_EQUIPMENTS), 6)))
        hot_docs = build_randomized_hot_docs(ep_dirs)
        hot_texts = [f"{d['id']} {d['category']}: {d['text']}" for d in hot_docs]
        hot_emb = retriever.encoder.encode(
            hot_texts, normalize_embeddings=True, show_progress_bar=False
        )

        rcfg = RunConfig(seed=t_seed, n_cycles=600, p_crit=0.22, n_actions=6, band_mode="directive")
        sim = SituationSimulator(rcfg)

        for cyc in range(600):
            raw_sit = sim.next_situation(cyc)
            sit = replace(raw_sit, has_directive=(raw_sit.equipment_id in ep_dirs))
            gt = compute_ground_truth(sit, band_mode="directive")
            sdict = sit.to_dict()

            q = retriever.build_cycle_query(sdict)
            q_vec = retriever.encoder.encode([q], normalize_embeddings=True, show_progress_bar=False)[0]
            order = np.argsort(-np.dot(hot_emb, q_vec))[:2]
            ctx = "\n".join(f"[{hot_docs[i]['id']}] {hot_docs[i]['text']}" for i in order)

            p_rag = build_chatml_prompt(sdict, ctx, ACTIONS_RICH, few_shot_k=0, band_mode="directive")
            p_norag = build_chatml_prompt(sdict, "", ACTIONS_RICH, few_shot_k=0, band_mode="directive")

            if sit.is_critical_gt:
                crit_cases.append((p_rag if (cyc % 2 == 0) else p_norag, gt))
            else:
                if gt == "schedule_maintenance":
                    pos_dir.append((p_rag, "schedule_maintenance"))
                else:
                    neg_rag.append((p_rag, "continue_operation"))
                no_rag_nom.append((p_norag, "continue_operation"))

    mix_rng = random.Random(20261030)
    mix_rng.shuffle(pos_dir)
    mix_rng.shuffle(neg_rag)
    mix_rng.shuffle(no_rag_nom)
    mix_rng.shuffle(crit_cases)

    dataset = pos_dir[:260] + neg_rag[:320] + no_rag_nom[:140] + crit_cases[:180]
    mix_rng.shuffle(dataset)
    print(
        f"[DATASET] Gerados {len(dataset)} exemplos livres de vazamento "
        f"(pos_dir=260, neg_rag=320, no_rag=140, crit=180).",
        flush=True,
    )
    return dataset


class MaskedChatDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], tok: AutoTokenizer):
        self.items = []
        eos = "<|im_end|>"
        for prompt_str, target_act in pairs:
            p_ids = tok.encode(prompt_str, add_special_tokens=False)
            t_ids = tok.encode(f"{target_act}{eos}", add_special_tokens=False)
            input_ids = p_ids + t_ids
            labels = ([-100] * len(p_ids)) + t_ids
            self.items.append((input_ids, labels))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def collate_pad(batch, pad_id: int):
    max_len = max(len(x[0]) for x in batch)
    in_b, att_b, lbl_b = [], [], []
    for ids, lbls in batch:
        pad_n = max_len - len(ids)
        in_b.append(ids + [pad_id] * pad_n)
        att_b.append([1] * len(ids) + [0] * pad_n)
        lbl_b.append(lbls + [-100] * pad_n)
    return (
        torch.tensor(in_b, dtype=torch.long),
        torch.tensor(att_b, dtype=torch.long),
        torch.tensor(lbl_b, dtype=torch.long),
    )


def convert_and_verify_gguf(hf_base_dir: Path, peft_dir: Path, out_gguf: Path) -> None:
    cmd = [
        sys.executable,
        str(CONV_SCRIPT),
        "--base",
        str(hf_base_dir),
        "--outfile",
        str(out_gguf),
        "--outtype",
        "f16",
        str(peft_dir),
    ]
    subprocess.run(cmd, check=True)
    eng = EngineConfig(
        model_size="0.5B",
        quantization="Q4_K_M",
        n_threads=8,
        n_gpu_layers=-1,
        lora_path=str(out_gguf),
    )
    llm = Llama(**eng.llama_init_kwargs())
    res = llm("<|im_start|>user\nPing<|im_end|>\n<|im_start|>assistant\n", max_tokens=4, temperature=0.0)
    print("[VERIFY GGUF+CUDA] Adaptador carregado com sucesso! Saida:", repr(res["choices"][0]["text"]), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(12)
    torch.manual_seed(2026)

    hf_cache_base = MODELS_DIR / "hf_qwen25_05b_base"
    if not (hf_cache_base / "config.json").exists():
        print(f"[HF] Baixando pesos base {HF_BASE_ID} para {hf_cache_base}...", flush=True)
        tok_tmp = AutoTokenizer.from_pretrained(HF_BASE_ID)
        mdl_tmp = AutoModelForCausalLM.from_pretrained(HF_BASE_ID, torch_dtype=torch.float32)
        hf_cache_base.mkdir(parents=True, exist_ok=True)
        tok_tmp.save_pretrained(hf_cache_base)
        mdl_tmp.save_pretrained(hf_cache_base)
        del mdl_tmp

    tok = AutoTokenizer.from_pretrained(hf_cache_base)
    base_model = AutoModelForCausalLM.from_pretrained(hf_cache_base, torch_dtype=torch.float32)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(base_model, lora_cfg)

    if args.preflight_only:
        pf_dir = MODELS_DIR / "tmp_lora_preflight"
        pf_gguf = MODELS_DIR / "tmp_lora_preflight.gguf"
        pf_dir.mkdir(parents=True, exist_ok=True)
        # Inicializa lora_B com valor nao-nulo minimo apenas para testar serializacao de tensores
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_B" in name:
                    param.add_(1e-4)
        model.save_pretrained(pf_dir)
        convert_and_verify_gguf(hf_cache_base, pf_dir, pf_gguf)
        shutil.rmtree(pf_dir, ignore_errors=True)
        pf_gguf.unlink(missing_ok=True)
        print("=== PRE-VOO LORA -> GGUF -> LLAMA_CPP (CUDA) 100% APROVADO ===", flush=True)
        return

    pairs = generate_leak_free_dataset()
    ds = MaskedChatDataset(pairs, tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    loader = DataLoader(
        ds,
        batch_size=8,
        shuffle=True,
        collate_fn=lambda b: collate_pad(b, pad_id),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    model.train()

    total_steps = len(loader) * args.epochs
    step_i = 0
    for ep in range(1, args.epochs + 1):
        running = 0.0
        for in_ids, att_m, lbls in loader:
            step_i += 1
            opt.zero_grad()
            out = model(input_ids=in_ids, attention_mask=att_m, labels=lbls)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item())
            if step_i % 25 == 0 or step_i == total_steps:
                print(
                    f"[LORA ep={ep}/{args.epochs} step={step_i}/{total_steps}] "
                    f"loss={running / 25.0 if step_i % 25 == 0 else float(loss.item()):.4f}",
                    flush=True,
                )
                running = 0.0

    peft_out = MODELS_DIR / "lora_qwen25_05b_peft"
    gguf_out = MODELS_DIR / "Qwen2.5-0.5B-MCAF-LoRA-f16.gguf"
    peft_out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(peft_out)
    convert_and_verify_gguf(hf_cache_base, peft_out, gguf_out)
    print(f"=== TREINAMENTO CONCLUIDO! Adaptador GGUF salvo em: {gguf_out} ===", flush=True)


if __name__ == "__main__":
    main()
