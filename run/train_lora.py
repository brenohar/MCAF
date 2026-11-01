"""Leak-free LoRA fine-tuning (Attention + SwiGLU MLP) with hard entity-mismatch negatives."""
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
TRAIN_SEEDS = (9101, 9102, 9103, 9104, 9105, 9106, 9107, 9108)
CONV_SCRIPT = Path.home() / "llama-cpp-conv" / "convert_lora_to_gguf.py"


def build_randomized_hot_docs(directive_eqs: Tuple[str, ...]) -> List[Dict[str, str]]:
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

    pos_dir, hard_neg_id, easy_neg_temp, no_rag_nom, crit_cases = [], [], [], [], []

    for t_seed in TRAIN_SEEDS:
        rng = random.Random(t_seed)
        ep_dirs = tuple(sorted(rng.sample(list(ALL_EQUIPMENTS), 6)))
        hot_docs = build_randomized_hot_docs(ep_dirs)
        hot_texts = [f"{d['id']} {d['category']}: {d['text']}" for d in hot_docs]
        hot_emb = retriever.encoder.encode(
            hot_texts, normalize_embeddings=True, show_progress_bar=False
        )

        rcfg = RunConfig(seed=t_seed, n_cycles=600, p_crit=0.16, n_actions=6, band_mode="directive")
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
                crit_cases.append((p_rag if (cyc % 3 != 0) else p_norag, gt))
            else:
                if gt == "schedule_maintenance":
                    pos_dir.append((p_rag, "schedule_maintenance"))
                elif 90.0 <= sit.engine_temp_c <= 105.0:
                    hard_neg_id.append((p_rag, "continue_operation"))
                else:
                    easy_neg_temp.append((p_rag, "continue_operation"))
                no_rag_nom.append((p_norag, "continue_operation"))

    mix_rng = random.Random(20261031)
    for bucket in (pos_dir, hard_neg_id, easy_neg_temp, no_rag_nom, crit_cases):
        mix_rng.shuffle(bucket)

    dataset = (
        pos_dir[:320]
        + hard_neg_id[:360]
        + easy_neg_temp[:140]
        + no_rag_nom[:220]
        + crit_cases[:160]
    )
    mix_rng.shuffle(dataset)
    print(
        f"[DATASET] {len(dataset)} exemplos livres de vazamento "
        f"(pos={len(pos_dir[:320])}, hard_neg_id={len(hard_neg_id[:360])}, "
        f"easy_neg={len(easy_neg_temp[:140])}, no_rag={len(no_rag_nom[:220])}, crit={len(crit_cases[:160])}).",
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
            self.items.append((p_ids + t_ids, ([-100] * len(p_ids)) + t_ids))

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
    print("[VERIFY GGUF+CUDA] Adaptador Attention+MLP validado! Saida:", repr(res["choices"][0]["text"]), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(12)
    torch.manual_seed(2026)

    hf_cache_base = MODELS_DIR / "hf_qwen25_05b_base"
    tok = AutoTokenizer.from_pretrained(hf_cache_base)
    base_model = AutoModelForCausalLM.from_pretrained(hf_cache_base, dtype=torch.float32)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(base_model, lora_cfg)

    pairs = generate_leak_free_dataset()
    ds = MaskedChatDataset(pairs, tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=lambda b: collate_pad(b, pad_id))

    total_steps = len(loader) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=2.5e-4, total_steps=total_steps, pct_start=0.10
    )
    model.train()

    step_i = 0
    for ep in range(1, args.epochs + 1):
        win_loss, win_n = 0.0, 0
        for in_ids, att_m, lbls in loader:
            step_i += 1
            opt.zero_grad()
            loss = model(input_ids=in_ids, attention_mask=att_m, labels=lbls).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            win_loss += float(loss.item())
            win_n += 1
            if step_i % 50 == 0 or step_i == total_steps:
                print(
                    f"[LORA ep={ep}/{args.epochs} step={step_i}/{total_steps}] "
                    f"loss={win_loss / max(win_n, 1):.4f} lr={sched.get_last_lr()[0]:.2e}",
                    flush=True,
                )
                win_loss, win_n = 0.0, 0

    peft_out = MODELS_DIR / "lora_qwen25_05b_peft"
    gguf_out = MODELS_DIR / "Qwen2.5-0.5B-MCAF-LoRA-f16.gguf"
    shutil.rmtree(peft_out, ignore_errors=True)
    peft_out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(peft_out)
    convert_and_verify_gguf(hf_cache_base, peft_out, gguf_out)


if __name__ == "__main__":
    main()
