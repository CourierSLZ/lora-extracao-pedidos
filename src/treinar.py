"""Fine-tuning com LoRA de um LLM pequeno para extração de pedidos em JSON."""
import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from src.prompt import construir_prompt, para_json

MODELO_BASE = "Qwen/Qwen2.5-0.5B-Instruct"


def carregar(caminho: str) -> list[dict]:
    with open(caminho, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def tokenizar(ex: dict, tok, max_len: int = 256) -> dict:
    prompt_ids = tok(construir_prompt(ex["texto"]), add_special_tokens=False)["input_ids"]
    alvo_ids = tok(" " + para_json(ex["saida"]), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = (prompt_ids + alvo_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + alvo_ids)[:max_len]  # só calcula loss na resposta
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=MODELO_BASE)
    ap.add_argument("--saida", default="outputs/adaptador")
    ap.add_argument("--epocas", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--r", type=int, default=16)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.modelo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    modelo = AutoModelForCausalLM.from_pretrained(a.modelo)

    lora = LoraConfig(
        r=a.r,
        lora_alpha=2 * a.r,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    modelo = get_peft_model(modelo, lora)
    modelo.print_trainable_parameters()

    treino = Dataset.from_list([tokenizar(e, tok) for e in carregar("data/treino.jsonl")])
    val = Dataset.from_list([tokenizar(e, tok) for e in carregar("data/val.jsonl")])

    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir="outputs/checkpoints",
        num_train_epochs=a.epocas,
        learning_rate=a.lr,
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=25,
        bf16=bf16,
        fp16=cuda and not bf16,
        report_to="none",
        seed=42,
    )
    trainer = Trainer(
        model=modelo,
        args=args,
        train_dataset=treino,
        eval_dataset=val,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )
    trainer.train()
    modelo.save_pretrained(a.saida)
    tok.save_pretrained(a.saida)
    print(f"Adaptador salvo em {a.saida}")


if __name__ == "__main__":
    main()
