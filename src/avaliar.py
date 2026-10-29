"""Compara: modelo base (zero-shot), base (few-shot) e base + LoRA.

Métricas: JSON válido (%) e exact match (%) no conjunto de teste.
"""
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompt import construir_prompt, extrair_json
from src.treinar import MODELO_BASE, carregar


def normalizar(obj):
    """Ignora a ordem dos itens ao comparar."""
    if not isinstance(obj, dict) or not isinstance(obj.get("itens"), list):
        return None
    try:
        itens = sorted((str(i["nome"]), int(i["quantidade"])) for i in obj["itens"])
    except (KeyError, TypeError, ValueError):
        return None
    return itens, obj.get("observacoes")


@torch.no_grad()
def gerar(modelo, tok, prompt: str) -> str:
    entrada = tok(prompt, return_tensors="pt").to(modelo.device)
    saida = modelo.generate(
        **entrada, max_new_tokens=128, do_sample=False, pad_token_id=tok.eos_token_id
    )
    return tok.decode(saida[0][entrada["input_ids"].shape[1]:], skip_special_tokens=True)


def avaliar(modelo, tok, exemplos, few_shot: bool) -> dict:
    validos = exatos = 0
    for ex in exemplos:
        obj = extrair_json(gerar(modelo, tok, construir_prompt(ex["texto"], few_shot)))
        norm = normalizar(obj)
        validos += norm is not None
        exatos += norm is not None and norm == normalizar(ex["saida"])
    n = len(exemplos)
    return {"json_valido": round(100 * validos / n, 1), "exact_match": round(100 * exatos / n, 1), "n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=MODELO_BASE)
    ap.add_argument("--adaptador", default="outputs/adaptador")
    ap.add_argument("--n", type=int, default=100, help="quantos exemplos de teste avaliar")
    a = ap.parse_args()

    exemplos = carregar("data/teste.jsonl")[: a.n]
    tok = AutoTokenizer.from_pretrained(a.modelo)
    base = AutoModelForCausalLM.from_pretrained(a.modelo).eval()

    resultados = {
        "base_zero_shot": avaliar(base, tok, exemplos, few_shot=False),
        "base_few_shot": avaliar(base, tok, exemplos, few_shot=True),
    }
    lora = PeftModel.from_pretrained(base, a.adaptador).eval()
    resultados["base_lora"] = avaliar(lora, tok, exemplos, few_shot=False)

    print(json.dumps(resultados, indent=2, ensure_ascii=False))
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/resultados.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
