"""Gera dados sintéticos: texto livre de pedido -> JSON estruturado."""
import argparse
import json
import random
from pathlib import Path

from src.prompt import para_json

# (nome canônico, gênero)
ITENS = [
    ("pizza calabresa", "f"), ("pizza margherita", "f"), ("pizza de frango", "f"),
    ("hambúrguer", "m"), ("batata frita", "f"), ("coca-cola", "f"),
    ("guaraná", "m"), ("suco de laranja", "m"), ("água com gás", "f"), ("brownie", "m"),
]
PALAVRAS = {"m": {1: "um", 2: "dois"}, "f": {1: "uma", 2: "duas"}}
OBS = [None, None, "sem cebola", "sem gelo", "bem passado", "com bastante queijo", "para viagem"]
ABERTURAS = ["quero", "me vê", "manda", "pode ser", "preciso de", "gostaria de", "traz"]


def plural(nome: str, q: int) -> str:
    if q == 1:
        return nome
    palavras = nome.split()
    palavras[0] = palavras[0] + ("s" if not palavras[0].endswith("s") else "")
    return " ".join(palavras)


def gerar_exemplo(rng: random.Random) -> dict:
    escolhidos = rng.sample(ITENS, rng.randint(1, 3))
    itens, partes = [], []
    for nome, gen in escolhidos:
        q = rng.randint(1, 4)
        if q <= 2 and rng.random() < 0.6:
            qtxt = PALAVRAS[gen][q]
        else:
            qtxt = str(q)
        itens.append({"nome": nome, "quantidade": q})
        partes.append(f"{qtxt} {plural(nome, q)}")

    if len(partes) == 1:
        corpo = partes[0]
    else:
        corpo = ", ".join(partes[:-1]) + " e " + partes[-1]
    obs = rng.choice(OBS)
    texto = f"{rng.choice(ABERTURAS)} {corpo}" + (f" {obs}" if obs else "")
    return {"texto": texto, "saida": {"itens": itens, "observacoes": obs}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_treino", type=int, default=2000)
    ap.add_argument("--n_val", type=int, default=100)
    ap.add_argument("--n_teste", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    total = a.n_treino + a.n_val + a.n_teste
    vistos, exemplos = set(), []
    while len(exemplos) < total:  # textos únicos para não vazar teste no treino
        ex = gerar_exemplo(rng)
        if ex["texto"] not in vistos:
            vistos.add(ex["texto"])
            exemplos.append(ex)

    cortes = {
        "treino": exemplos[: a.n_treino],
        "val": exemplos[a.n_treino : a.n_treino + a.n_val],
        "teste": exemplos[a.n_treino + a.n_val :],
    }
    Path("data").mkdir(exist_ok=True)
    for nome, lista in cortes.items():
        with open(f"data/{nome}.jsonl", "w", encoding="utf-8") as f:
            for ex in lista:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"data/{nome}.jsonl: {len(lista)} exemplos")
    print("Exemplo:", cortes["treino"][0]["texto"], "->", para_json(cortes["treino"][0]["saida"]))


if __name__ == "__main__":
    main()
