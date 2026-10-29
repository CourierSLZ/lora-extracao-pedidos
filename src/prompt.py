"""Formato de prompt compartilhado entre treino e avaliação (precisa ser idêntico)."""
import json

INSTRUCAO = "Extraia o pedido em JSON com as chaves 'itens' (nome, quantidade) e 'observacoes'.\n\n"

EXEMPLOS_FEW_SHOT = [
    ("quero 2 pizzas calabresa e uma coca-cola sem gelo",
     {"itens": [{"nome": "pizza calabresa", "quantidade": 2}, {"nome": "coca-cola", "quantidade": 1}], "observacoes": "sem gelo"}),
    ("me vê um hambúrguer",
     {"itens": [{"nome": "hambúrguer", "quantidade": 1}], "observacoes": None}),
]


def para_json(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def construir_prompt(texto: str, few_shot: bool = False) -> str:
    base = INSTRUCAO
    if few_shot:
        for t, o in EXEMPLOS_FEW_SHOT:
            base += f"Pedido: {t}\nJSON: {para_json(o)}\n\n"
    return base + f"Pedido: {texto}\nJSON:"


def extrair_json(saida: str):
    """Pega o primeiro objeto JSON da saída do modelo; retorna None se inválido."""
    ini = saida.find("{")
    if ini == -1:
        return None
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(saida[ini:])
        return obj
    except json.JSONDecodeError:
        return None
