import random

from src.gerar_dados import gerar_exemplo, plural
from src.prompt import extrair_json, para_json


def test_plural():
    assert plural("pizza calabresa", 2) == "pizzas calabresa"
    assert plural("coca-cola", 1) == "coca-cola"


def test_exemplo_consistente():
    ex = gerar_exemplo(random.Random(0))
    assert ex["saida"]["itens"], "deve ter ao menos um item"
    assert all(i["quantidade"] >= 1 for i in ex["saida"]["itens"])


def test_extrair_json():
    assert extrair_json(' {"a":1} lixo') == {"a": 1}
    assert extrair_json("sem json") is None
    assert extrair_json('{"a":') is None


def test_roundtrip():
    obj = {"itens": [{"nome": "brownie", "quantidade": 2}], "observacoes": None}
    assert extrair_json(para_json(obj)) == obj
