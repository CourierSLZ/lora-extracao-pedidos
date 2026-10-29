# Extração de pedidos em JSON com LoRA

Fine-tuning com **LoRA** de um LLM pequeno (`Qwen2.5-0.5B-Instruct`) para converter texto livre
("quero 2 pizzas calabresa e uma coca sem gelo") em **JSON estruturado**.
O projeto compara três abordagens no mesmo conjunto de teste:

1. Modelo base, zero-shot
2. Modelo base com prompt few-shot
3. Modelo base + adaptador LoRA

## Como rodar
```bash
pip install -r requirements.txt
python -m src.gerar_dados          # dados sintéticos (train/val/test)
python -m src.treinar              # treina o adaptador (GPU recomendada)
python -m src.avaliar --n 100      # compara as 3 abordagens
pytest -q
```
Para publicar no Hugging Face: `huggingface-cli login` e depois
`PeftModel.from_pretrained(...).push_to_hub("seu-usuario/lora-extracao-pedidos")`.

## Resultados (teste, n=100)
| Abordagem | JSON válido | Exact match |
|---|---|---|
| Base zero-shot | _preencher_ | _preencher_ |
| Base few-shot | _preencher_ | _preencher_ |
| Base + LoRA | _preencher_ | _preencher_ |

## Limitações
- Dados sintéticos de templates: o modelo pode não generalizar para falas reais (gírias, erros de digitação).
- Itens fixos do cardápio; itens novos exigem mais dados.

## Próximos passos
- Adicionar ruído (typos, abreviações) e pedidos reais anotados.
- Decodificação restrita a JSON (ex.: outlines) para chegar a 100% de JSON válido.
