# Tokenizador BPE

Implementação do algoritmo **Byte Pair Encoding (BPE)** do zero em Python, com exploração do tokenizador **WordPiece** do BERT via Hugging Face.

## Estrutura do Projeto

```
.
├── bpe_tokenizer.py   # Implementação das três tarefas
├── requirements.txt   # Dependências
└── README.md
```

## Como Executar

```bash
pip install -r requirements.txt
python bpe_tokenizer.py
```

## Tarefas Implementadas

### Tarefa 1 — Motor de Frequências

A função `get_stats(vocab)` recebe o corpus de treinamento (palavras já segmentadas em caracteres com o símbolo especial `</w>`) e retorna um dicionário com a frequência de cada par adjacente de símbolos.

**Validação:** o par `('e', 's')` atinge a contagem máxima de **9** (6 ocorrências em *newest* + 3 em *widest*).

### Tarefa 2 — Loop de Fusão

A função `merge_vocab(pair, v_in)` recebe o par mais frequente e o vocabulário atual, substituindo todas as ocorrências isoladas desse par pelo token fundido correspondente.

O **loop principal** executa 5 iterações (`K=5`), imprimindo a cada rodada o par fundido e o estado do vocabulário. Ao final das 5 iterações é possível observar a emergência de tokens morfológicos lógicos, como o sufixo `est</w>`.

### Tarefa 3 — WordPiece com BERT Multilíngue

Utiliza o tokenizador `bert-base-multilingual-cased` do Hugging Face para segmentar a frase:

> "Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar."

## Análise: O que significam os tokens com `##`?

No vocabulário WordPiece, o prefixo `##` indica que aquele fragmento é uma **continuação** — ou seja, não inicia uma nova palavra, mas se une ao token anterior. Por exemplo, a palavra *inconstitucionalmente* pode ser segmentada em `in`, `##constitu`, `##cion`, `##al`, `##mente`: cada peça com `##` representa um sufixo que se encaixa no token precedente.

Esse mecanismo resolve o problema do **vocabulário desconhecido (OOV — Out-Of-Vocabulary)**: em vez de representar toda uma palavra rara como um único token `[UNK]`, o modelo decompõe a palavra em sub-palavras que ele conhece. Palavras que nunca foram vistas durante o treinamento podem, ainda assim, ser aproximadas pela combinação de morfemas já presentes no vocabulário. Isso garante que o modelo nunca "trava" diante de neologismos, nomes próprios ou termos técnicos — basta decompô-los nos pedaços mais granulares disponíveis.
