"""
Tokenizador BPE (Byte Pair Encoding) - Implementação do zero
Laboratório 6 - P2: Construindo um Tokenizador BPE e Explorando o WordPiece
"""

# Corpus de treinamento: palavras segmentadas em caracteres com símbolo </w>
vocab = {
    'l o w </w>': 5,
    'l o w e r </w>': 2,
    'n e w e s t </w>': 6,
    'w i d e s t </w>': 3
}


def get_stats(vocab: dict) -> dict:
    """Retorna a frequência de cada par de símbolos adjacentes no vocabulário."""
    pairs = {}
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pairs[pair] = pairs.get(pair, 0) + freq
    return pairs


# Validação: ('e', 's') deve ter contagem 9
stats = get_stats(vocab)
print("=== Motor de Frequências ===")
print(f"Par mais frequente: {max(stats, key=stats.get)} -> {max(stats.values())}")
print(f"Contagem de ('e', 's'): {stats[('e', 's')]}")
assert stats[('e', 's')] == 9
print("Validação OK\n")


def merge_vocab(pair: tuple, v_in: dict) -> dict:
    """Funde todas as ocorrências isoladas do par de símbolos no vocabulário."""
    replacement = ''.join(pair)
    v_out = {}
    for word, freq in v_in.items():
        tokens = word.split()
        merged = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                merged.append(replacement)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        v_out[' '.join(merged)] = freq
    return v_out


# Loop principal de treinamento — K=5 iterações
print("=== Loop de Fusão (K=5) ===")
vocab_atual = dict(vocab)

for i in range(1, 6):
    stats = get_stats(vocab_atual)
    melhor_par = max(stats, key=stats.get)
    vocab_atual = merge_vocab(melhor_par, vocab_atual)
    print(f"Iteração {i} | Par fundido: {melhor_par}")
    print(f"  {vocab_atual}\n")


# Integração com WordPiece via BERT multilíngue
print("=== WordPiece — BERT Multilíngue ===")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

frase = "Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar."
tokens = tokenizer.tokenize(frase)

print(f"Frase: {frase}")
print(f"Tokens: {tokens}")
