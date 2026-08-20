import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

corpus = [
    ("excelente servicio y buena atencion", 1),
    ("muy mal producto me llego roto", 0),
    ("me encanto todo perfecto", 1),
    ("no me gusto para nada pésimo", 0),
    ("comida muy rica y rápida", 1),
    ("terrible experiencia no lo recomiendo", 0),
    ("terrible", 0),
    ("excelente", 1),
    ("muy mala comida", 0),
    ("comida fea", 0),
    ("muy rapido y excelente", 0),
    ("me llego roto, pesimo", 0)
]

palabras = set()
for frase, _ in corpus:
    palabras.update(frase.split())

vocab = {palabra: i+2 for i, palabra in enumerate(sorted(list(palabras)))}

vocab['PAD'] =1
vocab['<UNK>'] = 0

vocab_size = len(vocab)
max_len = 6

def dato_a_id(text:str, max_len:int) -> list[int]:
    tokens = [vocab.get(p, vocab["<UNK>"]) for p in text.split()[:max_len]]
    if len(tokens) < max_len:
        tokens += vocab['UNK'] * (max_len - len(tokens))
    return tokens[:max_len] 


X_data = [dato_a_id(frase) for frase, _ in corpus]
Y_data = [label for _, label in corpus]