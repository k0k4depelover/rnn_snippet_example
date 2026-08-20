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
    ("muy rapido y excelente", 1),
    ("me llego roto, pesimo", 0),
    ("muy excelente", 1),
    ("me la pase bien", 1)
]



palabras = set()
for frase, _ in corpus:
    palabras.update(frase.split())

vocab = {palabra: i+2 for i, palabra in enumerate(sorted(list(palabras)))}

vocab['<PAD>'] =1
vocab['<UNK>'] = 0

vocab_size = len(vocab)
max_len = 6

def dato_a_id(text:str, max_len:int) -> list[int]:
    tokens = [vocab.get(p, vocab["<UNK>"]) for p in text.split()[:max_len]]
    if len(tokens) < max_len:
        tokens += [vocab['<UNK>']] * (max_len - len(tokens))
    return tokens[:max_len] 


X_data = [dato_a_id(frase, max_len) for frase, _ in corpus]
Y_data = [label for _, label in corpus]

X_tensor = torch.tensor(X_data, dtype=torch.long)
Y_tensor = torch.tensor(Y_data, dtype=torch.float32)


class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X= X
        self.y=y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


loader = DataLoader(TextDataset(X_tensor, Y_tensor), batch_size=2, shuffle=True)

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=8, hidden_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.rc= nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        embedded = self.embedding(x)
        out_seq, h_n = self.rnn(embedded)
        out = self.rc(h_n.squeeze(0))
        return self.sigmoid(out)




if torch.cuda.is_available():
    # Instanciar objeto en GPU
    # Utilizar CUDA para procesamiento paralelo en GPU
    device = torch.device("cuda")
    model = SentimentRNN(vocab_size).to(device)
else:
    model = SentimentRNN(vocab_size)


criterio = nn.BCELoss()
optimizer= torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1, 51):
     total_loss=0.0
     for batch_x, batch_y in loader:
         batch_x = batch_x.to(device)
         batch_y = batch_y.to(device)


         optimizer.zero_grad()
         preds=model(batch_x).squeeze(1)
         loss= criterio(preds, batch_y)
         loss.backward()
         optimizer.step()
         total_loss+=loss.item()

def predecir(frase:str): 
    model.eval()
    with torch.no_grad():
        ids = torch.tensor( [dato_a_id(frase, max_len)], dtype=torch.long).to(device)
        prob = model(ids).item()
        sentimiento = "Positivo" if prob > 0.5 else "Negativo"
        print(f"frase con sentimiento {sentimiento}")

predecir("encantado y feliz")