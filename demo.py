"""
Interactive demo: type Beirut/Lebanese Arabic dialect, get Modern Standard Arabic.
Run: python3 demo.py
"""
import torch
from data import PAD
from model import ConvSeq2Seq
from evaluate import greedy_decode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load("best_model.pt", map_location=device)
stoi, itos = ckpt["stoi"], ckpt["itos"]
pad_idx = stoi[PAD]

model = ConvSeq2Seq(len(itos), emb_dim=128, hid_dim=128, n_layers=3, kernel_size=3, pad_idx=pad_idx).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

print("Lebanese/Levantine Arabic -> MSA translator")
print("Type a sentence (or 'quit' to exit):\n")

while True:
    text = input("> ").strip()
    if text.lower() in ("quit", "exit"):
        break
    if not text:
        continue
    pred = greedy_decode(model, text, stoi, itos)
    print("MSA:", pred, "\n")
