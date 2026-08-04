import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial

from data import read_pairs, build_vocab, TranslationDataset, collate_fn, PAD
from model import ConvSeq2Seq

device = torch.device("cpu")

train_pairs = read_pairs("train.tsv")
dev_pairs = read_pairs("dev.tsv")
test_pairs = read_pairs("test.tsv")

stoi, itos = build_vocab(train_pairs)
pad_idx = stoi[PAD]
vocab_size = len(itos)
print("vocab size:", vocab_size)

train_ds = TranslationDataset(train_pairs, stoi)
dev_ds = TranslationDataset(dev_pairs, stoi)

collate = partial(collate_fn, pad_idx=pad_idx)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=64, shuffle=False, collate_fn=collate)

model = ConvSeq2Seq(vocab_size, emb_dim=128, hid_dim=128, n_layers=3, kernel_size=3, pad_idx=pad_idx).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

n_params = sum(p.numel() for p in model.parameters())
print("trainable params:", n_params)


def run_epoch(loader, train_mode=True):
    model.train() if train_mode else model.eval()
    total_loss = 0.0
    n_batches = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        if train_mode:
            optimizer.zero_grad()
        output = model(src, tgt_in)  # [b, t-1, vocab]
        loss = criterion(output.reshape(-1, output.shape[-1]), tgt_out.reshape(-1))

        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


if __name__ == "__main__":
    import os

    n_epochs = 30  # additional epochs to run this session
    start_epoch = 1
    best_dev_loss = float("inf")

    if os.path.exists("best_model.pt"):
        print("Found existing checkpoint, resuming training from it...")
        ckpt = torch.load("best_model.pt", map_location=device)
        model.load_state_dict(ckpt["model_state"])
        best_dev_loss = ckpt.get("best_dev_loss", float("inf"))

    for epoch in range(start_epoch, start_epoch + n_epochs):
        t0 = time.time()
        train_loss = run_epoch(train_loader, train_mode=True)
        with torch.no_grad():
            dev_loss = run_epoch(dev_loader, train_mode=False)
        elapsed = time.time() - t0
        print(f"epoch {epoch:02d} | train_loss {train_loss:.4f} | dev_loss {dev_loss:.4f} | {elapsed:.1f}s")
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save({
                "model_state": model.state_dict(),
                "stoi": stoi,
                "itos": itos,
                "best_dev_loss": best_dev_loss,
            }, "best_model.pt")

    print("done. best dev loss:", best_dev_loss)
