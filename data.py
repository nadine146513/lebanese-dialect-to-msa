import torch
from torch.utils.data import Dataset

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
SPECIALS = [PAD, SOS, EOS, UNK]
MAX_LEN = 100  # covers ~p95 of both src/tgt lengths with margin


def read_pairs(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            src, tgt = line.split("\t", 1)
            if src.strip() and tgt.strip():
                pairs.append((src.strip(), tgt.strip()))
    return pairs


def build_vocab(pairs):
    chars = set()
    for src, tgt in pairs:
        chars.update(src)
        chars.update(tgt)
    itos = SPECIALS + sorted(chars)
    stoi = {ch: i for i, ch in enumerate(itos)}
    return stoi, itos


def encode(text, stoi, max_len=MAX_LEN, add_sos_eos=True):
    ids = [stoi.get(ch, stoi[UNK]) for ch in text[: max_len - 2]]
    if add_sos_eos:
        ids = [stoi[SOS]] + ids + [stoi[EOS]]
    return ids


class TranslationDataset(Dataset):
    def __init__(self, pairs, stoi, max_len=MAX_LEN):
        self.pairs = pairs
        self.stoi = stoi
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_ids = encode(src, self.stoi, self.max_len, add_sos_eos=True)
        tgt_ids = encode(tgt, self.stoi, self.max_len, add_sos_eos=True)
        return torch.tensor(src_ids), torch.tensor(tgt_ids)


def collate_fn(batch, pad_idx):
    srcs, tgts = zip(*batch)
    src_max = max(len(s) for s in srcs)
    tgt_max = max(len(t) for t in tgts)
    src_pad = torch.full((len(batch), src_max), pad_idx, dtype=torch.long)
    tgt_pad = torch.full((len(batch), tgt_max), pad_idx, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_pad[i, : len(s)] = s
        tgt_pad[i, : len(t)] = t
    return src_pad, tgt_pad
