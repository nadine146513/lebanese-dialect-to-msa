import torch
import torch.nn as nn
import torch.nn.functional as F

from data import MAX_LEN


class ConvEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hid_dim=128, n_layers=3, kernel_size=3, max_len=MAX_LEN, dropout=0.2):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size, padding=kernel_size // 2)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [batch, src_len]
        batch, src_len = src.shape
        pos = torch.arange(src_len, device=src.device).unsqueeze(0).expand(batch, src_len)
        embedded = self.dropout(self.tok_emb(src) + self.pos_emb(pos))  # [b, s, emb]
        conv_in = self.emb2hid(embedded).permute(0, 2, 1)  # [b, hid, s]

        for conv in self.convs:
            conved = conv(self.dropout(conv_in))
            conved = F.glu(conved, dim=1)  # [b, hid, s]
            conved = (conved + conv_in) * (0.5 ** 0.5)  # residual
            conv_in = conved

        conved = conv_in.permute(0, 2, 1)  # [b, s, hid]
        combined = (self.hid2emb(conved) + embedded) * (0.5 ** 0.5)
        return conved, combined  # conved: for attention keys, combined: for attention values


class ConvDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hid_dim=128, n_layers=3, kernel_size=3, max_len=MAX_LEN, dropout=0.2, pad_idx=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad_idx = pad_idx
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)
        self.attn_hid2emb = nn.Linear(hid_dim, emb_dim)
        self.attn_emb2hid = nn.Linear(emb_dim, hid_dim)
        self.out = nn.Linear(emb_dim, vocab_size)
        self.convs = nn.ModuleList([
            nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def attention(self, embedded, conved, enc_conved, enc_combined):
        conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1))  # [b, t, emb]
        combined = (conved_emb + embedded) * (0.5 ** 0.5)
        energy = torch.matmul(combined, enc_conved.permute(0, 2, 1))  # [b, t, s]
        attention = F.softmax(energy, dim=2)
        attended = torch.matmul(attention, enc_combined)  # [b, t, emb]
        attended_hid = self.attn_emb2hid(attended).permute(0, 2, 1)  # [b, hid, t]
        attended_combined = (conved + attended_hid) * (0.5 ** 0.5)
        return attention, attended_combined

    def forward(self, tgt, enc_conved, enc_combined):
        batch, tgt_len = tgt.shape
        pos = torch.arange(tgt_len, device=tgt.device).unsqueeze(0).expand(batch, tgt_len)
        embedded = self.dropout(self.tok_emb(tgt) + self.pos_emb(pos))
        conv_in = self.emb2hid(embedded).permute(0, 2, 1)  # [b, hid, t]

        for conv in self.convs:
            conv_in_dropped = self.dropout(conv_in)
            padding = torch.zeros(batch, conv_in.shape[1], self.kernel_size - 1, device=tgt.device).fill_(self.pad_idx).float()
            padded = torch.cat((padding, conv_in_dropped), dim=2)  # causal left padding
            conved = conv(padded)
            conved = F.glu(conved, dim=1)
            _, conved = self.attention(embedded, conved, enc_conved, enc_combined)
            conved = (conved + conv_in) * (0.5 ** 0.5)
            conv_in = conved

        conved = conv_in.permute(0, 2, 1)
        output = self.hid2emb(conved)
        output = self.dropout(output)
        return self.out(output)


class ConvSeq2Seq(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hid_dim=128, n_layers=3, kernel_size=3, max_len=MAX_LEN, dropout=0.2, pad_idx=0):
        super().__init__()
        self.encoder = ConvEncoder(vocab_size, emb_dim, hid_dim, n_layers, kernel_size, max_len, dropout)
        self.decoder = ConvDecoder(vocab_size, emb_dim, hid_dim, n_layers, kernel_size, max_len, dropout, pad_idx)

    def forward(self, src, tgt):
        enc_conved, enc_combined = self.encoder(src)
        output = self.decoder(tgt, enc_conved, enc_combined)
        return output
