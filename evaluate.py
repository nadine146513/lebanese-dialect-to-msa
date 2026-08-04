import torch
from data import read_pairs, encode, MAX_LEN, PAD, SOS, EOS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def greedy_decode(model, src_text, stoi, itos, max_len=MAX_LEN):
    model.eval()
    src_ids = encode(src_text, stoi, max_len, add_sos_eos=True)
    src = torch.tensor(src_ids).unsqueeze(0).to(device)

    with torch.no_grad():
        enc_conved, enc_combined = model.encoder(src)

    tgt_ids = [stoi[SOS]]
    for _ in range(max_len):
        tgt = torch.tensor(tgt_ids).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model.decoder(tgt, enc_conved, enc_combined)
        next_id = output[0, -1].argmax().item()
        if next_id == stoi[EOS]:
            break
        tgt_ids.append(next_id)

    chars = [itos[i] for i in tgt_ids[1:]]  # skip SOS
    return "".join(chars)


if __name__ == "__main__":
    from model import ConvSeq2Seq

    ckpt = torch.load("best_model.pt", map_location=device)
    stoi, itos = ckpt["stoi"], ckpt["itos"]
    vocab_size = len(itos)
    pad_idx = stoi[PAD]

    model = ConvSeq2Seq(vocab_size, emb_dim=128, hid_dim=128, n_layers=3, kernel_size=3, pad_idx=pad_idx).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_pairs = read_pairs("test.tsv")

    # ---- qualitative examples ----
    print("=== Sample translations ===")
    for src, gold in test_pairs[:10]:
        pred = greedy_decode(model, src, stoi, itos)
        print(f"BEI : {src}")
        print(f"PRED: {pred}")
        print(f"GOLD: {gold}")
        print("-" * 40)

    # ---- BLEU / chrF over the full test set ----
    try:
        import sacrebleu
    except ImportError:
        print("Run: pip install sacrebleu --break-system-packages")
        raise SystemExit

    preds, golds = [], []
    for src, gold in test_pairs:
        preds.append(greedy_decode(model, src, stoi, itos))
        golds.append(gold)

    bleu = sacrebleu.corpus_bleu(preds, [golds])
    chrf = sacrebleu.corpus_chrf(preds, [golds])
    print(f"\nTest BLEU: {bleu.score:.2f}")
    print(f"Test chrF: {chrf.score:.2f}")

    with open("test_predictions.tsv", "w", encoding="utf-8") as f:
        for src, pred, gold in zip([p[0] for p in test_pairs], preds, golds):
            f.write(f"{src}\t{pred}\t{gold}\n")
    print("Saved all predictions to test_predictions.tsv")
