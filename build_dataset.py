import csv

BASE = "../madar_data/MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021/MADAR_Corpus"

def load_tsv(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows[(r["sentID.BTEC"], r["split"])] = r["sent"]
    return rows

bei = load_tsv(f"{BASE}/MADAR.corpus.Beirut.tsv")
msa = load_tsv(f"{BASE}/MADAR.corpus.MSA.tsv")

pairs_by_split = {"train": [], "dev": [], "test": []}

def split_bucket(split_name):
    if split_name == "corpus-6-train":
        return "train"
    if split_name == "corpus-6-dev":
        return "dev"
    if split_name.startswith("corpus-6-test"):
        return "test"
    return None

count = 0
for key, bei_sent in bei.items():
    if key in msa:
        sid, split = key
        bucket = split_bucket(split)
        if bucket is None:
            continue
        msa_sent = msa[key]
        if bei_sent.strip() and msa_sent.strip():
            pairs_by_split[bucket].append((bei_sent.strip(), msa_sent.strip()))
            count += 1

for bucket, pairs in pairs_by_split.items():
    with open(f"{bucket}.tsv", "w", encoding="utf-8") as f:
        for src, tgt in pairs:
            f.write(f"{src}\t{tgt}\n")
    print(bucket, len(pairs))

print("total aligned pairs:", count)
