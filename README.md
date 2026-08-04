# Lebanese Dialect → MSA Translation (CNN Seq2Seq)

Translates informal Beirut/Levantine Arabic into Modern Standard Arabic (MSA)
using a character-level CNN encoder-decoder (ConvS2S-style, with attention).

Data: MADAR Corpus-6 (Beirut ↔ MSA), Bouamor et al. 2018.

## Files
- `build_dataset.py` — extracts aligned Beirut↔MSA pairs from the raw MADAR TSVs into train/dev/test.tsv
- `data.py` — vocab building + PyTorch Dataset
- `model.py` — CNN encoder-decoder model
- `train.py` — training loop, saves `best_model.pt`
- `evaluate.py` — computes BLEU/chrF on test set + prints example translations
- `demo.py` — interactive: type a sentence, get the MSA translation

## Setup (run once)

1. Install Python 3.9+ if you don't have it.
2. Open a terminal in this folder and run:
   ```
   pip install -r requirements.txt
   ```

## Getting the data

1. Download the MADAR corpus zip (you already registered and have the email link).
2. Unzip it. You should get a folder like:
   `MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021/MADAR_Corpus/`
   containing files like `MADAR.corpus.Beirut.tsv`, `MADAR.corpus.MSA.tsv`, etc.
3. Put that unzipped folder **next to this project folder** (one level up), so the
   structure looks like:
   ```
   your_workspace/
     madar_data/
       MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021/
         MADAR_Corpus/
           MADAR.corpus.Beirut.tsv
           MADAR.corpus.MSA.tsv
           ...
     project/              <- this folder (all the .py files)
   ```
   If you put it somewhere else, edit the `BASE` path at the top of `build_dataset.py`.

## Step-by-step: how to run everything

Open a terminal, `cd` into this project folder, then run each command in order:

```bash
# 1. Build train/dev/test.tsv from the raw MADAR files
python3 build_dataset.py

# 2. Train the model (takes ~20-30 min on a laptop CPU, faster with a GPU)
python3 train.py

# 3. Evaluate on the test set (prints examples + BLEU/chrF scores)
python3 evaluate.py

# 4. Try it interactively
python3 demo.py
```

Training prints progress per epoch, e.g.:
```
epoch 01 | train_loss 2.72 | dev_loss 2.26 | 83.6s
```
Loss should steadily decrease. The best checkpoint (lowest dev loss) is saved to
`best_model.pt` automatically — `evaluate.py` and `demo.py` load this file.

## Notes for your report
- Dataset: MADAR Corpus-6, Beirut dialect ↔ MSA, 9,000 train / 1,000 dev / 2,000 test pairs.
- Model: character-level CNN encoder-decoder with attention (ConvS2S-style), ~741K parameters.
- Tokenization: character-level was chosen because Lebanese dialect has no standardized
  spelling, which would cause severe out-of-vocabulary issues at the word level.
- Limitation to mention: MADAR's Beirut subset is travel/tourism-domain sentences (BTEC),
  not naturalistic social-media Lebanese text — worth noting as future work (e.g. combining
  with social-media dialect data for broader coverage).
