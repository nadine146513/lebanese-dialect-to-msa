# Lebanese Dialect → MSA Translation (CNN Seq2Seq)

Translates informal Beirut/Levantine Arabic into Modern Standard Arabic (MSA)
using a character-level CNN encoder-decoder (ConvS2S-style, with attention).

Built for the **LebNet Tech Fellows Program (2026)** ; AI for Lebanon track.
Motivation: support Arabic literacy and education tools for students who write
in informal Lebanese dialect but need formal Arabic for schoolwork/exams.

Data: MADAR Corpus-6 (Beirut ↔ MSA), Bouamor et al. 2018.

## Files
- `build_dataset.py` : extracts aligned Beirut↔MSA pairs from the raw MADAR TSVs into train/dev/test.tsv
- `data.py` : vocab building + PyTorch Dataset
- `model.py` : CNN encoder-decoder model
- `train.py` : training loop, saves `best_model.pt` (resumes automatically if a checkpoint exists)
- `evaluate.py` : computes BLEU/chrF on test set + prints example translations
- `demo.py` : interactive: type a sentence, get the MSA translation

## Setup (run once)

1. Install Python 3.9+ if you don't have it.
2. Open a terminal in this folder and run:
   ```
   pip install -r requirements.txt
   ```


## Getting the data

MADAR is distributed under an academic-use license that does not permit
redistribution, so the raw data is not included in this repo.

1. Register for the corpus at https://camel.abudhabi.nyu.edu/madar-parallel-corpus/
2. Download and unzip it. You should get a folder like:
   `MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021/MADAR_Corpus/`
3. Put that unzipped folder **next to this project folder** (one level up):
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

```bash
# 1. Build train/dev/test.tsv from the raw MADAR files
python build_dataset.py

# 2. Train the model (~20-30 min per 15 epochs on a laptop CPU)
python train.py

# 3. Evaluate on the test set (prints examples + BLEU/chrF scores)
python evaluate.py

# 4. Try it interactively
python demo.py
```

## Results

Trained for 45 epochs total (9,000 train / 1,000 dev / 2,000 test pairs).

| Metric | Value |
|---|---|
| Train loss (final) | 1.32 |
| Dev loss (final) | 1.23 |
| Test BLEU | 2.83 |
| Test chrF | 19.94 |

Loss decreased consistently across all 45 epochs with no sign of overfitting,
suggesting further training would likely continue to improve results.

Qualitative examples (model output vs. gold MSA):

| Beirut dialect | Model output | Gold MSA |
|---|---|---|
| وين القهوة؟ | أين القهوة؟ | أين المقهى؟ |
| فيك تصرفلي الشيك تبعي تبع المتين دولار؟ | هل يمكنك أن تصرف بعض الشيك دولاراً؟ | هل يمكنكم صرف الشيك ذو المائتي دولار الخاص بي؟ |

The model correctly learns core translation patterns (interrogative structure,
common vocabulary swaps like وين→أين) but struggles with longer, more complex
sentences and precise numeric/entity details.

## Notes

- Model: character-level CNN encoder-decoder with attention (ConvS2S-style), ~741K parameters.
- Tokenization: character-level was chosen because Lebanese dialect has no standardized
  spelling, which would cause severe out-of-vocabulary issues at the word level.
- Limitations / future work:
  - MADAR's Beirut subset is travel/tourism-domain (BTEC) sentences, not naturalistic
    social-media Lebanese text - broader coverage would need social-media dialect data.
  - BLEU/chrF scores are modest; longer training, beam search decoding, or a larger
    model would likely improve fluency and accuracy further.
  - Current decoding is greedy with a simple repetition guard; beam search is a
    natural next step.
