# DeepLearning HW3 — Spoken-SQuAD Extractive QA

A minimal, reproducible BERT baseline for extractive question answering over **ASR-transcribed** passages (Spoken-SQuAD). Trains a span-prediction head on top of `bert-base-uncased`, logs **EM/F1/WER**, and saves figures/tables for the report.

## Project Structure

DL_HW3/
├─ Bert.py # main training+evaluation script (report-ready outputs)
├─ README.md
├─ run_outputs/ # figures, tables, logs (created after running)
│ ├─ figures/ # plots: loss, EM/F1, WER, length histograms
│ ├─ tables/ # history.json, sample preds per epoch
│ └─ base_model_wer.txt # WER per epoch
└─ Spoken-SQuAD-master/ # place dataset JSONs here (or pass paths via args)


## Dataset
Spoken-SQuAD JSON files:
- `spoken_train-v1.1.json`
- `spoken_test-v1.1.json`

Put them under `Spoken-SQuAD-master/` **or** pass absolute paths via CLI flags.

## Environment
Python 3.10+, CUDA (optional), and:
```bash
pip install torch transformers tqdm matplotlib
# if using evaluate/jiwer variants:
pip install evaluate jiwer

## Train and Evaluation
python3 Bert.py \
  --train_json /path/to/spoken_train-v1.1.json \
  --valid_json /path/to/spoken_test-v1.1.json

