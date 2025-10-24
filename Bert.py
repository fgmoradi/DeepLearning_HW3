
import os, json, math, random, time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import BertModel, BertTokenizerFast
from torch.optim import AdamW

import matplotlib.pyplot as plt


SEED = 42
MAX_LENGTH = 512
MODEL_PATH = "bert-base-uncased"
LR = 2e-5
WEIGHT_DECAY = 2e-2
EPOCHS = 3
TRAIN_BS = 8
VALID_BS = 1
WARMUP_STEPS = 0  
FOCAL_GAMMA = 1
OUTDIR = "run_outputs"

TRAIN_JSON = "./data/spoken_train-v1.1.json"
VALID_JSON = "./data/spoken_test-v1.1.json"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(f"{OUTDIR}/figures", exist_ok=True)
os.makedirs(f"{OUTDIR}/tables", exist_ok=True)

# reproducibility
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def plot_and_save_hist(data, title, xlabel, outpath, bins=20):
    plt.figure()
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_and_save_curve(xs, ys_list, labels, title, xlabel, ylabel, outpath):
    plt.figure()
    for ys, lab in zip(ys_list, labels):
        plt.plot(xs, ys, marker="o", label=lab)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    if len(ys_list) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def get_data(path):
    with open(path, "rb") as f:
        raw = json.load(f)

    contexts, questions, answers = [], [], []
    num_q, num_pos, num_imp = 0, 0, 0

    for group in raw["data"]:
        for paragraph in group["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                num_q += 1
                # if "is_impossible" in qa and qa["is_impossible"]:
                #     num_imp += 1
                # else:
                #     num_pos += 1
                for answer in qa["answers"]:
                    contexts.append(context.lower())
                    questions.append(question.lower())
                    answers.append(answer)  # dict with text + answer_start
    return num_q, num_pos, num_imp, contexts, questions, answers

print("Loading data…")
num_q, num_pos, num_imp, train_contexts, train_questions, train_answers = get_data(TRAIN_JSON)
num_qv, num_posv, num_impv, valid_contexts, valid_questions, valid_answers = get_data(VALID_JSON)

def add_answer_end(answers, contexts):
    for ans, ctx in zip(answers, contexts):
        ans["text"] = ans["text"].lower()
        ans["answer_end"] = ans["answer_start"] + len(ans["text"])

add_answer_end(train_answers, train_contexts)
add_answer_end(valid_answers, valid_contexts)

# Dataset stats (for report)
ctx_len_words = [len(x.strip().split()) for x in train_contexts]
q_len_words   = [len(x.strip().split()) for x in train_questions]
ans_len_words = [len(a["text"].strip().split()) for a in train_answers]

print("Train sizes:", len(train_contexts), len(train_questions), len(train_answers))
print("Valid sizes:", len(valid_contexts), len(valid_questions), len(valid_answers))

plot_and_save_hist(ctx_len_words, "Distribution of Context Lengths", "Words", f"{OUTDIR}/figures/hist_context_len.png")
plot_and_save_hist(q_len_words,   "Distribution of Question Lengths", "Words", f"{OUTDIR}/figures/hist_question_len.png")
plot_and_save_hist(ans_len_words, "Distribution of Answer Lengths",   "Words", f"{OUTDIR}/figures/hist_answer_len.png")


tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
train_enc = tokenizer(train_questions, train_contexts, max_length=MAX_LENGTH, truncation=True, padding=True)
valid_enc = tokenizer(valid_questions, valid_contexts, max_length=MAX_LENGTH, truncation=True, padding=True)

def match_answer_positions(enc_batch, answers, split="train"):
    start_positions, end_positions = [], []
    miss = 0
    span_lengths = []

    for idx in range(len(enc_batch["input_ids"])):
        ret_start, ret_end = 0, 0
        ans_ids = tokenizer(answers[idx]["text"], max_length=MAX_LENGTH, truncation=True, padding=True)["input_ids"]

        # simple sub-sequence match inside the tokenized pair
        ids = enc_batch["input_ids"][idx]
        for a in range(len(ids) - len(ans_ids)):
            match = True
            for i in range(1, len(ans_ids) - 1):
                if ans_ids[i] != ids[a + i]:
                    match = False
                    break
            if match:
                ret_start = a + 1
                ret_end   = a + (len(ans_ids) - 2) + 1  # i ends at len(ans_ids)-2
                break

        if ret_start == 0:
            miss += 1
        else:
            span_lengths.append(ret_end - ret_start + 1)

        start_positions.append(ret_start)
        end_positions.append(ret_end)

    enc_batch.update({"start_positions": start_positions, "end_positions": end_positions})
    miss_rate = 100.0 * miss / len(enc_batch["input_ids"])
    print(f"[{split}] span-match failed: {miss} / {len(enc_batch['input_ids'])} ({miss_rate:.2f}%)")
    if span_lengths:
        plot_and_save_hist(span_lengths, f"Span Lengths ({split})", "subword tokens", f"{OUTDIR}/figures/hist_span_len_{split}.png", bins=30)
    return miss, span_lengths

miss_train, span_lengths_train = match_answer_positions(train_enc, train_answers, split="train")
miss_valid, span_lengths_valid = match_answer_positions(valid_enc, valid_answers, split="valid")

class QADataset(Dataset):
    def __init__(self, enc):
        self.enc = enc
    def __getitem__(self, i):
        return {
            "input_ids":       torch.tensor(self.enc["input_ids"][i]),
            "token_type_ids":  torch.tensor(self.enc["token_type_ids"][i]),
            "attention_mask":  torch.tensor(self.enc["attention_mask"][i]),
            "start_positions": torch.tensor(self.enc["start_positions"][i]),
            "end_positions":   torch.tensor(self.enc["end_positions"][i]),
        }
    def __len__(self):
        return len(self.enc["input_ids"])

train_ds = QADataset(train_enc)
valid_ds = QADataset(valid_enc)

train_loader = DataLoader(train_ds, batch_size=TRAIN_BS, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=VALID_BS)


bert = BertModel.from_pretrained(MODEL_PATH)

class QAModel(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.drop = nn.Dropout(0.1)
        self.l1   = nn.Linear(768 * 2, 768 * 2)
        self.l2   = nn.Linear(768 * 2, 2)
        self.head = nn.Sequential(self.drop, self.l1, nn.LeakyReLU(), self.l2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        hs  = out.hidden_states
        cat = torch.cat((hs[-1], hs[-3]), dim=-1)
        logits = self.head(cat)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)

model = QAModel(bert).to(device)

def focal_loss_fn(start_logits, end_logits, start_positions, end_positions, gamma=FOCAL_GAMMA):
    smax = nn.Softmax(dim=1)
    lsmax = nn.LogSoftmax(dim=1)
    nll = nn.NLLLoss()
    p_s = smax(start_logits); ip_s = 1 - p_s
    p_e = smax(end_logits);   ip_e = 1 - p_e
    lp_s = lsmax(start_logits); lp_e = lsmax(end_logits)
    fl_s = nll((ip_s**gamma) * lp_s, start_positions)
    fl_e = nll((ip_e**gamma) * lp_e, end_positions)
    return 0.5 * (fl_s + fl_e)

optim = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
total_steps = len(train_loader) * EPOCHS
scheduler = transformers.get_linear_schedule_with_warmup(optim, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)


from evaluate import load as load_metric
wer_metric = load_metric("wer")

def normalize_text(s):
    return " ".join(s.strip().split())

def f1_score(pred, gold):
    # token-level F1
    p_toks = normalize_text(pred).split()
    g_toks = normalize_text(gold).split()
    if len(p_toks) == 0 and len(g_toks) == 0:
        return 1.0
    if len(p_toks) == 0 or len(g_toks) == 0:
        return 0.0
    common = {}
    for t in p_toks:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in g_toks:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_toks)
    recall    = overlap / len(g_toks)
    return 2 * precision * recall / (precision + recall)

def exact_match(pred, gold):
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0


def train_one_epoch(model, loader, epoch_idx):
    model.train()
    losses = []
    start_acc, end_acc = [], []

    pbar = tqdm(loader, desc=f"Train Epoch {epoch_idx}")
    for batch in pbar:
        optim.zero_grad()
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        start_pos      = batch["start_positions"].to(device)
        end_pos        = batch["end_positions"].to(device)

        s_logits, e_logits = model(input_ids, attention_mask, token_type_ids)
        loss = focal_loss_fn(s_logits, e_logits, start_pos, end_pos, gamma=FOCAL_GAMMA)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        scheduler.step()

        with torch.no_grad():
            s_pred = torch.argmax(s_logits, dim=1)
            e_pred = torch.argmax(e_logits, dim=1)
            start_acc.append((s_pred == start_pos).float().mean().item())
            end_acc.append((e_pred == end_pos).float().mean().item())
            losses.append(loss.item())

        pbar.set_postfix(loss=np.mean(losses), s_acc=np.mean(start_acc), e_acc=np.mean(end_acc))

    return float(np.mean(losses)), float(np.mean(start_acc)), float(np.mean(end_acc))

def eval_model(model, loader):
    model.eval()
    preds, refs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            s_true         = batch["start_positions"].to(device)
            e_true         = batch["end_positions"].to(device)

            s_logits, e_logits = model(input_ids, attention_mask, token_type_ids)
            s_pred = torch.argmax(s_logits, dim=1)
            e_pred = torch.argmax(e_logits, dim=1)

            # decode predicted span
            # clamp to ensure e >= s
            s_idx = int(s_pred[0].item())
            e_idx = int(e_pred[0].item())
            if e_idx < s_idx:
                s_idx, e_idx = e_idx, s_idx

            pred_text = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(input_ids[0][s_idx:e_idx])
            )

            s_t = int(s_true[0].item())
            e_t = int(e_true[0].item())
            if e_t < s_t:
                s_t, e_t = e_t, s_t
            gold_text = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(input_ids[0][s_t:e_t])
            )

            preds.append(pred_text if len(pred_text) > 0 else "$")
            refs.append(gold_text if len(gold_text) > 0 else "$")

    # metrics
    wer = wer_metric.compute(predictions=preds, references=refs)
    ems = [exact_match(p, r) for p, r in zip(preds, refs)]
    f1s = [f1_score(p, r) for p, r in zip(preds, refs)]
    metrics = {
        "WER": float(wer),
        "EM":  float(np.mean(ems)),
        "F1":  float(np.mean(f1s)),
        "N":   len(preds)
    }
    return preds, refs, metrics


history = {
    "epoch": [],
    "train_loss": [],
    "train_start_acc": [],
    "train_end_acc": [],
    "val_WER": [],
    "val_EM": [],
    "val_F1": []
}

start_time = time.time()
for ep in range(1, EPOCHS + 1):
    tr_loss, tr_sacc, tr_eacc = train_one_epoch(model, train_loader, ep)
    preds, refs, val_metrics  = eval_model(model, valid_loader)

    history["epoch"].append(ep)
    history["train_loss"].append(tr_loss)
    history["train_start_acc"].append(tr_sacc)
    history["train_end_acc"].append(tr_eacc)
    history["val_WER"].append(val_metrics["WER"])
    history["val_EM"].append(val_metrics["EM"])
    history["val_F1"].append(val_metrics["F1"])

    # save per-epoch predictions sample (first 100)
    sample_rows = []
    for i in range(min(100, len(preds))):
        sample_rows.append({"pred": preds[i], "gold": refs[i]})
    save_json(f"{OUTDIR}/tables/preds_epoch_{ep}.json", sample_rows)

    print(f"[Epoch {ep}] loss={tr_loss:.4f} s_acc={tr_sacc:.4f} e_acc={tr_eacc:.4f} | "
          f"WER={val_metrics['WER']:.4f} EM={val_metrics['EM']:.4f} F1={val_metrics['F1']:.4f}")

total_time = time.time() - start_time
save_json(f"{OUTDIR}/tables/history.json", history)

# also write simple text list for WER (to match your original)
with open(f"{OUTDIR}/base_model_wer.txt", "w", encoding="utf-8") as f:
    for w in history["val_WER"]:
        f.write(str(w) + "\n")

epochs = history["epoch"]
plot_and_save_curve(epochs, [history["train_loss"]], ["Train Loss"], "Training Loss", "Epoch", "Loss",
                    f"{OUTDIR}/figures/curve_train_loss.png")
plot_and_save_curve(epochs, [history["val_WER"]], ["WER"], "Validation WER", "Epoch", "WER",
                    f"{OUTDIR}/figures/curve_val_wer.png")
plot_and_save_curve(epochs, [history["val_EM"], history["val_F1"]], ["EM", "F1"], "Validation EM/F1", "Epoch", "Score",
                    f"{OUTDIR}/figures/curve_val_em_f1.png")


report = {
    "setup": {
        "model": MODEL_PATH,
        "max_length": MAX_LENGTH,
        "epochs": EPOCHS,
        "train_bs": TRAIN_BS,
        "valid_bs": VALID_BS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "linear_warmup_decay",
        "warmup_steps": WARMUP_STEPS,
        "loss": f"focal (gamma={FOCAL_GAMMA})",
        "seed": SEED,
        "device": str(device),
    },
    "data": {
        "train_samples": len(train_contexts),
        "valid_samples": len(valid_contexts),
        "span_match_fail_rate_train_%": round(100.0 * miss_train / len(train_contexts), 2),
        "span_match_fail_rate_valid_%": round(100.0 * miss_valid / len(valid_contexts), 2),
    },
    "results": history,
    "runtime_sec": round(total_time, 2)
}
save_json(f"{OUTDIR}/report_summary.json", report)

# quick human-readable bullets you can paste
with open(f"{OUTDIR}/report_notes.md", "w", encoding="utf-8") as f:
    f.write("# Results (paste-ready)\n\n")
    f.write(f"- Model: {MODEL_PATH}, Max Len: {MAX_LENGTH}, Loss: Focal γ={FOCAL_GAMMA}\n")
    f.write(f"- Optim: AdamW (lr={LR}, weight_decay={WEIGHT_DECAY}) + Linear warmup/decay\n")
    f.write(f"- Epochs: {EPOCHS}, Train BS: {TRAIN_BS}, Valid BS: {VALID_BS}\n")
    f.write(f"- Span-match failures (train/valid): "
            f"{round(100.0*miss_train/len(train_contexts),2)}% / {round(100.0*miss_valid/len(valid_contexts),2)}%\n")
    f.write(f"- Final (epoch {epochs[-1]}): "
            f"WER={history['val_WER'][-1]:.4f}, EM={history['val_EM'][-1]:.4f}, F1={history['val_F1'][-1]:.4f}\n")
    f.write("- Plots saved in ./run_outputs/figures: train loss, WER, EM/F1, and length histograms.\n")

print("\nDone. Files written to:", OUTDIR)
