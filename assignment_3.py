import os
import json
import time
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from seqeval.metrics import precision_score as ner_precision, recall_score as ner_recall
from seqeval.metrics import f1_score as ner_f1, classification_report
from seqeval.scheme import IOB2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_utils.uit_vsfc import UIT_VSFC, Vocab, collate_fn
from data_utils.phoner import PhoNER, PhoNERVocab, collate_fn as ner_collate_fn
from models.classifiers.text_classifier import TextClassifier
from models.encoders.bilstm_encoder import BiLstmEncoder

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
CLS_DATA = "datasets/UIT-VSFC/"
NER_DATA = "datasets/PhoNER_COVID19/data/syllable/"
NER_TRAIN = NER_DATA + "train_syllable.json"
NER_DEV = NER_DATA + "dev_syllable.json"
NER_TEST = NER_DATA + "test_syllable.json"
OUT = "results/"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ================= Phân loại (Bài 1, 2) =================

def train_cls(model, dataloader, loss_fn, optimizer):
    model.train()
    losses = []
    for item in tqdm(dataloader):
        input_ids = item["input_ids"].to(device)
        labels = item["label"].to(device)
        logits = model(input_ids, item["lengths"])
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))

@torch.no_grad()
def predict_cls(model, dataloader, loss_fn):
    model.eval()
    gts, pds, losses = [], [], []
    for item in dataloader:
        input_ids = item["input_ids"].to(device)
        labels = item["label"].to(device)
        logits = model(input_ids, item["lengths"])
        losses.append(loss_fn(logits, labels).item())
        gts.extend(labels.tolist())
        pds.extend(logits.argmax(dim=-1).tolist())
    return gts, pds, float(np.mean(losses))

def fit_cls(model, train_dl, dev_dl, loss_fn, optimizer, ckpt, max_epoch=30, patience=5):
    history = {"train_loss": [], "dev_loss": [], "dev_f1": [], "epoch_time": []}
    best_f1, best_epoch, wait = 0, 0, 0

    for epoch in range(1, max_epoch + 1):
        start = time.time()
        train_loss = train_cls(model, train_dl, loss_fn, optimizer)
        gts, pds, dev_loss = predict_cls(model, dev_dl, loss_fn)
        f1 = f1_score(gts, pds, average="macro")

        history["train_loss"].append(train_loss)
        history["dev_loss"].append(dev_loss)
        history["dev_f1"].append(f1)
        history["epoch_time"].append(time.time() - start)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} dev_loss={dev_loss:.4f} dev_f1={f1:.4f}")

        if f1 > best_f1:
            best_f1, best_epoch, wait = f1, epoch, 0
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1
        if wait == patience:
            break

    history["best_epoch"] = best_epoch
    history["best_dev_f1"] = best_f1
    return history

def get_metrics(gts, pds):
    return {
        "accuracy": accuracy_score(gts, pds),
        "precision_macro": precision_score(gts, pds, average="macro", zero_division=0),
        "recall_macro": recall_score(gts, pds, average="macro", zero_division=0),
        "f1_macro": f1_score(gts, pds, average="macro"),
        "f1_per_class": f1_score(gts, pds, average=None).tolist(),
        "confusion_matrix": confusion_matrix(gts, pds).tolist()
    }

def print_metrics(name, m, label_names):
    print(f"===== {name} (test) =====")
    print(f"Accuracy: {m['accuracy']:.4f}")
    print(f"Precision-macro: {m['precision_macro']:.4f}")
    print(f"Recall-macro: {m['recall_macro']:.4f}")
    print(f"F1-macro: {m['f1_macro']:.4f}")
    for n, f in zip(label_names, m["f1_per_class"]):
        print(f"F1 {n}: {f:.4f}")
    print("Confusion matrix:")
    print(np.array(m["confusion_matrix"]))

def run_tfidf():
    train = json.load(open(CLS_DATA + "UIT-VSFC-train.json", encoding="utf-8"))
    test = json.load(open(CLS_DATA + "UIT-VSFC-test.json", encoding="utf-8"))

    # dùng chung label và cách tiền xử lý với LSTM
    vocab = Vocab(CLS_DATA + "UIT-VSFC-train.json", label_key="sentiment")
    x_train = [d["sentence"] for d in train]
    y_train = [vocab.l2i[d["sentiment"]] for d in train]
    x_test = [d["sentence"] for d in test]
    y_test = [vocab.l2i[d["sentiment"]] for d in test]

    tfidf = TfidfVectorizer(preprocessor=vocab.preprocess_sentence)
    x_train = tfidf.fit_transform(x_train)
    x_test = tfidf.transform(x_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_train, y_train)
    pds = clf.predict(x_test)

    metrics = get_metrics(y_test, pds)
    label_names = [str(vocab.i2l[i]) for i in range(vocab.n_labels)]
    print_metrics("TF-IDF", metrics, label_names)
    return {"name": "TF-IDF", "metrics": metrics}

def run_cls(kind):
    set_seed(42)

    vocab = Vocab(CLS_DATA + "UIT-VSFC-train.json", label_key="sentiment")
    train_ds = UIT_VSFC(CLS_DATA + "UIT-VSFC-train.json", vocab)
    dev_ds = UIT_VSFC(CLS_DATA + "UIT-VSFC-dev.json", vocab)
    test_ds = UIT_VSFC(CLS_DATA + "UIT-VSFC-test.json", vocab)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_dl = DataLoader(dev_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = TextClassifier(vocab.len, 256, 5, vocab.n_labels, kind=kind, padding_idx=0).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{kind} params: {n_params}")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss().to(device)
    ckpt = OUT + f"{kind.lower()}_best.pt"

    history = fit_cls(model, train_dl, dev_dl, loss_fn, optimizer, ckpt)

    # load checkpoint tốt nhất rồi chấm test
    model.load_state_dict(torch.load(ckpt))
    gts, pds, _ = predict_cls(model, test_dl, loss_fn)
    metrics = get_metrics(gts, pds)
    label_names = [str(vocab.i2l[i]) for i in range(vocab.n_labels)]
    print_metrics(kind, metrics, label_names)

    # đường loss và F1
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["dev_loss"], label="dev")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["dev_f1"])
    plt.xlabel("epoch")
    plt.ylabel("dev F1-macro")
    plt.savefig(OUT + f"{kind.lower()}_curve.png")
    plt.close()

    # các câu dự đoán sai trên dev
    gts, pds, _ = predict_cls(model, dev_dl, loss_fn)
    with open(OUT + f"{kind.lower()}_dev_errors.txt", "w", encoding="utf-8") as f:
        count = 0
        for i, (g, p) in enumerate(zip(gts, pds)):
            if g != p:
                f.write(f"{dev_ds._data[i]['sentence']} | gold={vocab.i2l[g]} | pred={vocab.i2l[p]}\n")
                count += 1
            if count == 30:
                break

    return {
        "name": kind,
        "metrics": metrics,
        "n_params": n_params,
        "time_per_epoch": float(np.mean(history["epoch_time"])),
        "best_epoch": history["best_epoch"]
    }

# ================= NER (Bài 3) =================

def train_ner(model, dataloader, loss_fn, optimizer, n_labels):
    model.train()
    losses = []
    for item in tqdm(dataloader):
        input_ids = item["input_ids"].to(device)
        labels = item["labels"].to(device)
        logits = model(input_ids, item["lengths"])
        loss = loss_fn(logits.reshape(-1, n_labels), labels.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))

@torch.no_grad()
def predict_ner(model, dataloader, vocab):
    model.eval()
    gts, pds = [], []
    for item in dataloader:
        input_ids = item["input_ids"].to(device)
        preds = model(input_ids, item["lengths"]).argmax(dim=-1).cpu()
        labels = item["labels"]
        # cắt theo length để bỏ pad
        for i, n in enumerate(item["lengths"].tolist()):
            gts.append([vocab.i2l[x] for x in labels[i][:n].tolist()])
            pds.append([vocab.i2l[x] for x in preds[i][:n].tolist()])
    return gts, pds

def run_ner():
    set_seed(42)

    vocab = PhoNERVocab(NER_TRAIN, NER_DEV, NER_TEST)
    train_ds = PhoNER(NER_TRAIN, vocab)
    dev_ds = PhoNER(NER_DEV, vocab)
    test_ds = PhoNER(NER_TEST, vocab)
    print(len(train_ds), len(dev_ds), len(test_ds), vocab.n_labels)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=ner_collate_fn)
    dev_dl = DataLoader(dev_ds, batch_size=32, shuffle=False, collate_fn=ner_collate_fn)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=ner_collate_fn)

    model = BiLstmEncoder(vocab.len, 256, 5, vocab.n_labels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("params:", n_params)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    ckpt = OUT + "bilstm_ner_best.pt"

    best_f1, best_epoch, wait = 0, 0, 0
    epoch_times = []
    for epoch in range(1, 51):
        start = time.time()
        loss = train_ner(model, train_dl, loss_fn, optimizer, vocab.n_labels)
        gts, pds = predict_ner(model, dev_dl, vocab)
        f1 = ner_f1(gts, pds, mode="strict", scheme=IOB2)
        epoch_times.append(time.time() - start)
        print(f"Epoch {epoch}: loss={loss:.4f} dev_entity_f1={f1:.4f}")

        if f1 > best_f1:
            best_f1, best_epoch, wait = f1, epoch, 0
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1
        if wait == 5:
            break

    # chấm test bằng checkpoint tốt nhất
    model.load_state_dict(torch.load(ckpt))
    gts, pds = predict_ner(model, test_dl, vocab)
    p = ner_precision(gts, pds, mode="strict", scheme=IOB2)
    r = ner_recall(gts, pds, mode="strict", scheme=IOB2)
    f1 = ner_f1(gts, pds, mode="strict", scheme=IOB2)
    print("===== BiLSTM NER (test) =====")
    print(f"Precision={p:.4f} Recall={r:.4f} F1={f1:.4f}")
    print(classification_report(gts, pds, mode="strict", scheme=IOB2, digits=4))
    print("Params:", n_params)
    print(f"Time/epoch: {np.mean(epoch_times):.1f}s")
    print("Best dev epoch:", best_epoch, "dev F1:", round(best_f1, 4))

    # các câu dự đoán sai trên dev
    gts, pds = predict_ner(model, dev_dl, vocab)
    with open(OUT + "bilstm_ner_dev_errors.txt", "w", encoding="utf-8") as f:
        count = 0
        for i in range(len(gts)):
            if gts[i] != pds[i]:
                words = dev_ds._data[i]["words"]
                for w, g, pd in zip(words, gts[i], pds[i]):
                    if g != pd:
                        f.write(f"{w}\tgold={g}\tpred={pd}\n")
                f.write("---- " + " ".join(words) + "\n\n")
                count += 1
            if count == 30:
                break

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)

    print("########## BÀI 1 ##########")
    tfidf = run_tfidf()
    lstm = run_cls("LSTM")

    print("\n########## BÀI 2 ##########")
    gru = run_cls("GRU")
    print("\n| Model | F1-macro | Params | Time/epoch (s) | Best dev epoch |")
    print("|---|---|---|---|---|")
    print(f"| TF-IDF | {tfidf['metrics']['f1_macro']:.4f} | - | - | - |")
    for r in [lstm, gru]:
        print(f"| {r['name']} | {r['metrics']['f1_macro']:.4f} | {r['n_params']} | {r['time_per_epoch']:.1f} | {r['best_epoch']} |")

    print("\n########## BÀI 3 ##########")
    run_ner()
