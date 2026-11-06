import json
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

def read_file(path):
    data = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            data.append(json.loads(line))
    return data

def collate_fn(items):
    input_ids = [item["input_ids"] for item in items]
    labels = [item["labels"] for item in items]
    lengths = torch.tensor([len(x) for x in input_ids], dtype=torch.long)
    max_len = int(lengths.max())

    input_ids = torch.stack([F.pad(x, (0, max_len - len(x)), value=0) for x in input_ids])
    # nhãn pad = -100 để loss bỏ qua
    labels = torch.stack([F.pad(x, (0, max_len - len(x)), value=-100) for x in labels])

    return {"input_ids": input_ids, "lengths": lengths, "labels": labels}

class PhoNERVocab:
    def __init__(self, train_path, dev_path, test_path):
        self.pad = "<pad>"
        self.unk = "<unk>"

        train = read_file(train_path)
        words = set()
        for item in train:
            words.update(w.lower() for w in item["words"])

        self.w2i = {self.pad: 0, self.unk: 1}
        for idx, w in enumerate(sorted(words), start=2):
            self.w2i[w] = idx

        # lấy nhãn từ cả 3 tập cho đủ 21 nhãn
        tags = set()
        for path in [train_path, dev_path, test_path]:
            for item in read_file(path):
                tags.update(item["tags"])
        self.l2i = {t: i for i, t in enumerate(sorted(tags))}
        self.i2l = {i: t for t, i in self.l2i.items()}

    @property
    def len(self):
        return len(self.w2i)

    @property
    def n_labels(self):
        return len(self.l2i)

class PhoNER(Dataset):
    def __init__(self, path, vocab):
        super().__init__()
        self._data = read_file(path)
        self.vocab = vocab

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        item = self._data[idx]
        unk = self.vocab.w2i[self.vocab.unk]
        ids = [self.vocab.w2i.get(w.lower(), unk) for w in item["words"]]
        labels = [self.vocab.l2i[t] for t in item["tags"]]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
