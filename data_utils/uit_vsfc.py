import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

import json
import os
import string

# gom các mẫu thành batch 
def collate_fn(items: list[dict]) -> dict:
    input_ids = [item["input_ids"] for item in items]
    lengths = torch.tensor([ids.shape[0] for ids in input_ids], dtype=torch.long)
    max_len = int(lengths.max())

    input_ids = torch.stack([
        F.pad(
            input, 
            pad=(0, max_len - input.shape[0]), 
            mode="constant", 
            value=0
        ) for input in input_ids
    ])

    labels = torch.stack([item["label"] for item in items])

    return {
        "input_ids": input_ids,     # (bs, max_len)
        "lengths": lengths,         # (bs, )
        "label": labels             # (bs, )
    }

class Vocab:
    def __init__(self, train_path: str, label_key: str = "topic"):
        self.pad = "<pad>"
        self.unk = "<unk>"
        self.label_key = label_key

        data  = json.load(open(train_path, encoding='utf-8'))
        all_words = set()
        labels = set()
        
        for item in data:
            # chuyển chữ thường, bỏ dấu câu
            sentence = self.preprocess_sentence(item["sentence"])
            all_words.update(sentence.split())
            labels.add(item[label_key])

        # 0 = <pad>, 1 = <unk>, các từ khác bắt đầu từ 2
        self.w2i = {
            self.pad : 0,
            self.unk : 1
        }

        # word -> idx 
        for idx, word in enumerate(sorted(all_words), start=2):
            self.w2i[word] = idx

        # idx -> word 
        self.i2w = {idx: word for word, idx in self.w2i.items()}

        # label -> idx 
        self.l2i = {
            label: idx for idx, label in enumerate(sorted(labels))
        }

        # idx -> label
        self.i2l = {
            idx: label for label, idx in self.l2i.items()
        }

    @property
    def n_labels(self) -> int:
        return len(self.l2i)
    
    @property
    def len(self) -> int:
        return len(self.w2i)        # kích thước vocab -> số dòng của Embedding

    @property
    def pad_idx(self) -> int:
        return self.w2i[self.pad]

    def preprocess_sentence(self, sentence: str) -> str:
        translator = str.maketrans("", "", string.punctuation)  # xóa dấu câu
        sentence = sentence.lower()                             # chuyển chữ thường 
        sentence = sentence.translate(translator)
        return sentence

    # chuyển câu thành tensor số 
    def encode_sentence(self, sentence: str) -> torch.Tensor:
        sentence = self.preprocess_sentence(sentence)
        words = sentence.split()

        # Gán câu thành [<unk>] khi câu chỉ có dấu câu (rỗng sau khi xử lý dấu câu)
        if len(words) == 0:
            words = [self.unk]

        unk_id = self.w2i[self.unk]
        word_ids = [self.w2i.get(word, unk_id) for word in words]

        return torch.tensor(word_ids, dtype=torch.long)

    # Chuyển label thành số  
    def encode_label(self, label: str) -> torch.Tensor:
        label_idx = self.l2i[label]

        return torch.tensor(label_idx, dtype=torch.long)
    
    # chuyển idx -> label 
    def decode_label(self, label_ids: torch.Tensor) -> list:
        label_ids = label_ids.tolist()
        labels = [self.i2l[idx] for idx in label_ids]

        return labels

class UIT_VSFC(Dataset):
    def __init__(self, path: str, vocab: Vocab):
        super().__init__()

        self._data = json.load(open(path, encoding='utf-8'))
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        item = self._data[idx]
        
        sentence = item["sentence"]
        label = item[self.vocab.label_key]

        input_ids = self.vocab.encode_sentence(sentence)
        label_idx = self.vocab.encode_label(label)

        return {
            "input_ids": input_ids,
            "label": label_idx
        }
