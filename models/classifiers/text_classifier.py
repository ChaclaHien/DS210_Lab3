import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layer, n_labels, kind="LSTM", padding_idx=0, dropout=0.2):
        super().__init__()
        self.kind = kind
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)

        if kind == "LSTM":
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=n_layer, batch_first=True, dropout=dropout)
        else:
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=n_layer, batch_first=True, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, n_labels)

    def forward(self, inputs, lengths):
        x = self.embedding(inputs)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        if self.kind == "LSTM":
            _, (h_n, _) = self.rnn(packed)
        else:
            _, h_n = self.rnn(packed)

        # lấy hidden state của lớp cuối
        feats = self.dropout(h_n[-1])
        return self.classifier(feats)     # không softmax vì đã có CrossEntropyLoss
