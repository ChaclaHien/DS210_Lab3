import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLstmEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layer, n_labels, padding_idx=0, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=n_layer,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, n_labels)    # 512 -> 21

    def forward(self, inputs, lengths):
        x = self.embedding(inputs)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        # total_length để giữ đúng độ dài max_len của batch
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=inputs.shape[1])
        out = self.dropout(out)
        return self.classifier(out)     # (bs, len, n_labels)
