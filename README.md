# Lab 3: Mạng neural hồi quy (RNN)

Bài thực hành 3 – xây dựng, huấn luyện và đánh giá các mạng LSTM, GRU và BiLSTM bằng PyTorch. Tất cả mô hình dùng **Adam** làm optimizer, 5 lớp, hidden size 256, có packing, chọn checkpoint tốt nhất theo F1 trên tập dev rồi chấm trên tập test.

| Bài | Mô hình | Bộ dữ liệu | Độ đo |
|-----|---------|-----------|-------|
| 1 | LSTM 5 lớp (+ baseline TF-IDF) | UIT-VSFC (sentiment) | accuracy, precision/recall/F1 macro |
| 2 | GRU 5 lớp (so sánh với Bài 1) | UIT-VSFC (sentiment) | accuracy, precision/recall/F1 macro |
| 3 | Encoder 5 lớp BiLSTM | PhoNER_COVID19 (syllable) | precision/recall/F1 mức thực thể (seqeval strict IOB2) |

## Cấu trúc thư mục

```
.
├── assignment_3.py              # File chạy chính (Bài 1 → 3)
├── data_utils/
│   ├── __init__.py
│   ├── uit_vsfc.py              # Vocab, Dataset, collate_fn cho UIT-VSFC (Bài 1, 2)
│   └── phoner.py                # Vocab, Dataset, collate_fn cho PhoNER (Bài 3)
├── models/
│   ├── __init__.py
│   ├── classifiers/
│   │   ├── __init__.py
│   │   └── text_classifier.py   # TextClassifier(kind="LSTM" | "GRU")
│   └── encoders/
│       ├── __init__.py
│       └── bilstm_encoder.py    # BiLstmEncoder (5 lớp BiLSTM + Linear 512 -> 21)
├── datasets/
│   ├── UIT-VSFC/
│   │   ├── UIT-VSFC-train.json
│   │   ├── UIT-VSFC-dev.json
│   │   └── UIT-VSFC-test.json
│   └── PhoNER_COVID19/          # clone riêng, không đẩy lên repo
│       └── data/syllable/
│           ├── train_syllable.json
│           ├── dev_syllable.json
│           └── test_syllable.json
├── results/                     # tự sinh khi chạy (checkpoint, hình, file lỗi)
├── requirements.txt
└── README.md
```

## Cài đặt

```bash
pip install -r requirements.txt
```

Yêu cầu Python 3.9+. Nên dùng GPU (CUDA); code tự chọn `cuda` nếu có, ngược lại dùng `cpu` (rất chậm vì mạng 5 lớp).

## Chuẩn bị dữ liệu

- **UIT-VSFC (Bài 1, 2):** đã có sẵn 3 file JSON trong `datasets/UIT-VSFC/` (11.426 train, 1.583 dev, 3.166 test). Chỉ dùng `sentence` làm đầu vào và `sentiment` làm nhãn.
- **PhoNER_COVID19 (Bài 3):** clone về thư mục `datasets/PhoNER_COVID19` (bản syllable: 5.027 train, 2.000 dev, 3.000 test):

```bash
git clone https://github.com/VinAIResearch/PhoNER_COVID19.git datasets/PhoNER_COVID19
ls datasets/PhoNER_COVID19/data/syllable
```

## Cách chạy

### Chạy trên máy local

```bash
git clone https://github.com/ChaclaHien/DS210_Lab3.git
cd DS210_Lab3
pip install -r requirements.txt
git clone https://github.com/VinAIResearch/PhoNER_COVID19.git datasets/PhoNER_COVID19
python assignment_3.py
```

Một lần chạy sẽ thực hiện lần lượt Bài 1 (TF-IDF, LSTM), Bài 2 (GRU + bảng so sánh) và Bài 3 (BiLSTM NER).

### Chạy trên Kaggle

1. Tạo Notebook mới. Ở **Settings** bên phải, đặt **Accelerator = GPU (T4 hoặc P100)** và **Internet = On**.
2. Chạy lần lượt các cell:

```python
%cd /kaggle/working
!rm -rf DS210_Lab3
!git clone https://github.com/ChaclaHien/DS210_Lab3.git
%cd /kaggle/working/DS210_Lab3
!pip install seqeval
```

```python
!git clone https://github.com/VinAIResearch/PhoNER_COVID19.git datasets/PhoNER_COVID19
!ls datasets/PhoNER_COVID19/data/syllable
```

```python
import torch
print(torch.cuda.is_available())   # phải in True
```

```python
!python assignment_3.py
```

3. Tải kết quả về:

```python
!zip -r results.zip results
```

Sau đó tải `results.zip` ở tab **Output** bên phải.

## Cấu hình huấn luyện

| Bài | Batch size | Learning rate | Epochs tối đa | Early stopping | Ghi chú |
|-----|-----------|---------------|---------------|----------------|---------|
| 1 | 32 | 1e-3 | 30 | patience 5 (dev F1-macro) | LSTM 5 lớp, hidden 256, dropout 0.2, lấy hidden state lớp cuối |
| 2 | 32 | 1e-3 | 30 | patience 5 (dev F1-macro) | GRU 5 lớp, hidden 256, khởi tạo lại seed 42 và model |
| 3 | 32 | 1e-3 | 50 | patience 5 (dev entity-F1) | BiLSTM 5 lớp, hidden 256 mỗi chiều, `CrossEntropyLoss(ignore_index=-100)` |

Seed cố định là 42. Checkpoint tốt nhất theo dev được nạp lại trước khi chấm test. TF-IDF dùng Logistic Regression với cùng cách tiền xử lý và cùng chia tập.

## File kết quả sinh ra trong `results/`

| File | Nội dung |
|------|----------|
| `lstm_curve.png`, `gru_curve.png` | Đường loss (train/dev) và dev F1 theo epoch |
| `lstm_dev_errors.txt`, `gru_dev_errors.txt` | Các câu dự đoán sai trên dev để phân tích |
| `bilstm_ner_dev_errors.txt` | Các token dự đoán sai trên dev để phân tích lỗi NER |
| `*_best.pt` | Checkpoint tốt nhất theo dev |

## Kết quả (trên tập test)

**Bài 1, 2: phân loại cảm xúc UIT-VSFC**

| Mô hình | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---------|----------|-------------------|----------------|-----------|
| TF-IDF + LogReg | 0.8806 | 0.8004 | 0.6456 | 0.6514 |
| LSTM 5 lớp | 0.8986 | 0.7933 | 0.7308 | 0.7514 |
| GRU 5 lớp | 0.9005 | 0.7864 | 0.7246 | 0.7447 |

F1 từng lớp:

| Mô hình | negative | neutral | positive |
|---------|----------|---------|----------|
| TF-IDF | 0.8972 | 0.1481 | 0.9088 |
| LSTM | 0.9139 | 0.4151 | 0.9252 |
| GRU | 0.9190 | 0.3893 | 0.9258 |

So sánh LSTM và GRU:

| Mô hình | F1-macro | Số tham số | Thời gian/epoch | Epoch tốt nhất (dev) |
|---------|----------|-----------|-----------------|----------------------|
| TF-IDF | 0.6514 | - | - | - |
| LSTM | 0.7514 | 3.269.123 | 6.2 s | 3 |
| GRU | 0.7447 | 2.611.203 | 5.6 s | 13 |

**Bài 3: NER trên PhoNER_COVID19 (mức thực thể, seqeval strict IOB2)**

| Mô hình | Precision | Recall | F1 (micro) | Số tham số | Thời gian/epoch | Epoch tốt nhất (dev) |
|---------|-----------|--------|-----------|-----------|-----------------|----------------------|
| BiLSTM 5 lớp | 0.9058 | 0.8834 | 0.8945 | 8.241.429 | 7.8 s | 14 |

F1 từng loại thực thể:

| Thực thể | F1 | Thực thể | F1 |
|----------|----|----------|----|
| AGE | 0.9251 | ORGANIZATION | 0.8154 |
| DATE | 0.9531 | PATIENT_ID | 0.9520 |
| GENDER | 0.9436 | SYMPTOM_AND_DISEASE | 0.8022 |
| JOB | 0.4820 | TRANSPORTATION | 0.8684 |
| LOCATION | 0.9017 | NAME | 0.6874 |

> Kết quả có thể thay đổi nhẹ giữa các lần chạy do GPU và thứ tự batch ngẫu nhiên. Lớp neutral (UIT-VSFC) và thực thể JOB, NAME (PhoNER) có ít mẫu nên F1 dao động nhiều hơn các lớp còn lại.
