# Practice 3 — Get started with Hugging Face

Bài thực hành phân tích cảm xúc tiếng Anh với Hugging Face. Kết quả đã chạy:
**82,40% test accuracy**, **83,46% F1** trên 500 mẫu test IMDb.
Xem [báo cáo](report.md), [notebook](Practice3.ipynb) và [minh chứng](results/README.md).
## 1. Nội dung

| Bài | File | Các bước |
|---|---|---|
| Exercise 1 | `exercise1_sentiment.py` | Tải mô hình SST-2 có sẵn, tokenize, in token/ID, dự đoán nhãn và xác suất |
| Exercise 2 | `exercise2_finetune.py` | Tải IMDb, chia dữ liệu, tokenize, cấu hình TrainingArguments, Trainer, fine-tune, đánh giá |
| Dự đoán mới | `predict.py` | Nạp mô hình đã lưu, phân loại câu nhập vào |
| Notebook | `Practice3.ipynb` | Chạy và xem từng bước, có thể dùng để trình bày |

Bài 1 dùng `distilbert/distilbert-base-uncased-finetuned-sst-2-english`.
Bài 2 bắt đầu từ `distilbert/distilbert-base-uncased`, huấn luyện đầu phân loại nhị phân.
IMDb có 25.000 mẫu train và 25.000 mẫu test có nhãn; bỏ phần unsupervised.
Nhãn `0 = NEGATIVE`, `1 = POSITIVE`.

## 2. Cài đặt Windows / VS Code

Mở thư mục repo `LAB_DEEP_LEARNING` bằng **File → Open Folder**.

Lần chạy trong notebook dùng **Python 3.14.0 trên CPU**. Phiên bản thực tế được ghi ở
[requirements-run.txt](requirements-run.txt) và [results/environment.json](results/environment.json).
Để tạo môi trường theo các phiên bản đã chạy:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-run.txt
```

`requirements.txt` là cấu hình Python 3.11 ban đầu, khác với môi trường đã sinh kết quả.
Nếu chọn cấu hình này, tạo môi trường riêng, không trộn hai bộ phiên bản:

Chạy trong terminal PowerShell của VS Code:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Chọn Ctrl+Shift+P → **Python: Select Interpreter** → `.venv\Scripts\python.exe`.
Các lệnh dưới gọi Python trực tiếp nên không cần kích hoạt môi trường hoặc đổi ExecutionPolicy.
Cần Internet lần đầu để tải thư viện, mô hình và IMDb; nên để trống vài GB đĩa.
Mặc định chạy CPU nếu PyTorch không nhận CUDA. Bản PyTorch cài từ pip không bảo đảm có CUDA.
Nếu muốn GPU, cài bản PyTorch phù hợp theo https://pytorch.org/get-started/locally/.
Kiểm tra thiết bị: `.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"`.

## 3. Chạy bài 1

```powershell
.\.venv\Scripts\python.exe exercise1_sentiment.py
.\.venv\Scripts\python.exe exercise1_sentiment.py --text "I love this movie!" "This film was terrible."
```

Lệnh mặc định chạy 5 câu tiếng Anh, cùng danh sách trong notebook; dùng `--text` để nhập các câu khác.
Terminal in văn bản, tokens, input_ids, nhãn và score. Lưu ở `outputs/exercise1.json`.
Score là xác suất softmax của nhãn được chọn, không phải độ chính xác toàn bộ mô hình.

## 4. Chạy bài 2

Thử toàn bộ luồng với ít dữ liệu trước (vẫn tải mô hình và bộ dữ liệu):

```powershell
.\.venv\Scripts\python.exe exercise2_finetune.py --train-size 64 --val-size 32 --test-size 32 --epochs 1 --output-dir outputs/demo
```

Huấn luyện bài thực hành mặc định: 2.000 train, 500 validation, 500 test, 2 epoch:

```powershell
.\.venv\Scripts\python.exe exercise2_finetune.py
```

Chạy dữ liệu đầy đủ, giữ riêng 5.000 validation:

```powershell
.\.venv\Scripts\python.exe exercise2_finetune.py --train-size 20000 --val-size 5000 --test-size 25000 --epochs 3 --output-dir outputs/full
```

Chế độ ít dữ liệu chỉ kiểm tra luồng; không dùng kết quả đó kết luận chất lượng.
Train và validation lấy từ các đoạn không giao nhau của tập train đã shuffle.
Chọn checkpoint theo **validation F1**; chỉ dùng test để đánh giá cuối.
Seed mặc định 42; thiết bị và thư viện vẫn có thể gây chênh lệch kết quả.
Chạy lại cùng `--output-dir` sẽ ghi đè báo cáo/mô hình; dùng thư mục khác để so sánh thí nghiệm.

## 5. Kết quả được tạo sau khi chạy

Trong `outputs/finetune/`:

- `model/`: mô hình và tokenizer đã fine-tune, dùng cho predict.py.
- `metrics.json`: accuracy, precision, recall, F1 và test loss.
- `classification_report.txt`: chỉ số từng lớp và macro/weighted average.
- `confusion_matrix.png`: ma trận nhầm lẫn; hàng là nhãn thật, cột là dự đoán.
- `loss_curve.png`: loss huấn luyện và validation theo epoch.
- `training_history.json`: lịch sử loss và chỉ số validation.
- `config.json`: tham số thí nghiệm.

Precision/recall/F1 trong metrics.json tính với lớp POSITIVE là lớp dương.
Accuracy = số dự đoán đúng / tổng mẫu. F1 là trung bình điều hòa precision và recall.
Kết quả lần chạy đã hoàn thành được đính kèm trong `results/`, gồm cấu hình,
chỉ số, lịch sử, hai biểu đồ và ví dụ dự đoán. Đây là bản sao của lần chạy được báo cáo;
nếu chạy lại thí nghiệm, cần cập nhật minh chứng và báo cáo đồng bộ.

```powershell
.\.venv\Scripts\python.exe predict.py --output outputs/finetune/predictions.json --text "The acting was excellent!" "I regret watching this movie."
```

## 6. Notebook

Mở `Practice3.ipynb`, chọn kernel `.venv`, chạy từng cell từ trên xuống.
Notebook gọi cùng mã nguồn nên chỉ cần sửa code ở một nơi.
Mỗi cell có giải thích mục đích và đã có đầu ra thực tế được lưu.
Nếu chỉ xem kết quả đã nộp, có thể đọc notebook và `results/` mà không huấn luyện lại.
Khi chạy lại, cell huấn luyện có thể chạy lâu trên CPU.

## 7. Lỗi thường gặp

- `No module named ...`: cài requirements bằng đúng `.venv\Scripts\python.exe`.
- Không có Python 3.11: cài Python 3.11 rồi chạy lại lệnh tạo môi trường.
- CUDA out of memory: thêm `--batch-size 2 --max-length 64`, hoặc `--cpu`.
- Không tải được mô hình/dataset: kiểm tra Internet tới huggingface.co; chạy lại sau khi mạng ổn định.
- Cảnh báo classifier weights mới khởi tạo ở bài 2: bình thường, đầu phân loại cần được huấn luyện.
- Muốn ép CPU: thêm `--cpu` vào lệnh bài 2.

## 8. Kiểm tra và phạm vi xác minh

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Đã có đầu ra bài 1, huấn luyện 2 epoch và đánh giá 500 mẫu test.
Kiểm tra tự động bao gồm metric, các tùy chọn dự đoán, cú pháp Python,
cấu trúc notebook và tính nhất quán của minh chứng. Dự đoán từ mô hình đã lưu
được kiểm tra lại trên CPU; không huấn luyện lại khi rà soát tài liệu.

## 9. Tài liệu nộp bài và demo

- [report.md](report.md): báo cáo phương pháp, môi trường thực tế, số liệu, biểu đồ và nhận xét.
- [demo.md](demo.md): các bước cài đặt, chạy demo, xem kết quả và checklist trước khi nộp.
- [Practice3.ipynb](Practice3.ipynb): notebook chạy hai bài, hiển thị kết quả và ghi nhận xét; chọn chế độ `report` hoặc `demo` ở cell cấu hình.

Notebook mặc định dùng cấu hình bài thực hành 2.000/500/500 mẫu và 2 epoch.
Chế độ thử nhanh dùng `outputs/demo/`; chế độ báo cáo dùng `outputs/finetune/`.
Lưu notebook sau khi chạy để giữ đầu ra. Các minh chứng nhỏ đã có trong `results/` để nộp cùng repo.
Thư mục `outputs/` chứa mô hình và checkpoint lớn vẫn được Git bỏ qua.

## Tài liệu

- https://huggingface.co/docs/transformers/tasks/sequence_classification
- https://huggingface.co/docs/transformers/main_classes/trainer
- https://huggingface.co/datasets/stanfordnlp/imdb
- https://huggingface.co/distilbert/distilbert-base-uncased
- https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
