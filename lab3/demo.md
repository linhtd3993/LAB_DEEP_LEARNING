# Demo Lab 3 — Hugging Face

Tài liệu này hướng dẫn trình diễn bài thực hành bằng terminal hoặc [Practice3.ipynb](Practice3.ipynb). Nội dung báo cáo ở [report.md](report.md). Các lệnh dưới đây chạy trong PowerShell tại thư mục `lab3/`. Nếu terminal đang ở thư mục gốc repo, chạy `cd lab3` trước.

## 1. Chuẩn bị môi trường

Lần chạy được báo cáo dùng Python 3.14.0 trên CPU và các phiên bản trong `requirements-run.txt`.
Để tạo môi trường tương ứng:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-run.txt
.\.venv\Scripts\python.exe -c "import sys, torch; print(sys.version); print('CUDA:', torch.cuda.is_available())"
```

Nếu đã có môi trường chạy thành công, dùng đúng interpreter/kernel đó và bỏ qua bước tạo/cài lại. Trong môi trường mới ở trên, chọn interpreter `.venv\Scripts\python.exe` trong VS Code và cùng môi trường đó làm kernel. `requirements.txt` là cấu hình Python 3.11 ban đầu; nếu chọn cấu hình đó, dùng `py -3.11` và cài file tương ứng, không trộn hai bộ phiên bản.

Nếu chỉ trình bày kết quả, mở notebook đã lưu output và `results/`; không cần huấn luyện lại.

Cần Internet để tải mô hình và IMDb lần đầu. Tải và huấn luyện thử trước buổi demo; thời gian phụ thuộc máy và mạng. Chuẩn bị sẵn mô hình đã huấn luyện nếu thời gian trình bày ngắn.

## 2. Demo bài 1: tokenizer và suy luận

```powershell
.\.venv\Scripts\python.exe exercise1_sentiment.py
```

Lệnh mặc định chạy **5 câu tiếng Anh**, cùng danh sách trong notebook. Trình bày các trường `tokens`, `input_ids`, `label`, `score` trên terminal và mở `outputs/exercise1.json`. Giải thích rằng tokenizer chuyển văn bản thành đầu vào mô hình, còn score là xác suất của nhãn dự đoán. Đối chiếu nhãn thực tế với ý nghĩa từng câu; không trình bày nhãn kỳ vọng như kết quả đã chạy.

## 3. Demo bài 2: kiểm tra luồng với ít dữ liệu

```powershell
.\.venv\Scripts\python.exe exercise2_finetune.py --train-size 64 --val-size 32 --test-size 32 --epochs 1 --output-dir outputs/demo
```

Giải thích: train dùng cập nhật trọng số; validation dùng chọn checkpoint; test dùng đánh giá cuối. Mô hình nền là DistilBERT với đầu phân loại hai nhãn.

Khi lệnh hoàn thành, xem kết quả:

```powershell
Get-Content outputs/demo/config.json
Get-Content outputs/demo/metrics.json
Get-Content outputs/demo/classification_report.txt
```

Mở `outputs/demo/confusion_matrix.png` và `outputs/demo/loss_curve.png` trong VS Code. Với một epoch, biểu đồ chỉ có ít điểm; không đủ cơ sở nhận xét xu hướng huấn luyện. Kết quả từ tập nhỏ chỉ xác nhận luồng có thể chạy, không đại diện chất lượng mô hình.

## 4. Demo dự đoán câu mới

Dùng đúng mô hình của lần thử nhanh:

```powershell
.\.venv\Scripts\python.exe predict.py --model outputs/demo/model --output outputs/demo/predictions.json --text "The acting was excellent!" "I regret watching this movie."
```

Đọc nhãn và score thực tế. Nếu kết quả sai, giải thích hạn chế của lần huấn luyện rất ít dữ liệu. Script yêu cầu mô hình đã được lưu thành công ở bước trước.

## 5. Chạy cấu hình dùng cho báo cáo

Thực hiện trước khi hoàn thiện báo cáo; không cần huấn luyện lại trong lúc thuyết trình:

```powershell
.\.venv\Scripts\python.exe exercise2_finetune.py --train-size 2000 --val-size 500 --test-size 500 --epochs 2 --output-dir outputs/finetune
.\.venv\Scripts\python.exe predict.py --model outputs/finetune/model --output outputs/finetune/predictions.json --text "The acting was excellent!" "I regret watching this movie."
```

Lần chạy hiện tại đã có **82,40% accuracy**, **83,46% F1** và đã được điền vào `report.md`; minh chứng ở `results/`. Nếu chạy lại, lấy số liệu từ `outputs/finetune/metrics.json` và cập nhật báo cáo cùng các minh chứng của **cùng lần chạy**. `outputs/demo` và `outputs/finetune` tách riêng để lần thử nhanh không ghi đè kết quả báo cáo. Nếu chạy lại cùng output directory, các file kết quả có thể bị ghi đè.

## 6. Trình diễn bằng notebook

1. Mở `Practice3.ipynb`, chọn kernel đã cài đủ thư viện (lần chạy đã lưu dùng Python 3.14.0 trên CPU).
2. Chạy cell kiểm tra môi trường.
3. Chạy phần bài 1 và quan sát bảng dự đoán, tokens và ID.
4. Ở cell cấu hình bài 2, giữ `MODE = "report"` cho bài nộp hoặc đổi sang `"demo"` để thử nhanh.
5. Chạy cell huấn luyện, xem cấu hình, chỉ số, classification report và biểu đồ.
6. Chạy dự đoán mới, điền phần nhận xét và lưu notebook bằng Ctrl+S để giữ output thực tế.

Nếu đã có kết quả huấn luyện, có thể bỏ qua cell huấn luyện và chạy cell đọc kết quả sau khi chạy các cell import/cấu hình. Không đổi chế độ giữa lúc huấn luyện và đọc kết quả.

## 7. Checklist trước khi nộp

- [x] Đã chạy bài 1 và lưu đầu ra thực tế.
- [x] Đã huấn luyện cấu hình bài thực hành, ghi lại môi trường và cấu hình thực tế.
- [x] Điền các chỉ số và nhận xét trong `report.md` từ kết quả thật.
- [x] Notebook đã có output; hai biểu đồ và báo cáo được đính kèm trong `results/`.
- [ ] Kiểm tra file trước khi commit; giữ mô hình/checkpoint lớn trong `outputs/` được Git bỏ qua.

## 8. Xử lý lỗi trong lúc demo

- Thiếu thư viện: kiểm tra kernel/interpreter rồi cài bộ requirements tương ứng bằng Python của môi trường đang dùng.
- Không tìm thấy mô hình dự đoán: hoàn thành huấn luyện và kiểm tra `--model` trỏ đúng thư mục.
- Thiếu bộ nhớ GPU: thêm `--batch-size 2 --max-length 64` hoặc `--cpu` vào lệnh huấn luyện, đồng thời ghi lại cấu hình thay đổi trong báo cáo.
- Không tải được mô hình/dataset: kiểm tra kết nối tới Hugging Face và chạy lại sau khi mạng ổn định.
