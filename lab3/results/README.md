# Minh chứng lần chạy Lab 3

Các file nhỏ được sao chép từ `outputs/exercise1.json` và `outputs/finetune/` sau lần chạy 2.000 train / 500 validation / 500 test, 2 epoch, seed 42.

- `environment.json`: phiên bản và thiết bị trong output notebook đã lưu.
- `exercise1.json`: token/ID, nhãn và score của bài 1.
- `config.json`, `metrics.json`, `classification_report.txt`: cấu hình và kết quả test.
- `training_history.json`: lịch sử huấn luyện và validation.
- `confusion_matrix.png`, `loss_curve.png`: biểu đồ của cùng lần chạy.
- `predictions.json`: kết quả kiểm tra lại mô hình đã lưu trên hai câu mới, không huấn luyện lại.

Thư mục này giữ được minh chứng khi đưa bài lên Git. Mô hình, tokenizer và checkpoint lớn vẫn nằm trong `outputs/` được Git bỏ qua. Các file ở đây là ảnh chụp của lần chạy được báo cáo, không tự thay đổi khi chạy lại notebook. Khi báo cáo một lần chạy mới, cần cập nhật đồng bộ các file và `report.md`.
