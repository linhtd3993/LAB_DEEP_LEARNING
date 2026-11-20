# Minh chứng lần chạy Lab 3

Các file nhỏ được sao chép từ `outputs/exercise1.json` và `outputs/finetune/` sau lần chạy 2.000 train / 500 validation / 500 test, 2 epoch, seed 42.

- `environment.json`: phiên bản và thiết bị trong output notebook đã lưu.
- `exercise1.json`: token/ID, nhãn và score của bài 1.
- `config.json`, `metrics.json`, `classification_report.txt`: cấu hình và kết quả test.
- `training_history.json`: lịch sử huấn luyện và validation.
- `eda_summary.json`: thống kê EDA tập gốc IMDb và 3 tập con (`2,000 / 500 / 500`), độ dài WordPiece token, tỷ lệ `<br />` và log-odds từ khóa.
- `error_analysis.json`: tổng hợp đánh giá Đúng/Sai trên 500 mẫu test (theo ngưỡng cắt 128 token, độ tự tin, 4 nhóm `TP/TN/FP/FN` và các mẫu tiêu biểu).
- `confusion_matrix.png`, `loss_curve.png`: biểu đồ đánh giá huấn luyện và ma trận nhầm lẫn.
- `eda_class_and_length.png`, `eda_text_insights.png`: biểu đồ trực quan hóa EDA dữ liệu IMDb.
- `error_analysis_overview.png`, `correct_vs_wrong_cases.png`: biểu đồ tổng quan Error Analysis và thẻ so sánh các ca Đúng/Sai tiêu biểu.
- `predictions.json`: kết quả kiểm tra lại mô hình đã lưu trên hai câu mới, không huấn luyện lại.

Thư mục này giữ được minh chứng khi đưa bài lên Git. Mô hình, tokenizer và checkpoint lớn vẫn nằm trong `outputs/` được Git bỏ qua. Các file ở đây là ảnh chụp của lần chạy được báo cáo, không tự thay đổi khi chạy lại notebook. Khi báo cáo một lần chạy mới, cần cập nhật đồng bộ các file và `report.md`.
