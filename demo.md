# Hướng dẫn chạy Demo - Lab 1: PyTorch FashionMNIST Classification

Tài liệu hướng dẫn cài đặt môi trường, chạy huấn luyện và kiểm thử mô hình phân loại FashionMNIST bằng kiến trúc **Multi-Layer Perceptron (MLP)**.

---

## 1. Danh sách tệp tin Lab 1
- `lab1_fashion_mnist.py`: Mã nguồn chính định nghĩa mạng `FashionClassifier` (MLP), huấn luyện và xuất biểu đồ[cite: 17].
- `lab1_fashion_mnist.ipynb`: Jupyter Notebook chạy từng cell kèm kết quả trực quan.
- `fashion_mnist_model.pth`: Tệp trọng số mô hình đã lưu (đạt Test Accuracy **88.11%**)[cite: 17].
- `loss_graph.png`: Biểu đồ Training Loss qua 10 Epochs[cite: 17].
- `predictions_display.png`: Hình ảnh hiển thị kết quả dự đoán ngẫu nhiên 10 mẫu trên tập Test[cite: 17].
- `report.md`: Báo cáo chi tiết kiến trúc mạng và kết quả đánh giá[cite: 12].

---

## 2. Yêu cầu thư viện
Cài đặt các gói phụ thuộc cần thiết:
```bash
pip install torch torchvision matplotlib numpy