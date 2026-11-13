# Lab 1: PyTorch FashionMNIST Classification

**Họ và tên:** Bùi Đặng Bảo Phúc  
**Mã số sinh viên:** 079206003745  
**Lớp / Môn học:** Deep Learning  
**Kết quả đạt được (Test Accuracy):** **88.75%**

---

## 1. Nội dung công việc (Tasks Completed)

- [x] **Load & Preprocess Dataset:** Tải tập dữ liệu FashionMNIST và chuẩn hóa hình ảnh về khoảng `[-1, 1]`.
- [x] **Model Architecture:** Thiết kế mạng nơ-ron Multi-Layer Perceptron (MLP) gồm 3 lớp Fully-Connected tích hợp `ReLU` và `Dropout(0.2)` để hạn chế Overfitting.
- [x] **Training Pipeline:** Xây dựng vòng lặp huấn luyện với hàm mất mát `CrossEntropyLoss` và thuật toán tối ưu `Adam` (lr=0.001) trong 10 Epochs.
- [x] **Evaluation & Saving:** Đánh giá mô hình trên tập Test, đạt độ chính xác **88.75%**. Lưu và tải lại trọng số mô hình `fashion_mnist_model.pth`.
- [x] **Visualization:** Trực quan hóa biểu đồ Training Loss qua các Epochs và hiển thị kết quả phân loại ngẫu nhiên 10 hình ảnh.

---

## 2. Kiến trúc Mô hình (Model Architecture)

```
FashionClassifier(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc_layers): Sequential(
    (0): Linear(in_features=784, out_features=256, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.2, inplace=False)
    (3): Linear(in_features=256, out_features=128, bias=True)
    (4): ReLU()
    (5): Dropout(p=0.2, inplace=False)
    (6): Linear(in_features=128, out_features=10, bias=True)
  )
)
```

---

## 3. Trực quan hóa & Kết quả (Results & Visualizations)

- **Training Loss Over Epochs:** Loss giảm đều từ **0.5420** (Epoch 1) xuống **0.2874** (Epoch 10).
- **Test Accuracy:** **88.75%**
- Các file hình ảnh kết quả được tạo tự động:
  - `loss_graph.png`: Đồ thị theo dõi tiến trình giảm Loss.
  - `predictions_display.png`: So sánh nhãn dự đoán (`Pred`) và nhãn thực tế (`Actual`).
  - `fashion_mnist_model.pth`: File lưu trọng số mô hình.

---

## 4. Hướng dẫn chạy Code (How to Run)

1. **Cài đặt thư viện cần thiết:**
   ```bash
   py -m pip install torch torchvision matplotlib numpy
   ```

2. **Chạy bài làm:**
   ```bash
   py lab1_fashion_mnist.py
   ```
