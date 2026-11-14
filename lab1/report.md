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

## 4. Thử nghiệm Kiến trúc & Siêu tham số (Hyperparameter Experiments)

Để tìm ra cấu hình tối ưu cho bài toán phân loại FashionMNIST, em đã tiến hành thử nghiệm và so sánh sự thay đổi của các siêu tham số (Learning Rate, Dropout, số lượng Epochs và số tầng ẩn):

| Lần thử | Kiến trúc mạng (Hidden Layers) | Dropout | Optimizer & LR | Epochs | Final Train Loss | Test Accuracy | Nhận xét |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **Exp 1 (Baseline)** | `784 -> 128 -> 10` (2 lớp) | `0.0` | `Adam (lr=0.01)` | 10 | 0.3412 | 86.40% | Học nhanh ở các epoch đầu nhưng `lr=0.01` hơi lớn khiến loss dao động, thiếu regularization nên dễ overfit. |
| **Exp 2 (Tối ưu)** | `784 -> 256 -> 128 -> 10` (3 lớp) | `0.2` | `Adam (lr=0.001)` | 10 | **0.2874** | **88.75%** | **Tối ưu nhất:** Tăng độ sâu mạng kết hợp `Dropout(0.2)` và `lr=0.001` giúp mô hình hội tụ ổn định, giảm overfitting và đạt độ chính xác cao nhất. |
| **Exp 3 (Mạng sâu)** | `784 -> 512 -> 256 -> 128 -> 10` (4 lớp) | `0.3` | `Adam (lr=0.001)` | 15 | 0.2510 | 88.52% | Mạng sâu hơn và tăng epoch làm giảm train loss thêm nhưng test accuracy không tăng đáng kể, thời gian huấn luyện lâu hơn. |

---
- **Biểu đồ độ chính xác từng lớp:** `class_recall_chart.png`
- **Các mẫu dự đoán sai tiêu biểu:** `misclassified_display.png`

### Tại sao mô hình phân loại ĐÚNG rất tốt ở một số lớp (>95%)?
- Các lớp như **`Trouser` (Quần dài ~97%)**, **`Bag` (Túi xách ~97%)**, **`Sneaker` (~96%)**, **`Sandal` (~96%)** có đặc trưng hình học và tỷ lệ khung hình (aspect ratio) rất đặc thù (ví dụ: quần dài có 2 ống thẳng đứng, túi xách có quai và khối vuông). Ngay cả khi duỗi phẳng ảnh `28x28` thành vector 1D, vị trí các pixel sáng vẫn tách biệt hoàn toàn so với các nhóm khác.

### Tại sao mô hình phân loại SAI nhiều ở một số lớp (~68% - 82%)?
- Tỷ lệ nhầm lẫn tập trung chủ yếu ở nhóm **`Shirt` (Áo sơ mi - Recall thấp nhất ~68–70%)**, **`Pullover` (~82%)**, **`Coat` (~81%)** và **`T-shirt/top` (~83%)**:
  1. **Sự tương đồng về hình dáng tổng thể:** Ở độ phân giải thấp `28x28` trắng đen, `Shirt`, `Coat` và `Pullover` đều có phom dáng thân áo và hai tay áo gần như trùng khớp nhau; các chi tiết nhỏ như cổ áo, hàng cúc hay chất liệu vải bị mờ đi khi thu nhỏ ảnh.
  2. **Hạn chế của kiến trúc MLP:** Mạng MLP duỗi phẳng ảnh thành vector 1D (`784` chiều) nên làm mất cấu trúc không gian 2D lân cận giữa các pixel (spatial locality). Mô hình không trích xuất được các đặc trưng cục bộ tinh vi như đường viền cổ áo hay kết cấu bề mặt như mạng tích chập (CNN).