# 📊 BÀI BÁO CÁO THUYẾT TRÌNH: LAB 1

## PyTorch FashionMNIST Classification

* **Họ và tên:** Bùi Đặng Bảo Phúc
* **Mã số sinh viên:** 079206003745
* **Môn học:** Deep Learning
* **Độ chính xác đạt được (Test Accuracy):** **88.11%**

---

## 🎯 1. MỤC TIÊU BÀI LAB (OBJECTIVES)

* **Đề bài:** Phân loại tập dữ liệu thời trang **FashionMNIST** gồm 10 lớp sản phẩm (Áo T-shirt, Quần, Giày, Túi xách,...).
* **Công nghệ sử dụng:** Framework PyTorch, Matplotlib, Torchvision.
* **Các yêu cầu chính:**
  1. Load & Tiền xử lý dữ liệu (Transforms, DataLoaders).
  2. Thiết kế Mạng Nơ-ron (Neural Network Architecture).
  3. Viết Vòng lặp huấn luyện (Training Loop: Forward, Loss, Backward, Optimize).
  4. Đánh giá độ chính xác (Evaluation) và Lưu/Tải mô hình (Model Saving/Loading).
  5. Trực quan hóa kết quả (Visualize Loss & Predictions).

---

## 🏗️ 2. KIẾN TRÚC MÔ HÌNH (MODEL ARCHITECTURE)

Mô hình sử dụng kiến trúc **Multi-Layer Perceptron (MLP)** gồm 3 lớp Fully Connected (Linear):

* **Input Layer:** Flatten hình ảnh từ $28 \times 28 \to 784$ vector 1D.
* **Hidden Layer 1:** Linear($784 \to 256$) + ReLU + Dropout(0.2).
* **Hidden Layer 2:** Linear($256 \to 128$) + ReLU + Dropout(0.2).
* **Output Layer:** Linear($128 \to 10$) đại diện cho 10 class thời trang.

> 💡 **Kỹ thuật tối ưu:** Sử dụng `Dropout(0.2)` giữa các lớp ẩn giúp giảm nguy cơ **Overfitting** (học vẹt) trên tập huấn luyện.

---

## 🔄 3. QUY TRÌNH HUẤN LUYỆN (TRAINING PIPELINE)

* **Hàm mất mát (Loss Function):** `nn.CrossEntropyLoss()` - Phù hợp cho bài toán phân loại đa lớp.
* **Thuật toán tối ưu (Optimizer):** `optim.Adam(lr=0.001)` - Giúp mô hình hội tụ nhanh và ổn định.
* **Kích thước Batch (Batch Size):** 64
* **Số Epochs:** 10

### 📈 Tiến trình giảm Loss qua các Epochs:

| Epoch | Training Loss |
| :--- | :--- |
| Epoch 1 | **0.5420** |
| Epoch 3 | **0.3734** |
| Epoch 5 | **0.3368** |
| Epoch 8 | **0.3028** |
| **Epoch 10** | **0.2874** |

> **Nhận xét:** Loss giảm liên tục và ổn định từ 0.5420 xuống 0.2874 qua 10 vòng lặp.

---

## 📊 4. KẾT QUẢ ĐẠT ĐƯỢC (RESULTS & EVALUATION)

* **Test Accuracy:** **88.11%** trên tập dữ liệu kiểm thử (10,000 ảnh test).
* **Lưu mô hình:** Trọng số mô hình đã được lưu thành công ra file `fashion_mnist_model.pth` và kiểm tra load lại hoạt động hoàn hảo.
* **File hình ảnh đầu ra:**
  1. `loss_graph.png`: Biểu diễn đồ thị tiến trình giảm Loss.
  2. `predictions_display.png`: Hiển thị 10 mẫu ảnh ngẫu nhiên kèm nhãn Dự đoán (`Pred`) vs Nhãn thực tế (`Actual`).

---

## ❓ 5. CÂU HỎI PHẢN BIỆN THƯỜNG GẶP TỪ THẦY CÔ (Q&A)

### Q1: Tại sao lại chọn hàm kích hoạt ReLU và lớp Dropout?
* **Trả lời:** `ReLU` giúp giải quyết vấn đề Triệt tiêu đạo hàm (Vanishing Gradient) và tính toán nhanh. `Dropout(0.2)` ngẫu nhiên tắt 20% số nơ-ron ở mỗi bước huấn luyện để tránh mô hình quá phụ thuộc vào một số nơ-ron nhất định, giảm overfitting.

### Q2: Tại sao phải chuẩn hóa dữ liệu với `Normalize((0.5,), (0.5,))`?
* **Trả lời:** Hàm đưa giá trị pixel từ khoảng $[0, 1]$ về khoảng $[-1, 1]$. Việc này giúp trung bình hóa dữ liệu quanh số 0, làm cho quá trình tính toán Gradient ổn định hơn và tăng tốc độ hội tụ của thuật toán tối ưu.

---

## 🚀 6. KẾT LUẬN

* Bài Lab 1 đã hoàn thành đầy đủ tất cả các yêu cầu đề ra.
* Mã nguồn được quản lý chặt chẽ theo từng nhánh Git (`lab1`) và sẵn sàng cho việc đóng góp vào hệ thống.
