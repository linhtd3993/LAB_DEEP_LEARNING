# Slide Presentation: Lab 1 - PyTorch FashionMNIST Classification

---

## Slide 1: Trang bìa (Title Slide)
**Tiêu đề:** Thực hành 1: Phân loại Ảnh Thời trang với PyTorch (Fashion-MNIST)  
**Người thực hiện:** Bùi Đặng Bảo Phúc (MSSV: 079206003745)  
**Nhóm:** Nhóm 11 — Môn: Deep Learning  

> **Speaker Note (Lời thoại):**  
> "Chào thầy/cô và các bạn, em là Bùi Đặng Bảo Phúc, đại diện Nhóm 11 trình bày bài Lab 1: Xây dựng mô hình mạng nơ-ron học sâu bằng PyTorch để phân loại bộ dữ liệu Fashion-MNIST."

---

## Slide 2: Định nghĩa bài toán & Khám phá dữ liệu (Define & EDA)
* **Mục tiêu:** Phân loại tự động 10 danh mục sản phẩm thời trang từ ảnh đa cấp xám.
* **Quy mô dữ liệu:** 60,000 ảnh Train & 10,000 ảnh Test (`1 x 28 x 28` pixels).
* **Phân bố lớp:** Cân bằng tuyệt đối — mỗi lớp có đúng 6,000 ảnh Train và 1,000 ảnh Test.
* **Tiền xử lý (Transforms):** Chuyển sang Tensor và chuẩn hóa pixel về dải `[-1.0, 1.0]` (`Normalize((0.5,), (0.5,))`).

> **Speaker Note (Lời thoại):**  
> "Trước khi xây dựng mô hình, nhóm tiến hành EDA để hiểu dữ liệu. Bộ Fashion-MNIST gồm 70,000 ảnh xám kích thước 28x28 chia làm 10 lớp cân bằng nhau, không bị lệch nhãn. Dữ liệu được chuẩn hóa về khoảng [-1, 1] để quá trình tính đạo hàm hội tụ ổn định và nhanh hơn."

---

## Slide 3: Thiết kế Kiến trúc & Quy trình Huấn luyện (Design)
* **Kiến trúc Multi-Layer Perceptron (MLP):**
  * `Flatten`: `28 x 28` $\rightarrow$ `784` đặc trưng đầu vào.
  * `Hidden 1`: `Linear(784, 256)` + `ReLU` + `Dropout(0.2)`.
  * `Hidden 2`: `Linear(256, 128)` + `ReLU` + `Dropout(0.2)`.
  * `Output`: `Linear(128, 10)` (Logits cho 10 lớp).
* **Cấu hình huấn luyện:** `CrossEntropyLoss`, tối ưu hóa `Adam (lr=0.001)`, `batch_size=64`, `10 epochs`.
* **Lưu & Khôi phục:** Lưu `state_dict` (`fashion_mnist_model.pth`) và nạp lại ở chế độ `eval()` để suy luận.

> **Speaker Note (Lời thoại):**  
> "Về thiết kế mô hình, nhóm sử dụng kiến trúc MLP 3 lớp kết nối đầy đủ với hàm kích hoạt ReLU và chèn thêm Dropout 0.2 sau các tầng ẩn để chống overfitting. Sau khi huấn luyện xong, trọng số được lưu lại thành file .pth và nạp lại để kiểm thử thực tế."

---

## Slide 4: Thử nghiệm Kiến trúc & Siêu tham số (Experiments)
| Lần thử | Kiến trúc mạng | Dropout | Optimizer & LR | Epochs | Train Loss | Test Accuracy |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| **Exp 1 (Baseline)** | `784 -> 128 -> 10` | `0.0` | `Adam (lr=0.01)` | 10 | 0.3412 | 86.40% |
| **Exp 2 (Tối ưu)** | `784 -> 256 -> 128 -> 10` | `0.2` | `Adam (lr=0.001)` | 10 | **0.2874** | **88.75%** |
| **Exp 3 (Mạng sâu)** | `784 -> 512 -> 256 -> 128 -> 10` | `0.3` | `Adam (lr=0.001)` | 15 | 0.2510 | 88.52% |

* **Kết luận chọn mô hình:** **Exp 2** cân bằng tốt nhất giữa độ chính xác (**88.75%**), khả năng chống overfit và thời gian huấn luyện.

> **Speaker Note (Lời thoại):**  
> "Nhóm đã thử nghiệm 3 cấu hình khác nhau. Ở Exp 1, learning rate 0.01 hơi lớn và thiếu Dropout nên độ chính xác chỉ đạt 86.4%. Khi tăng độ sâu lên 3 lớp, giảm lr về 0.001 và thêm Dropout 0.2 ở Exp 2, độ chính xác tăng lên mức cao nhất 88.75%. Sang Exp 3 dù tăng thêm tầng và số epoch nhưng độ chính xác không cải thiện thêm mà lại tốn thời gian chạy hơn, nên nhóm chọn Exp 2 làm mô hình chính thức."

---

## Slide 5: Kết quả Huấn luyện & Độ chính xác từng lớp
* **Tiến trình Loss (`loss_graph.png`):** Giảm đều từ `0.5420` (Epoch 1) xuống `0.2874` (Epoch 10), không có dấu hiệu phân kỳ.
* **Hiệu năng theo từng lớp (`class_recall_chart.png`):**
  * **Nhóm nhận diện xuất sắc (>95%):** `Trouser` (~97%), `Bag` (~97%), `Sneaker` (~96%), `Sandal` (~96%).
  * **Nhóm hay nhầm lẫn (~68%–83%):** `Shirt` (~69%), `Coat` (~81%), `Pullover` (~82%), `T-shirt/top` (~83%).

> **Speaker Note (Lời thoại):**  
> "Quan sát biểu đồ Loss, đường cong giảm mượt mà qua 10 epochs. Khi bóc tách độ chính xác trên từng lớp cụ thể, các vật dụng có biên dạng đặc thù như quần dài, túi xách hay giày dép đạt Recall trên 95%. Ngược lại, nhóm áo mặc thân trên như Shirt, Coat và Pullover có tỷ lệ nhận diện thấp hơn."

---

## Slide 6: Phân tích Output Đúng / Sai — Tại sao mô hình nhầm lẫn?
*(Hiển thị kèm 2 ảnh: `predictions_display.png` và `misclassified_display.png`)*

* **Tại sao dự đoán ĐÚNG tốt ở nhóm Giày/Túi/Quần?**
  * Hình dáng hình học và tỷ lệ khung hình tách biệt rõ ràng (ví dụ: quần có 2 ống thẳng đứng, túi có quai vuông), giữ nguyên sự khác biệt ngay cả khi duỗi phẳng ảnh.
* **Tại sao dự đoán SAI nhiều ở nhóm Áo (`Shirt`, `Coat`, `Pullover`)?**
  1. **Đặc trưng thị giác trùng lặp:** Ở ảnh xám độ phân giải thấp `28x28`, phom dáng thân áo và tay áo của áo sơ mi, áo khoác và áo len gần như giống nhau; chi tiết cổ áo hay hàng cúc bị mờ.
  2. **Hạn chế của kiến trúc MLP:** Phép `Flatten` biến ảnh 2D thành vector 1D làm mất cấu trúc không gian lân cận giữa các pixel.
* **Hướng cải tiến:** Chuyển sang sử dụng **Mạng Tích chập (CNN)** để giữ lại đặc trưng không gian 2D và trích xuất đường viền cục bộ tốt hơn.

> **Speaker Note (Lời thoại):**  
> "Nhìn vào lưới ảnh các ca dự đoán sai, có thể thấy mô hình chủ yếu nhầm lẫn qua lại giữa Shirt, T-shirt, Pullover và Coat. Nguyên nhân là do ảnh gốc chỉ có kích thước 28x28 nên các chi tiết nhỏ như cổ áo bị mờ, khiến hình dáng tổng thể của các loại áo này rất giống nhau. Đồng thời, kiến trúc MLP duỗi phẳng ảnh thành vector 1 chiều nên làm mất thông tin không gian 2D. Để khắc phục trong các bài sau, hướng cải tiến tốt nhất là sử dụng mạng tích chập CNN."