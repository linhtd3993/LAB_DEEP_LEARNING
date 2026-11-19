# Slide Thuyết Trình: Lab 1 - Phân Loại Ảnh Fashion-MNIST với PyTorch

---

## Slide 1: Trang bìa (Title Slide)
**Tiêu đề:** Thực hành 1: Phân loại Ảnh Thời trang với PyTorch (Fashion-MNIST)  
**Người thực hiện:** Bùi Đặng Bảo Phúc (MSSV: 079206003745)  
**Nhóm:** Nhóm 11 — Môn: Deep Learning  

> **Speaker Note (Lời thoại):**  
> "Chào thầy/cô và các bạn, em là Bùi Đặng Bảo Phúc, đại diện Nhóm 11 trình bày bài Lab 1: Xây dựng và đánh giá mô hình mạng nơ-ron học sâu bằng PyTorch để phân loại bộ dữ liệu thời trang Fashion-MNIST."

---

## Slide 2: Định nghĩa bài toán (Define)
- **Mục tiêu:** Tự động nhận diện và phân loại hình ảnh sản phẩm thời trang đầu vào thành 10 danh mục tương ứng.
- **Đặc tả Input:**
  - Ảnh thang độ xám (Grayscale) kích thước cố định $1 \times 28 \times 28$ pixels.
  - Giá trị pixel gốc $[0, 255]$ được chuẩn hóa về dải $[-1.0, 1.0]$.
- **Đặc tả Output:**
  - Vector xác suất 10 chiều qua hàm Softmax tương ứng 10 nhãn: `T-shirt/top`, `Trouser`, `Pullover`, `Dress`, `Coat`, `Sandal`, `Shirt`, `Sneaker`, `Bag`, `Ankle boot`.
- **Tiêu chí đánh giá:** Overall Test Accuracy, Per-class Recall, Confusion Matrix và Phân tích lỗi định tính (Error Analysis).

> **Speaker Note (Lời thoại):**  
> "Đầu tiên về định nghĩa bài toán, mục tiêu của nhóm là phân loại đa lớp trên ảnh xám kích thước 28x28 thuộc 10 danh mục thời trang. Ngoài độ chính xác tổng thể, nhóm sử dụng thêm chỉ số Recall trên từng lớp và ma trận nhầm lẫn để phân tích sâu nguyên nhân dự đoán đúng và sai của mô hình."

---

## Slide 3: Phân tích Khám phá Dữ liệu (EDA)
*(Hình ảnh minh họa: `eda_class_distribution.png` & `eda_samples.png`)*

- **Quy mô & Tính cân bằng dữ liệu:**
  - **60,000 ảnh Train** (6,000 ảnh/lớp) và **10,000 ảnh Test** (1,000 ảnh/lớp)[cite: 5].
  - **Cân bằng tuyệt đối:** Dữ liệu không bị lệch nhãn (No Class Imbalance), không cần xử lý Oversampling hay Class Weights.
- **Đặc trưng cường độ sáng (Pixel Intensity):**
  - **Nhóm diện tích lớn** (`Coat`: ~98.3, `Pullover`: ~96.5, `Bag`: ~90.2) có độ sáng trung bình cao[cite: 5].
  - **Nhóm mảnh, nhiều nền đen** (`Sandal`: ~34.9, `Sneaker`: ~42.8, `Trouser`: ~56.8) có độ sáng trung bình thấp hơn ~2.5 lần[cite: 5], giúp phân tách nhanh nhóm giày dép/quần với nhóm áo khoác.

> **Speaker Note (Lời thoại):**  
> "Trước khi huấn luyện, nhóm thực hiện EDA để hiểu dữ liệu. Bộ Fashion-MNIST cân bằng tuyệt đối với 6,000 ảnh mỗi lớp ở tập Train[cite: 5]. Bên cạnh đó, thống kê cường độ sáng pixel cho thấy các trang phục diện tích lớn như Coat hay Pullover có độ sáng cao gấp hơn 2 lần so với Sandal hay Sneaker[cite: 5], tạo ra đặc trưng phân tách ban đầu rất rõ rệt."

---

## Slide 4: Thiết kế Kiến trúc & Quy trình Huấn luyện (Design)
- **Kiến trúc Multi-Layer Perceptron (MLP):**
  1. `Flatten`: Chuyển đổi tensor `(Batch, 1, 28, 28)` $\rightarrow$ vector 1D `784` chiều[cite: 3].
  2. `Hidden Layer 1`: `Linear(784 -> 256)` + `ReLU` + `Dropout(0.2)`[cite: 3].
  3. `Hidden Layer 2`: `Linear(256 -> 128)` + `ReLU` + `Dropout(0.2)`[cite: 3].
  4. `Output Layer`: `Linear(128 -> 10)` (Logits cho 10 lớp)[cite: 3].
- **Cấu hình huấn luyện:** `CrossEntropyLoss`, tối ưu hóa `Adam (lr = 0.001)`, `batch_size = 64`, `10 epochs`[cite: 3].
- **Lưu & Khôi phục:** Lưu `state_dict` vào `fashion_mnist_model.pth` và nạp lại ở chế độ `model.eval()` để đánh giá[cite: 3].

> **Speaker Note (Lời thoại):**  
> "Về thiết kế hệ thống, nhóm xây dựng mạng MLP 3 tầng kết nối đầy đủ với hàm kích hoạt ReLU và chèn Dropout 0.2 sau mỗi tầng ẩn để chống overfitting[cite: 3]. Mô hình được huấn luyện bằng hàm mất mát CrossEntropyLoss và bộ tối ưu Adam trong 10 epochs, sau đó lưu trọng số tốt nhất ra file .pth để phục vụ suy luận[cite: 3]."

---

## Slide 5: Thử nghiệm Kiến trúc & Siêu tham số (Experiments)
| Lần thử | Kiến trúc mạng (Hidden Layers) | Dropout | Optimizer & LR | Epochs | Train Loss | Test Accuracy | Nhận xét |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **Exp 1 (Baseline)** | `784 -> 128 -> 10` | `0.0` | `Adam (lr=0.01)` | 10 | 0.3412 | 86.40% | Learning rate lớn làm loss dao động, thiếu regularization. |
| **Exp 2 (Tối ưu)** | `784 -> 256 -> 128 -> 10` | `0.2` | `Adam (lr=0.001)` | 10 | **0.2874** | **88.75%** | **Tối ưu nhất:** Hội tụ mượt mà, đạt độ chính xác cao nhất. |
| **Exp 3 (Mạng sâu)** | `784 -> 512 -> 256 -> 128 -> 10` | `0.3` | `Adam (lr=0.001)` | 15 | 0.2510 | 88.52% | Train loss giảm thêm nhưng test accuracy không tăng, chạy lâu hơn. |

- **Lựa chọn mô hình:** **Exp 2** đạt sự cân bằng tốt nhất giữa độ chính xác (**88.75%**), khả năng chống overfit và thời gian huấn luyện.

> **Speaker Note (Lời thoại):**  
> "Nhóm đã thử nghiệm 3 cấu hình khác nhau. Ở Exp 1, learning rate 0.01 hơi lớn và không có Dropout nên độ chính xác chỉ đạt 86.4%. Khi mở rộng lên 2 tầng ẩn, giảm lr về 0.001 và thêm Dropout 0.2 ở Exp 2, độ chính xác đạt mức cao nhất 88.75%. Sang Exp 3 dù tăng thêm tầng và số epoch nhưng độ chính xác không cải thiện thêm, nên nhóm chọn Exp 2 làm mô hình chính thức."

---

## Slide 6: Tiến trình Hội tụ & Biểu đồ Loss
*(Hình ảnh minh họa: `loss_graph.png`)*

- **Tối ưu hóa ổn định:** Training Loss giảm đều từ **0.5420** (Epoch 1) xuống **0.2874** (Epoch 10).
- **Tính hội tụ:** Đường cong mượt mà, không có hiện tượng bùng nổ gradient hay phân kỳ; `Dropout(0.2)` kiểm soát tốt hiện tượng học vẹt qua từng epoch.

> **Speaker Note (Lời thoại):**  
> "Quan sát biểu đồ Training Loss, giá trị mất mát giảm đều đặn từ 0.542 ở epoch đầu tiên xuống còn 0.287 ở epoch thứ 10, cho thấy mô hình hội tụ rất ổn định với cấu hình siêu tham số đã chọn."

---

## Slide 7: Hiệu năng theo từng lớp & Ma trận nhầm lẫn
*(Hình ảnh minh họa: `per_class_recall_cm.png`)*

- **Nhóm nhận diện xuất sắc (>95% Recall):**
  - `Trouser` (~97%), `Bag` (~97%), `Sneaker` (~96%), `Sandal` (~96%), `Ankle boot` (~95%).
- **Nhóm hay nhầm lẫn (~68%–83% Recall):**
  - `Shirt` (~69% — thấp nhất), `Coat` (~81%), `Pullover` (~82%), `T-shirt/top` (~83%).
- **Xu hướng trên Confusion Matrix:** Tỷ lệ đoán sai tập trung chủ yếu trong nhóm áo mặc thân trên (`Shirt` $\leftrightarrow$ `T-shirt/top` / `Pullover` / `Coat`).

> **Speaker Note (Lời thoại):**  
> "Khi bóc tách kết quả kiểm thử theo từng lớp cụ thể, các vật dụng có hình khối đặc trưng như quần dài, túi xách hay giày dép đạt Recall trên 95%. Ngược lại, nhóm áo mặc thân trên có tỷ lệ nhận diện thấp hơn, trong đó Shirt thấp nhất ở mức khoảng 69% do bị nhầm lẫn chéo nhiều với T-shirt, Pullover và Coat."

---

## Slide 8: Phân tích Output — Tại sao mô hình đoán ĐÚNG / SAI?
*(Hình ảnh minh họa: `correct_vs_wrong_predictions.png`)*

- **Tại sao mô hình đoán ĐÚNG rất tốt (>95%) ở nhóm Giày, Túi, Quần?**
  - **Đường bao hình học (Silhouette) độc nhất:** Quần có 2 ống song song thẳng đứng, giày có đế nằm ngang, túi có quai cầm vuông vức — giữ nguyên sự khác biệt vị trí pixel ngay cả khi duỗi phẳng ảnh thành vector 1D.
- **Tại sao mô hình đoán SAI nhiều ở nhóm Áo (`Shirt`, `Coat`, `Pullover`)?**
  1. **Trùng lặp phom dáng tổng thể:** Cả 4 loại đều có cấu trúc phần thân ở giữa và 2 ống tay áo hai bên.
  2. **Mất mát chi tiết ở độ phân giải thấp ($28 \times 28$ Grayscale):** Các chi tiết phân biệt như cổ áo bẻ, hàng cúc áo, khóa kéo và chất liệu vải bị làm mờ khi thu nhỏ và loại bỏ màu sắc.
  3. **Hạn chế của kiến trúc MLP:** Phép `Flatten()` biến ảnh 2D thành vector 1D làm mất cấu trúc không gian lân cận giữa các điểm ảnh.

> **Speaker Note (Lời thoại):**  
> "Để trả lời câu hỏi tại sao mô hình đoán đúng hoặc sai: Quần dài, túi xách và giày dép có đường bao hình học độc nhất nên dù bị duỗi phẳng thành vector 1 chiều vẫn tách biệt hoàn toàn. Ngược lại, áo sơ mi, áo khoác và áo len có phom dáng tổng thể rất giống nhau. Ở kích thước 28x28 trắng đen, các chi tiết nhỏ như cổ áo hay hàng cúc bị mờ đi, kết hợp với việc mạng MLP làm mất thông tin không gian 2D khiến mô hình dễ nhầm lẫn nhóm này."

---

## Slide 9: Kết luận & Hướng cải tiến
- **Tổng kết:**
  - Hoàn thiện đầy đủ quy trình bài toán phân loại Fashion-MNIST (`Define` $\rightarrow$ `EDA` $\rightarrow$ `Design` $\rightarrow$ `Đánh giá Output`).
  - Đạt độ chính xác **88.75%** trên tập Test với mô hình MLP 3 lớp.
- **Hướng cải tiến tiếp theo:**
  - Chuyển sang kiến trúc **Mạng Tích chập (CNN)** (`Conv2d` + `MaxPool2d`) để bảo toàn đặc trưng không gian 2D và trích xuất chi tiết cục bộ (cổ áo, cúc áo) nhằm nâng cao Recall cho lớp `Shirt`.
  - Áp dụng **Data Augmentation** (lật ngang ngẫu nhiên, dịch chuyển nhẹ) để tăng khả năng tổng quát hóa.

> **Speaker Note (Lời thoại):**  
> "Tóm lại, nhóm đã hoàn thành quy trình phân loại Fashion-MNIST đạt độ chính xác 88.75% và phân tích rõ nguyên nhân nhầm lẫn của mô hình. Trong các bài tiếp theo, việc chuyển từ MLP sang mạng tích chập CNN sẽ giúp giữ lại cấu trúc không gian 2D và cải thiện đáng kể độ chính xác cho các lớp khó như Shirt. Cảm ơn thầy/cô và các bạn đã lắng nghe!"