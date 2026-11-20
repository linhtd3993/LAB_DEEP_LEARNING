# Topic: Tìm hiểu kiến trúc Vision Transformer (ViT) và ứng dụng trong Thị giác máy tính

**Mục tiêu:** Giải thích được cách áp dụng kiến trúc Transformer thuần túy (không dùng tích chập - CNN) vào xử lý ảnh, mô tả được quá trình chuyển đổi ảnh thành chuỗi các patch, cơ chế Self-Attention trên ảnh và so sánh sự khác biệt giữa ViT với các mô hình CNN truyền thống.

---

## Cấu trúc nội dung chi tiết

| Các mục | Nội dung sinh viên cần trình bày | Yêu cầu minh họa |
| :--- | :--- | :--- |
| **1. Đặt vấn đề** | Vì sao Vision Transformer ra đời? Hạn chế của mạng CNN truyền thống (tính cục bộ - inductive bias, khó nắm bắt phụ thuộc tầm xa - global context) và động lực mang Transformer từ NLP sang CV. | Ví dụ minh họa sự khác biệt giữa vùng tiếp nhận cục bộ (receptive field) của CNN và khả năng nhìn toàn cục (global attention) của ViT trên cùng một bức ảnh. |
| **2. Kiến trúc tổng quan** | Các thành phần chính và luồng dữ liệu trong Vision Transformer (chỉ sử dụng phần Encoder của Transformer gốc) và đầu ra phân loại (MLP Head). | Sơ đồ kiến trúc tổng quan của ViT có chú thích đầy đủ các khối (từ ảnh đầu vào đến nhãn dự đoán). |
| **3. Biểu diễn đầu vào (Patch & Position)** | Cách chia ảnh thành các mảnh nhỏ (Image Patching), làm phẳng (Flattening), chiếu tuyến tính (Linear Projection / Patch Embedding), vai trò của `[CLS]` token và Positional Embedding (1D vs 2D). | Minh họa một bức ảnh kích thước cụ thể (ví dụ $224 \times 224 \times 3$) được cắt thành lưới các patch (ví dụ $16 \times 16$) và biến đổi thành chuỗi vector. |
| **4. Self-Attention trên ảnh** | Vai trò của Query, Key, Value đối với các patch ảnh; cách tính attention score giữa các vùng ảnh khác nhau và ý nghĩa vật lý của nó. | Ví dụ tính toán hoặc minh họa ma trận tương tác (attention matrix) giữa 3–4 patch tiêu biểu trên một bức ảnh. |
| **5. Multi-Head Self-Attention (MSA)** | Cách tổ chức nhiều attention head trên dữ liệu ảnh, giúp mô hình học đồng thời nhiều đặc trưng không gian khác nhau (màu sắc, kết cấu, hình dáng), cách ghép và chiếu kết quả. | Sơ đồ thể hiện sự thay đổi kích thước tensor $(B, N, D)$ qua các bước chia head, tính attention và nối (concat) lại. |
| **6. Khối ViT Encoder** | Cấu trúc chi tiết bên trong một Transformer Encoder block dành cho ảnh: LayerNorm (Pre-Norm), Multi-Head Attention, MLP (Feed-Forward Network) và Residual Connection. | So sánh trực quan cấu trúc bên trong một ViT Encoder block với một khối Residual Block của CNN (như ResNet). |
| **7. Các biến thể và mở rộng của ViT** | Sự phát triển sau ViT gốc: Giải quyết bài toán thiếu dữ liệu (DeiT - distillation), kiến trúc phân cấp đa tỷ lệ (Swin Transformer - Shifted Windows) hoặc mô hình lai (Hybrid CNN-ViT). | Bảng so sánh ViT gốc, DeiT và Swin Transformer (hoặc so sánh ViT với ResNet) theo tiêu chí: kiến trúc, lượng dữ liệu cần thiết, độ phức tạp tính toán và tác vụ phù hợp. |
| **8. Demo và nhận xét** | Thực hiện chạy thử nghiệm phân loại ảnh hoặc trích xuất đặc trưng bằng Python/PyTorch (sử dụng `timm` hoặc `transformers`); đánh giá ưu điểm, hạn chế (độ phức tạp $O(N^2)$, yêu cầu dữ liệu lớn). | Trực quan hóa **Attention Map** (bản đồ tập trung) đè lên ảnh gốc để thấy mô hình đang "nhìn" vào vùng nào của đối tượng. |

---

## Yêu cầu bắt buộc

* **Giải thích được công thức toán học:** Trình bày rõ công thức biến đổi đầu vào (tạo patch + cộng positional embedding), công thức tính Self-Attention và ý nghĩa từng thành phần.
* **Theo dõi được kích thước tensor:** Tính toán và giải thích chính xác kích thước tensor đi qua từng bước (ví dụ: từ ảnh $(3, 224, 224) \rightarrow (196, 768) \rightarrow (197, 768)$ sau khi thêm `[CLS]` token).
* **Làm rõ vai trò đặc biệt của `[CLS]` token và Positional Embedding:** Giải thích tại sao ViT cần thêm token `[CLS]` ở đầu chuỗi để phân loại và điều gì xảy ra nếu loại bỏ thông tin vị trí của các patch.
* **Phân tích tính chất Inductive Bias:** Giải thích được sự đánh đổi giữa CNN (có sẵn inductive bias về tính bất biến tịnh tiến và cục bộ) và ViT (ít inductive bias, cần pre-train trên tập dữ liệu khổng lồ như JFT-300M / ImageNet-21k).
* **Demo chạy được:** Mã nguồn thực thi trơn tru, nêu rõ đang sử dụng trọng số khởi tạo ngẫu nhiên hay mô hình đã huấn luyện sẵn (Pre-trained weights).
* **Trích dẫn nguồn:** Ghi rõ nguồn cho toàn bộ hình ảnh minh họa, công thức và mã nguồn tham khảo (ví dụ: bài báo gốc *"An Image is Worth 16x16 Words"* - Dosovitskiy et al., 2020).

---