# Kế hoạch thực hiện bài Vision Transformer (ViT)

Ngày lập: 21/11/2026. Nguồn yêu cầu: [Task.pdf](Task.pdf), trang 1.

## 1. Phạm vi và sản phẩm

Theo yêu cầu đã làm rõ, cần **kế hoạch xây dựng notebook demo ViT, một bộ slide trình bày đầy đủ và prompt dùng với Canva Pro**. Tài liệu này là kế hoạch triển khai; notebook, kết quả inference và bộ slide chưa được tạo/chạy.

Đề xuất bộ trình bày **20 trang, khoảng 18–22 phút**. Đây là phương án tổ chức nội dung, không phải số trang hoặc thời lượng bắt buộc trong đề. Trang 3 là một slide tổng quan có thể sử dụng độc lập.

| Sản phẩm | Nội dung cần có | Trạng thái |
|---|---|---|
| `task.md` | Phân tích đề, kế hoạch notebook, dàn ý và bố cục slide, tiêu chí nghiệm thu | Đã lập |
| `canva_prompt.txt` | Prompt hoàn chỉnh tạo bản nháp bộ slide trên Canva | Đã soạn |
| `notebooks/vit_demo.ipynb` | Inference pretrained, tensor trace, Top-5, thí nghiệm che vùng, phân tích | Chưa triển khai |
| `data/images/` và `data/manifest.csv` | 10–20 ảnh, nhãn tương ứng với lớp mô hình, nguồn ảnh | Chưa chuẩn bị |
| `outputs/` | Bảng dự đoán, hình minh họa, kết quả che vùng, ghi nhận môi trường | Chưa tạo |
| `slides/vit_presentation.pdf` hoặc `.pptx` | Bộ slide hoàn chỉnh có kết quả thực nghiệm thật | Chưa dựng |
| `README.md` và `references.md` | Hướng dẫn chạy, phiên bản thư viện, nguồn kiến trúc/hình/mã | Chưa viết |

Phạm vi thực nghiệm là **inference với mô hình pretrained**. Pretraining, fine-tuning và cross-entropy cần được giải thích trong bài; không có yêu cầu phải huấn luyện ViT từ đầu. Attention heatmap là phần mở rộng, ưu tiên sau khi demo bắt buộc hoàn tất.

## 2. Đối chiếu 10 phần của đề

| Phần | Nội dung phải giải quyết | Minh họa/bằng chứng dự kiến | Trang slide |
|---|---|---|---|
| 1. Đặt vấn đề | Vì sao dùng Transformer cho ảnh; ảnh 2D khác chuỗi văn bản | Hai nhánh biểu diễn CNN và ViT | 2 |
| 2. Tổng quan | Luồng từ patch embedding đến classification head | Sơ đồ có CLS, positional embedding và kích thước | 3 |
| 3. Chia patch | Patch size, số patch, chiều dài chuỗi | Ảnh RGB trên lưới 14×14; so sánh P=16 và P=8 | 4, 10 |
| 4. Patch embedding | Flatten, linear projection và Conv2d tương đương | Sơ đồ patch → vector → embedding, tensor từ notebook | 5, 9 |
| 5. Vị trí và CLS | Phân biệt nội dung patch, vị trí và token phân loại | Chuỗi 196 patch token + 1 CLS; cộng vị trí | 6 |
| 6. Encoder | MSA, MLP, LayerNorm, residual; đúng Pre-LN | Một encoder block có hai nhánh residual | 7 |
| 7. Self-attention | Tương tác patch, attention matrix, nhiều head | Q/K/V và ma trận một head; hình có nhãn “minh họa” | 8 |
| 8. Head và huấn luyện | CLS → logits, cross-entropy, pretraining/fine-tuning | Sơ đồ head; phân biệt huấn luyện và inference | 11 |
| 9. ViT và CNN | Locality, chia sẻ trọng số, inductive bias, dữ liệu, chi phí | Bảng so sánh theo điều kiện; không kết luận tuyệt đối | 12 |
| 10. Demo và phân tích | 10–20 ảnh, Top-5, che các vùng, ít nhất 3 trường hợp | Kết quả notebook, ảnh gốc/che, bảng số liệu và nhận xét | 13–18 |

Trang 1 giới thiệu; trang 19 tổng kết; trang 20 tài liệu tham khảo. Trích dẫn ngắn xuất hiện ngay tại slide dùng kiến thức/hình/mã, không chỉ ở cuối bài.

## 3. Các quyết định kỹ thuật

### 3.1. Cấu hình demo

- Dùng Python, Jupyter, PyTorch, TorchVision, Pillow, Matplotlib và pandas; ghi phiên bản thực tế sau khi chạy thành công.
- Chọn rõ checkpoint `torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)` để khớp ảnh 224×224 và patch 16×16. Không dùng tên `DEFAULT` trong bản bàn giao để tránh mơ hồ checkpoint.
- Dùng `weights.transforms()` cho tiền xử lý và `weights.meta["categories"]` cho tên lớp. Với checkpoint trên: resize cạnh ngắn 256, center crop 224, chuẩn hóa theo cấu hình pretrained. Chi tiết đã đối chiếu [tài liệu TorchVision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html).
- Dùng `model.eval()` và `torch.inference_mode()`. Chạy batch nhỏ, ưu tiên GPU nếu có; có phương án CPU.
- ViT-B/16 triển khai được chọn có 12 encoder block, 12 attention head, D=768, MLP hidden=3072. Kiểm tra lại từ mô hình đã nạp; nguồn: [mã TorchVision](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/vision_transformer.html).
- Không xem tập 10–20 ảnh là benchmark đánh giá tổng quát; đây là bộ demo có nhãn để quan sát hành vi.

### 3.2. Quy ước tensor bắt buộc

Quy ước: `B` là batch size; `P` là cạnh patch; `N` là số patch; `L=N+1` là số token gồm CLS; `D` là embedding dimension; `C` là số lớp. Ảnh RGB luôn có **3 kênh**, không dùng `C` đồng thời cho kênh ảnh và số lớp.

| Bước | Bỏ qua batch, dùng trong slide | Dạng dự kiến trong PyTorch |
|---|---|---|
| Ảnh đầu vào | 224×224×3 | B×3×224×224 |
| Chia patch, lưới 14×14 | 196 patch, mỗi patch 16×16×3 | B×196×768 nếu biểu diễn flatten tường minh |
| Flatten một patch | 16×16×3 = 768 phần tử | Chiều vector đầu vào của phép chiếu là 768 |
| Linear projection sang D=768 | 196×768 | B×196×768 |
| Ghép CLS | 197×768 | B×197×768 |
| Cộng positional embedding | 197×768, không tăng số token | Cộng tensor vị trí 1×197×768, broadcast theo B |
| Sau mỗi encoder block | 197×768 | B×197×768 |
| Final LayerNorm, lấy CLS | 768 | B×768 |
| Classification head | C logits | B×C; demo ImageNet-1K có C=1000 |

Điểm cần giải thích: `16×16×3=768` trùng với `D=768` trong ví dụ này, nhưng linear projection vẫn là phép biến đổi có trọng số học được. Trong cấu hình khác, hai chiều này có thể khác nhau.

Với patch không chồng lấp, phép chiếu dùng chung cho mọi patch có thể triển khai bằng `Conv2d(3, 768, kernel_size=16, stride=16, padding=0)`, tạo `B×768×14×14`, sau đó flatten hai chiều không gian và đổi trục thành `B×196×768`. Sự tương đương yêu cầu cùng trọng số/bias sau khi đổi cách sắp xếp, không chỉ cùng kích thước đầu ra. Đối chiếu [phần conv_proj trong mã mô hình](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/vision_transformer.html).

### 3.3. Ba điểm phải kiểm tra trước khi đưa lên slide

1. **Encoder Pre-LN:** vẽ `u = x + MSA(LN(x))`, sau đó `z = u + MLP(LN(u))`; thêm final LayerNorm sau chuỗi encoder. Nguồn kiến trúc: [bài báo ViT, mục 3.1](https://arxiv.org/html/2010.11929v2#S3.SS1).
2. **Giảm patch size:** giữ ảnh 224×224 và D không đổi, P=16 cho N=196, L=197; P=8 cho N=784, L=785. Ma trận attention một head tăng từ 38.809 lên 616.225 phần tử, tỷ lệ `(785/197)^2 ≈ 15,88`. Khi bỏ CLS, N tăng 4 lần và N² tăng 16 lần. Đây là phép tính cho thành phần attention bậc hai, **không phải khẳng định toàn bộ mô hình chạy chậm 16 lần**. Phép chiếu và MLP còn có chi phí theo L; kernel thực thi ảnh hưởng bộ nhớ thực tế.
3. **Đổi P khác che patch:** so sánh P=8 là phân tích lý thuyết; không sửa trực tiếp patch size của checkpoint B/16 để suy luận. Che vùng chỉ đổi giá trị pixel, ảnh vẫn 224×224 và chuỗi vẫn có 197 token.

## 4. Kế hoạch triển khai notebook

### 4.1. Thứ tự công việc và điều kiện hoàn thành

Ước lượng dành cho một người, chưa tính thời gian tải trọng số hoặc xử lý lỗi môi trường. Có thể điều chỉnh theo lịch học.

| Mã | Công việc | Phụ thuộc | Đầu ra / điều kiện hoàn thành | Ước lượng |
|---|---|---|---|---|
| T1 | Đọc đề, chốt kiến thức và danh sách nguồn | Không | Đủ 10 phần; công thức, thuật ngữ nhất quán | 1–2 giờ |
| T2 | Tạo môi trường, nạp pretrained và chạy một ảnh | T1 | Có logits B×1000 và Top-5; ghi cấu hình máy/thư viện | 1 giờ |
| T3 | Chuẩn bị bộ 10–20 ảnh và manifest | T1 | Mỗi ảnh có nhãn hợp lệ, nguồn và quyền sử dụng | 1–2 giờ |
| T4 | Làm tensor trace và hình patch | T2 | Kích thước khớp toàn bộ bảng mục 3.2 | 1–2 giờ |
| T5 | Chạy inference toàn bộ ảnh, xuất kết quả | T2, T3 | Top-5 mọi ảnh, ảnh tiền xử lý, bảng kết quả lưu được | 1 giờ |
| T6 | Che vùng và phân tích ít nhất 3 trường hợp | T5 | So sánh có kiểm soát, bảng xác suất cùng lớp, hình trước/sau | 2–3 giờ |
| T7 | Hoàn thiện notebook và hướng dẫn chạy | T4, T6 | Restart kernel + Run All thành công; đủ đầu ra và nguồn | 1–2 giờ |
| T8 | Dựng slide bằng prompt, thay bằng kết quả thật | T7 | Đủ 20 trang đề xuất, hình và số liệu khớp notebook | 2–3 giờ |
| T9 | Kiểm tra bản xuất và tập trình bày | T8 | PDF/PPTX không lỗi; demo có bản dự phòng | 1 giờ |

Tổng dự kiến: 11–17 giờ. Có thể làm dàn trang Canva trong khi chuẩn bị notebook; chỉ chốt các slide kết quả sau T6.

### 4.2. Cấu trúc notebook theo cell

| Nhóm cell | Nội dung triển khai | Đầu ra cần quan sát |
|---|---|---|
| 01. Mục tiêu và cách chạy | Giới thiệu ViT, phạm vi demo, đường dẫn dữ liệu, hướng dẫn CPU/GPU | Người khác biết cách bắt đầu và biết lần chạy đầu cần tải checkpoint |
| 02. Môi trường | Import, seed, device, phiên bản Python/torch/torchvision, thư mục đầu ra | Bản ghi cấu hình và môi trường |
| 03. Dữ liệu | Đọc manifest, kiểm tra ảnh, chuyển RGB, kiểm tra class_id | Gallery 10–20 ảnh và nhãn |
| 04. Mô hình và preprocessing | Nạp đúng weights, in transform, eval/inference mode | Ảnh gốc và ảnh sau crop; tensor B×3×224×224 |
| 05. Patch và embedding | Vẽ lưới, flatten minh họa, lấy đầu ra conv projection | Lưới 14×14, vector patch, B×196×768 |
| 06. Tensor trace | Quan sát CLS, vị trí, encoder, final LN, head bằng hook hoặc forward minh họa | Bảng shape thực tế; xác nhận logits từ trace khớp forward chuẩn |
| 07. Inference và Top-5 | Chạy mọi ảnh; softmax để hiển thị; ánh xạ tên lớp từ metadata | Top-5 dạng bảng/biểu đồ và dự đoán đúng/sai theo nhãn |
| 08. Thí nghiệm che vùng | Chọn ít nhất 3 ảnh; mỗi ảnh có các vị trí che được lưu | Ảnh gốc/ảnh che và Top-5 cho từng biến thể |
| 09. Phân tích | So sánh cùng lớp, phân tích lỗi, giới hạn thí nghiệm | Ít nhất 3 case study có số liệu thật |
| 10. Xuất tài nguyên và kết luận | Lưu bảng/hình/config, ghi nguồn, tổng kết | Tài nguyên sẵn dùng trên slide |

Trong phần lý thuyết huấn luyện, giải thích logits đi vào cross-entropy; softmax dùng khi hiển thị xác suất. Demo không thực hiện backward hoặc fine-tuning. Phân biệt classification head tuyến tính của triển khai demo với các cấu hình head trong bài báo gốc; không ngụ ý mọi phiên bản ViT có head giống nhau.

### 4.3. Thiết kế dữ liệu

- Mục tiêu ban đầu: 15 ảnh, trong phạm vi bắt buộc 10–20; kết hợp ảnh đối tượng rõ, vật thể nhỏ/nền phức tạp, góc chụp hoặc ánh sáng khó.
- Chọn nhãn đúng theo danh sách 1000 lớp của checkpoint. Không gán nhãn chung “chó” khi mô hình sử dụng lớp giống chó cụ thể; bỏ ảnh có nhãn không xác minh được khỏi phần tính đúng/sai.
- `manifest.csv` dự kiến gồm `image_id, relative_path, class_id, class_name, source_url, author, license, notes`. Ảnh tự chụp ghi rõ nguồn tự chụp.
- Giữ ảnh gốc riêng; lưu cả hình sau preprocessing để biết mô hình thực sự nhìn phần nào. Kiểm tra center crop không cắt mất đối tượng chính mà không được ghi nhận.
- Chọn dữ liệu và quy tắc thí nghiệm trước khi xem kết quả; khi chọn case để trình bày, ghi rõ tiêu chí chọn. Không suy rộng tỷ lệ đúng của tập minh họa thành độ chính xác ImageNet.

### 4.4. Giao thức che vùng ảnh

1. Với ít nhất 3 ảnh đã có kết quả gốc, dùng chính ảnh sau resize/center crop 224×224 làm hệ tọa độ chung.
2. Trên mỗi ảnh, thử lần lượt các vùng: một vùng trên đối tượng, một vùng nền và một vùng khác để so sánh. Dùng cửa sổ 64×64 nằm hoàn toàn trong ảnh; lưu `(x, y, width, height)` và ảnh minh họa vị trí. Chọn điểm bắt đầu chia hết cho 16 nếu muốn bám theo lưới patch.
3. Giữ nguyên kích thước cửa sổ và cách tô cho các vị trí trong một so sánh. Có thể tô bằng mean RGB của transform, hoặc đặt vùng tương ứng về 0 trong tensor đã chuẩn hóa; phải ghi rõ lựa chọn. Không áp dụng preprocessing hai lần.
4. Chạy lại cùng model/checkpoint ở eval mode. Không đổi crop, nhãn hoặc tham số mô hình giữa ảnh gốc và biến thể.
5. Ghi Top-1, Top-5, xác suất lớp nhãn thật và xác suất lớp Top-1 ban đầu. So sánh `Δp_true = p_true(masked) - p_true(original)`; không lấy chênh lệch giữa hai lớp khác nhau rồi gọi là mức giảm của cùng một lớp.
6. Phân tích ít nhất 3 trường hợp từ số liệu thực: thay đổi lớn/nhỏ khi che đối tượng, ảnh hưởng của nền, và lỗi/nhầm lẫn nếu quan sát được. Kết quả không cần khớp giả thuyết ban đầu.
7. Cố gắng chọn được cả dự đoán đúng và sai thực sự trong tập tối đa 20 ảnh. Nếu không quan sát được lỗi, ghi đúng tình trạng và giới hạn mẫu; không tạo kết quả sai giả để hoàn thiện slide.

Ảnh che là can thiệp có thể tạo dữ liệu khác phân phối tự nhiên. Nhận xét nên nói “dự đoán thay đổi khi che vùng này”, không tự kết luận mô hình hiểu đối tượng như con người. Nếu thêm attention heatmap, ghi rõ đây không tự động là lời giải thích đầy đủ cho quyết định, đúng lưu ý của đề.

### 4.5. Đầu ra và kiểm tra notebook

- Khi chạy demo các cell ra ảnh đều xuất ra outputs
- `outputs/predictions.csv`: một hàng mỗi ảnh/biến thể/rank Top-5, gồm ID ảnh, loại biến thể, rank, class_id, class_name, probability.
- `outputs/occlusion_summary.csv`: ID ảnh, vùng che, kích thước/kiểu tô, Top-1 trước/sau, p_true trước/sau, Δp_true, true_class_in_top5.
- `outputs/tensor_shapes.csv`: tên bước, shape kỳ vọng, shape quan sát được.
- `outputs/figures/`: patch grid, tensor trace, Top-5 và 3 case study; xuất PNG đủ rõ để đặt lên slide.
- `outputs/run_config.json`: checkpoint, transform, seed, device, phiên bản thư viện, thời điểm chạy; lưu trong giai đoạn triển khai.
- Kiểm tra shape các bước trọng yếu, class_id hợp lệ, xác suất nằm trong [0,1], Top-5 có 5 lớp khác nhau và xếp giảm dần. Tổng xác suất của **toàn bộ 1000 lớp** xấp xỉ 1; không yêu cầu riêng Top-5 cộng thành 1.
- Khi forward minh họa các bước nội bộ, đối chiếu logits với `model(x)` trên cùng đầu vào bằng sai số số học nhỏ. Gỡ hook sau khi dùng để notebook không bị ảnh hưởng khi chạy lại.
- Restart kernel + Run All từ đầu, kiểm tra mọi đường dẫn tương đối và toàn bộ bảng/hình được tạo lại. Giữ đầu ra đã chạy làm bản dự phòng khi trình bày.

### 4.6. Cấu trúc thư mục dự kiến

```text
Vision_Transfomer/
├── Task.pdf
├── task.md
├── canva_prompt.txt
├── README.md
├── references.md
├── requirements.txt
├── notebooks/
│   └── vit_demo.ipynb
├── data/
│   ├── manifest.csv
│   └── images/
├── outputs/
│   ├── predictions.csv
│   ├── occlusion_summary.csv
│   ├── tensor_shapes.csv
│   ├── run_config.json
│   └── figures/
└── slides/
    ├── vit_presentation.pdf
    └── vit_presentation.pptx
```

Đây là cấu trúc sẽ tạo trong giai đoạn triển khai, không phải danh sách tệp đã tồn tại. PDF hoặc PPTX là đủ theo đề; nên giữ cả hai nếu tiện trình bày và chỉnh sửa.

## 5. Kế hoạch dựng bộ slide đầy đủ

### 5.1. Hệ thống thiết kế

- Khổ 16:9; ngôn ngữ tiếng Việt, giữ thuật ngữ tiếng Anh cần thiết trong ngoặc ở lần đầu.
- Phong cách bài giảng kỹ thuật, nền sáng `#F8FAFC`, chữ `#0F172A`, patch màu xanh `#2563EB`, CLS màu cam `#F97316`, thông tin vị trí màu tím `#7C3AED`.
- Dùng font sans-serif có đủ dấu tiếng Việt. Tiêu đề khoảng 32–40 pt, nội dung 22–28 pt, chú thích 14–16 pt; nếu chữ không vừa thì rút gọn hoặc chia hình, không thu nhỏ toàn slide.
- Một thông điệp chính mỗi trang; thông thường 3–4 ý ngắn. Đưa diễn giải dài vào ghi chú thuyết trình.
- Dùng hình khối, mũi tên, bảng và biểu đồ chỉnh sửa được. Giữ cùng màu và cách ký hiệu qua toàn bộ bài.
- Hình patch, ảnh thử nghiệm và biểu đồ kết quả lấy từ notebook. Hình minh họa tự vẽ phải ghi rõ; không dùng heatmap trang trí như thể là kết quả đo.
- Mỗi trang kỹ thuật ghi nguồn ngắn ở chân; trang tham khảo có liên kết đầy đủ. Lưu tác giả/giấy phép cho ảnh bên ngoài.

### 5.2. Storyboard 20 trang

| # | Tiêu đề / thông điệp | Nội dung trên trang | Bố cục và hình cần dựng | Ghi chú khi nói |
|---|---|---|---|---|
| 01 | Vision Transformer: từ ảnh đến chuỗi token | Chủ đề, tên người/nhóm; 3 mục tiêu: kiến trúc, tensor, demo | Tiêu đề lớn bên trái, ảnh được chia patch bên phải | Giới thiệu câu hỏi trung tâm: Transformer nhận ảnh như thế nào? |
| 02 | Vì sao đưa ảnh vào Transformer? | Ảnh là lưới 2D; Transformer nhận chuỗi; CNN và ViT có cách tổ chức xử lý khác nhau | Hai nhánh: ảnh → feature map CNN; ảnh → patch token ViT | Phân biệt pixel/patch với token văn bản; không đồng nhất patch với một vật thể |
| 03 | Một ảnh đi qua ViT như thế nào? | Ảnh → patch embedding → CLS + vị trí → encoder → CLS/head | Pipeline 5 khối có shape; thiết kế chi tiết ở mục 5.3 | Dẫn đường cho toàn bài; chỉ rõ final LN trước head |
| 04 | Chia ảnh thành 196 patch | 224/16=14; 14×14=196; mỗi patch 16×16×3 | Ảnh lưới 14×14 và một patch phóng to | Giải thích N và P; lưới phải có đúng số hàng/cột |
| 05 | Patch embedding biến pixel thành vector | Flatten 768 → Linear → D=768; projection dùng chung; Conv2d tương đương | Một patch → vector → khối linear; nhánh conv ở dưới | 768 đầu vào trùng D chỉ trong cấu hình ví dụ; vẫn có trọng số học |
| 06 | Patch, vị trí và CLS có ba vai trò | Nội dung ảnh / thứ tự vị trí / token phân loại; L=197 | Dải token cam + xanh, hàng positional embedding tím và dấu cộng | CLS được ghép vào chuỗi; vị trí được cộng, không nối thêm 197 token |
| 07 | Bên trong một encoder block | LN → MSA → cộng residual; LN → MLP → cộng residual | Hai nhánh residual rõ ràng, nhãn “Pre-LN”; B/16 lặp 12 lần | LayerNorm đặt trước mỗi nhánh xử lý; shape toàn block giữ nguyên |
| 08 | Self-attention kết nối các patch | Q/K/V; A=softmax(QKᵀ/√dₖ); output=AV; nhiều head | Sơ đồ truy vấn vài vùng và ma trận một head 197×197 | Với 12 head, dₖ=64; mỗi head học quan hệ, không gán cứng một ý nghĩa |
| 09 | Theo dõi tensor từ ảnh đến logits | Toàn bộ bảng kích thước bắt buộc trong đề | Bảng hai cột, đánh màu điểm 196 → 197 → 768 → C | Bỏ batch trên slide; notebook có thêm B và thứ tự kênh PyTorch |
| 10 | Patch nhỏ hơn làm attention tốn hơn | P16: N196/L197; P8: N784/L785; score matrix ≈15,88× | Hai lưới cùng kích thước ảnh và số phần tử ma trận | Chỉ so thành phần attention bậc hai; đây là lý thuyết, không đổi checkpoint demo |
| 11 | Từ CLS đến dự đoán và huấn luyện | Final LN → CLS → head → C logits; CE khi train; pretraining/fine-tuning | Nhánh train và inference; softmax → Top-5 ở nhánh inference | Demo dùng checkpoint sẵn, C=1000; không đưa accuracy chưa đo |
| 12 | ViT và CNN: so sánh theo cơ chế | 5 hàng: locality, chia sẻ trọng số, inductive bias, dữ liệu, chi phí | Bảng hai cột CNN/ViT, câu kết phụ thuộc điều kiện | Cả hai có chia sẻ trọng số; CNN tích lũy ngữ cảnh qua nhiều lớp; không tuyên bố bên nào luôn hơn |
| 13 | Thiết kế demo có thể chạy lại | Checkpoint, transform, 10–20 ảnh có nhãn, eval/inference mode | Pipeline dữ liệu → preprocessing → model → Top-5 | Nêu số ảnh thực tế sau khi chuẩn bị, thiết bị và phiên bản đã chạy |
| 14 | Mô hình dự đoán gì? | Ảnh sau crop, nhãn thật, Top-5 và score thực tế | Ảnh trái, biểu đồ thanh Top-5 phải; ô ghi đúng/sai | Dữ liệu lấy từ CSV; chưa chạy thì để placeholder, không tạo thanh giả |
| 15 | Che vùng ảnh để kiểm tra độ nhạy | Ảnh gốc + các vị trí che cùng 64×64; cùng checkpoint và preprocessing | Dãy ảnh cùng tỷ lệ; công thức Δp_true ở dưới | Che pixel không xóa token; vị trí/diện tích/cách tô được lưu |
| 16 | Trường hợp 1: che vùng đối tượng | Ảnh gốc/che; Top-1 và p_true trước/sau; nhận xét thật | So sánh hai ảnh và bảng nhỏ | Kết quả có thể giảm hoặc không giảm; không ép theo giả thuyết |
| 17 | Trường hợp 2: che vùng nền | Cùng các trường dữ liệu trang 16; nhận xét về thay đổi quan sát | Giữ bố cục giống trang 16 | So với che đối tượng, giữ diện tích/kiểu tô để hạn chế nhiễu |
| 18 | Trường hợp 3: lỗi hoặc ảnh khó | Một case thực, nhãn thật, Top-5, các yếu tố có thể gây nhầm | Ảnh, bảng trước/sau và 2–3 nhận xét | Phân biệt quan sát với giả thuyết; nếu chưa có lỗi, ghi trung thực |
| 19 | Ba điều cần nhớ | Ảnh → token; 197×768 xuyên encoder; lựa chọn ViT/CNN tùy điều kiện | Ba thẻ kết luận và một ô giới hạn demo | Nêu giới hạn tập nhỏ, che vùng và attention heatmap nếu có |
| 20 | Tài liệu tham khảo | Bài báo, mã gốc, TorchVision, nguồn ảnh/mã demo | Danh sách nguồn gọn, liên kết đầy đủ | Không để nguồn giả hoặc URL chưa xác minh; ghi phần tự vẽ |

Nguồn cho kiến trúc trong storyboard: [bài báo ViT](https://arxiv.org/abs/2010.11929) và [mã TorchVision](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/vision_transformer.html). Công thức tensor và yêu cầu demo lấy từ đề; số liệu so sánh P=8 được tính từ N và L.

### 5.3. Bố cục chi tiết cho một slide mẫu: trang 3

**Tiêu đề:** “ViT biến ảnh thành chuỗi token để phân loại”.

**Thông điệp:** Mỗi patch được biểu diễn bằng một vector; encoder trao đổi thông tin giữa các token; biểu diễn CLS đi vào classification head.

```text
┌─────────────────────────────────────────────────────────────────────┐
│ ViT biến ảnh thành chuỗi token để phân loại                          │
│                                                                    │
│ [Ảnh RGB] → [Patch embedding] → [+ CLS, + vị trí] → [Encoder] → [Head]│
│ 224×224×3      196×768              197×768        197×768    C logits│
│ lưới 14×14    flatten → linear                    ×12        CLS 768│
│                                                    └─ Final LN ────│
│                                                                    │
│ [Patch: nội dung ảnh] [Vị trí: cộng vào token] [CLS: token phân loại] │
│ Ghi chú: bỏ batch; P=16, D=768. Nguồn: Dosovitskiy et al.; TorchVision│
└─────────────────────────────────────────────────────────────────────┘
```

Sơ đồ trên là bản bố trí khái niệm; khi dựng cần đặt final LN đúng trên đường từ encoder đến bước lấy CLS/head, không vẽ thành nhánh residual mới.

- Vùng tiêu đề chiếm khoảng 15% chiều cao; pipeline chiếm 60%; chú giải và nguồn dùng phần còn lại.
- Pipeline gồm 5 khối cùng hàng; ảnh ở khối đầu có thể thu gọn, nhưng kích thước tensor phải dễ đọc.
- Đặt chữ “ghép CLS” và “cộng vị trí” thành hai thao tác riêng trong khối thứ ba, tránh gây hiểu nhầm dấu cộng.
- Khối cuối ghi `Final LN → lấy CLS (768) → Linear → C logits`; nhãn C=1000 chỉ là cấu hình demo, không thay định nghĩa C tổng quát.
- Lời dẫn gợi ý: “Ảnh RGB 224×224 được chia thành 196 patch. Mỗi patch đi qua phép chiếu để có vector 768 chiều. Thêm một CLS token tạo chuỗi 197 token, rồi cộng thông tin vị trí. Các encoder giữ kích thước chuỗi và trao đổi thông tin giữa token. Cuối cùng, sau LayerNorm, biểu diễn CLS được đưa qua head để tạo C logits.”

## 6. Prompt và cách dựng trên Canva Pro

Prompt dùng trực tiếp nằm trong [canva_prompt.txt](canva_prompt.txt). Đây là **kịch bản bằng ngôn ngữ tự nhiên**, không phải mã Python/JavaScript hay chương trình tự đăng nhập Canva.

1. Mở Canva AI, chọn Presentation hoặc mô tả yêu cầu tạo bài trình bày. Đây là luồng được mô tả trong [hướng dẫn chính thức Canva](https://www.canva.com/help/using-magic-presentations/).
2. Sao chép prompt trong tệp; nếu giao diện giới hạn đầu vào, gửi phần yêu cầu thiết kế và dàn ý trước, rồi bổ sung nội dung theo nhóm trang. Có thể đính kèm tệp khi giao diện hỗ trợ.
3. Kiểm tra outline đủ 20 trang đề xuất rồi tạo bản nháp. Canva có thể tự thay đổi dàn trang/nội dung, vì vậy cần đối chiếu lại với storyboard.
4. Thay các ô `[CHƯA CHẠY DEMO]` bằng hình và số liệu xuất từ notebook. Tạo biểu đồ từ số thực, kiểm tra nhãn, thứ tự và xác suất.
5. Rà lại công thức, tensor, lưới patch, CLS, phép cộng vị trí và sơ đồ Pre-LN. Sơ đồ sai cần sửa thủ công bằng hình khối và mũi tên.
6. Thêm nguồn hình/mã, thông tin nhóm và liên kết notebook. Xuất PDF hoặc PPTX, mở lại từng trang để kiểm tra dấu tiếng Việt, font, công thức và hiện tượng tràn chữ.

## 7. Checklist nghiệm thu

### Hồ sơ kế hoạch hiện tại

- [x] Đọc đủ một trang Task.pdf, gồm bảng nội dung, ví dụ tensor, demo và sản phẩm nộp.
- [x] Đối chiếu đủ 10 mục với notebook và slide.
- [x] Xác định checkpoint, giao thức demo, cách lưu và phân tích kết quả.
- [x] Lập storyboard 20 trang và bố cục chi tiết một slide tổng quan.
- [x] Soạn prompt Canva; phân biệt placeholder với kết quả đã chạy.

### Công việc cần hoàn thành khi triển khai

- [ ] Có notebook chạy lại từ đầu và hướng dẫn môi trường/đường dẫn.
- [ ] Có 10–20 ảnh hợp lệ, nhãn khớp metadata lớp và nguồn sử dụng.
- [ ] Tiền xử lý đúng checkpoint; có Top-5 cho mọi ảnh.
- [ ] Tensor trace khớp 196 patch, 197 token, D=768 và C logits.
- [ ] Giải thích đúng patch embedding, positional embedding, CLS và Pre-LN.
- [ ] Có thí nghiệm che các vùng và ít nhất 3 case study thực tế.
- [ ] Có phân tích đúng/sai, giới hạn và giả thuyết; không ngụy tạo kết quả.
- [ ] Phân biệt tăng chi phí attention với tăng thời gian toàn mô hình.
- [ ] Bảng ViT/CNN đề cập locality, weight sharing, inductive bias, dữ liệu và tính toán.
- [ ] Slide bao phủ 10 phần, chứa nguồn kiến trúc/hình/mã, thay hết placeholder trước khi nộp.
- [ ] PDF/PPTX đã mở kiểm tra; có notebook và ảnh/kết quả dự phòng cho demo.

## 8. Tài liệu tham khảo ban đầu

1. **Đề bài:** [Task.pdf](Task.pdf), trang 1. Nguồn xác định phạm vi và yêu cầu bắt buộc.
2. **Dosovitskiy et al. (ICLR 2021):** [An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). Dùng cho kiến trúc, CLS, positional embedding, Pre-LN và bối cảnh huấn luyện. Khi tự vẽ sơ đồ, ghi “Vẽ lại dựa trên Dosovitskiy et al.”.
3. **Mã gốc của tác giả:** [google-research/vision_transformer](https://github.com/google-research/vision_transformer). Nguồn mã tham khảo; demo dự kiến dùng TorchVision, không ghi rằng đã chạy kho mã gốc này.
4. **Tài liệu checkpoint:** [TorchVision vit_b_16](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html). Dùng để kiểm tra weights, preprocessing và metadata; ghi phiên bản thực tế khi triển khai.
5. **Mã triển khai:** [TorchVision vision_transformer](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/vision_transformer.html). Dùng đối chiếu patch projection, encoder và cấu hình B/16.
6. **Hướng dẫn công cụ:** [Create a presentation with Canva AI](https://www.canva.com/help/using-magic-presentations/). Dùng để thao tác Canva, không phải nguồn kiến thức ViT.
7. **Nguồn ảnh và mã demo:** sẽ bổ sung vào manifest và references khi chọn dữ liệu/triển khai. Chỉ hoàn tất mục này khi có nguồn cụ thể; không tạo trích dẫn giả.

Các nguồn trực tuyến trên đã được đối chiếu ngày 21/11/2026. Thông số, hình và kết quả thực nghiệm cuối cùng phải phản ánh đúng phiên bản notebook đã chạy.
