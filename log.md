# Nhật ký thực hiện Vision Transformer

Ngày thực hiện: 24/11/2026 (Asia/Saigon).

1. Đọc toàn bộ ba tệp nguồn: `Task.pdf` (1 trang), `task.md` và `info.md`.
2. Đối chiếu 10 yêu cầu của đề, phạm vi inference pretrained, 10–20 ảnh và ít nhất 3 trường hợp che vùng.
3. Chọn ViT-B/16 với `IMAGENET1K_V1`, ảnh 224×224, patch 16, D=768, 12 encoder, 12 head và 1.000 lớp.
4. Kiểm tra môi trường: Python 3.10.11, PyTorch 2.14.1+cpu, TorchVision 0.29.2+cpu. Máy chạy CPU, không có CUDA.
5. Đối chiếu bài báo ViT, mã nguồn TorchVision và nguồn dữ liệu Imagenette. Thiết kế tập 15 ảnh có nhãn, chọn theo tên tệp trước khi xem dự đoán.
6. Tạo `requirment.txt` và `requirements.txt`. Theo yêu cầu mới nhất, không tạo tài liệu hướng dẫn sử dụng; cell code đầu notebook chứa lệnh cài thư viện.
7. Chuẩn bị tải dữ liệu và checkpoint. Kết nối mạng trong sandbox không phân giải được DNS; thực hiện tải từ hai URL công khai qua quyền mạng mở rộng.

8. Tải archive Imagenette từ fastai và checkpoint chính thức từ download.pytorch.org. Kiểm tra SHA256 checkpoint với tiền tố `c867db91`; lưu nguồn và SHA256 của từng ảnh vào `data/manifest.csv`.
9. Viết notebook `notebooks/vit_demo.ipynb` gồm 19 cell (10 code, 9 Markdown), có chú thích và giải thích tiếng Việt. Cell code đầu cài từ `requirment.txt` bằng Python của kernel.
10. Triển khai nạp ảnh RGB, transform của checkpoint, gallery dữ liệu và lưu ảnh mô hình thực sự nhìn sau crop.
11. Theo dõi tensor xuyên 12 encoder, kiểm tra unfold + linear tương đương conv projection, và đối chiếu logits trace với `model(x)`.
12. Trích attention thật từ encoder cuối, vẽ ma trận một head và bản đồ CLS trung bình 12 head. Ghi rõ đây không tự chứng minh nguyên nhân dự đoán.
13. Xuất bảng và đồ thị chi phí attention lý thuyết theo patch size. P16 → P8 làm L² tăng khoảng 15,88 lần, không đồng nhất với thời gian toàn mô hình.
14. Chạy inference 15 ảnh và Top-5. Kết quả: Top-1 đúng **13/15**, Top-5 đúng **15/15**. Hai lỗi là cassette player → tape player và church → monastery. Đây là tập minh họa nhỏ, không phải benchmark độc lập.
15. Chạy 9 can thiệp che vùng: 3 ảnh cố định × 3 cửa sổ 64×64 tại (0,0), (80,80), (160,160). Tô 0 trên tensor chuẩn hóa, không đổi crop hoặc số token. So sánh xác suất cùng lớp nhãn thật và cùng lớp Top-1 gốc.
16. Phân tích ba trường hợp: xác suất nhãn thật giảm lớn nhất lần lượt **8,97**, **17,32**, **0,45 điểm phần trăm** trong ba vị trí đã thử. Với cassette player, che trung tâm và góc phải dưới còn làm xác suất tăng; không ép kết quả theo giả thuyết.
17. Xuất **33 hình/biểu đồ**, **18 ảnh bảng**, 15 ảnh crop và 9 ảnh che. Bảng rộng được chia nhóm cột, lặp cột định danh; CSV giữ đủ cột và độ chính xác. Thêm bảng phân tích lỗi thật.
18. Chạy lại toàn bộ notebook bằng kernel mới sau khi sửa xuất bảng. 10 code cell đều thực thi, không có output lỗi; 8 kiểm tra tensor, xác suất, dữ liệu và che vùng đều đạt. Môi trường sandbox hạn chế kernel loopback nên phiên chạy dùng tiến trình cục bộ được cấp quyền.
19. Dựng slide 16:9 với nội dung chỉnh sửa được, hình ảnh thật, bảng native và biểu đồ native. Tách việc lưu thành nhóm nhỏ sau khi dịch vụ slide hết thời gian chờ ở nhóm lớn; kiểm tra trạng thái trước khi thử lại để tránh nhân đôi slide.

20. Rà hình từng slide, rút gọn câu kết luận dài, bổ sung vai trò vị trí/CLS và tăng khoảng cách giữa hai heatmap để không chồng nhãn. Chạy lại notebook bằng kernel mới sau chỉnh sửa; kết quả 13/15 và 15/15 giữ nguyên.
21. Xuất bộ slide đã lưu sang PPTX cục bộ. Bản xuất trung gian dùng asset ID trực tuyến chưa có byte ảnh; đã gắn lại đúng 22 ảnh PNG gốc và xác nhận tất cả ảnh nhúng đều có dữ liệu.
22. Làm tròn dữ liệu biểu đồ trên slide đến 8 chữ số thập phân để Excel nhúng giữ được giá trị. CSV/notebook giữ số gốc, số phần trăm hiển thị trên slide không đổi. Nhúng workbook chứa dữ liệu Top-5 để biểu đồ chỉnh sửa được.
23. Kiểm tra tệp cuối `slides/vit_presentation.pptx`: **20 slide, 6 bảng native, 1 biểu đồ native, 22 ảnh nhúng**. Kiểm tra cấu trúc, geometry, Arial, số slide, workbook và liên kết dữ liệu biểu đồ đều đạt, không có finding. Import lại bằng Artifact Tool thành công. Chưa mở bằng ứng dụng Microsoft PowerPoint native.
24. Render lại toàn bộ PPTX cuối thành 20 PNG trong `outputs/slides`, đối chiếu với các trang đã kiểm tra và xem các trang ảnh/sơ đồ/biểu đồ tiêu biểu sau import. Tệp cuối có 4.880.281 byte, SHA256 `a8b6647926081dcf89cfdb3a43cc49005c2b72840dedb7531572535ca38899b9`.
25. Kiểm tra notebook hợp lệ, đủ 10 code cell đã thực thi, không có lỗi. `predictions.csv` có 120 dòng (24 ảnh/biến thể × Top-5), mỗi nhóm đủ 5 lớp khác nhau, xác suất trong [0,1] và xếp giảm dần. Toàn bộ ảnh trong manifest đầu ra tồn tại.
26. Hoàn thiện `references.md`, giữ nguyên ba tài liệu nguồn. Tổng đầu ra ảnh: **95 PNG** gồm 33 hình/biểu đồ, 18 ảnh bảng, 15 ảnh crop, 9 ảnh che và 20 ảnh slide. Ghi thông tin lần chạy vào `outputs/run_config.json` và danh sách hình/bảng vào `outputs/asset_manifest.json`.

## Sản phẩm đã hoàn thành

- `notebooks/vit_demo.ipynb`: notebook có code, giải thích và output thực tế.
- `requirment.txt`: thư viện theo yêu cầu; `requirements.txt` tham chiếu cùng danh sách.
- `slides/vit_presentation.pptx`: bài thuyết trình 20 trang với nguồn và số liệu thật.
- `data/images/`, `data/manifest.csv`, `data/provenance.json`: ảnh, nhãn và nguồn.
- `outputs/`: bảng CSV, ảnh PNG, attention NPY, cấu hình và phân tích JSON.
- `references.md`: nguồn kiến trúc, checkpoint, ảnh và mã.

Không tạo hướng dẫn sử dụng riêng. Lệnh cài thư viện nằm ngay cell code đầu notebook.
