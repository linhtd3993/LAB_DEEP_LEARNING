# Nguồn kiến trúc, dữ liệu và mã

Đối chiếu ngày 24/11/2026.

1. Dosovitskiy et al., **An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale**, ICLR 2021. https://arxiv.org/abs/2010.11929. Nguồn kiến trúc patch embedding, CLS, vị trí, encoder Pre-LN và bối cảnh pretraining/fine-tuning. Các sơ đồ trong slide được dựng lại bằng đối tượng chỉnh sửa được.
2. **TorchVision 0.29, vit_b_16**. https://docs.pytorch.org/vision/0.29/models/generated/torchvision.models.vit_b_16.html. Nguồn checkpoint `IMAGENET1K_V1`, preprocessing và danh sách lớp. Bản chạy thực tế: TorchVision 0.29.2+cpu, PyTorch 2.14.1+cpu.
3. **Mã nguồn TorchVision VisionTransformer**. https://docs.pytorch.org/vision/0.29/_modules/torchvision/models/vision_transformer.html. Đối chiếu conv projection, encoder, final LayerNorm và head. Notebook sử dụng module của TorchVision và viết phần trace/kiểm tra riêng.
4. **fastai/Imagenette**. https://github.com/fastai/imagenette. Tập 10 lớp con của ImageNet; lấy 15 ảnh cố định từ split validation bản cạnh ngắn 160 px. URL tải: https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz. Mỗi ảnh có nhãn, tên member trong archive và SHA256 trong `data/manifest.csv`.
5. **Trọng số ViT-B/16 chính thức**. https://download.pytorch.org/models/vit_b_16-c867db91.pth. SHA256: `c867db91d3e12c6cbadabb610d73c24a546bf82d8c03a9fea34f43a712ddb0e9`.
6. **Đề bài trong dự án**: `Task.pdf`, 1 trang. `task.md` và `info.md` cung cấp kế hoạch trước triển khai. Chỉ dẫn mới nhất của người dùng thay phần hướng dẫn sử dụng bằng tệp thư viện và lệnh cài trong cell code đầu.

Ảnh thuộc các tác giả gốc của ImageNet. Archive không cung cấp tên và giấy phép riêng cho từng ảnh; manifest ghi rõ giới hạn này. Không suy diễn giấy phép Apache-2.0 của kho mã Imagenette thành giấy phép ảnh. Bộ ảnh dùng cho minh họa học thuật của bài tập.

Mọi số liệu Top-5, che vùng, tensor và attention trong sản phẩm được tạo từ lần chạy notebook thực tế. Không dùng số liệu benchmark của tài liệu làm kết quả thực nghiệm của dự án. Các biểu đồ và ảnh bảng nằm trong `outputs`; dữ liệu nguồn đi kèm CSV/JSON.
