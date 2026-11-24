# Kế hoạch tìm tài liệu và tìm hiểu Vision Transformer (ViT)

Ngày lập và tra cứu: **21/11/2026**.

Mục tiêu: tìm nội dung, hiểu kiến trúc và chuẩn bị kiến thức cho notebook cùng bộ slide trong [task.md](task.md). Tài liệu được chia đúng bốn nhóm: **english / post**, **english / youtube**, **vietnamese / post**, **vietnamese / youtube**.

Trong mục `post`, có cả bài viết, giáo trình, bài báo gốc và tài liệu kỹ thuật. Các mã EP, EY, VP, VY dùng để tham chiếu trong lộ trình học phía dưới. Mức ưu tiên và thời gian học là đề xuất cho bài tập này.

**Phạm vi kiểm tra nguồn:** đã đọc các trang bài viết/tài liệu được chọn; với YouTube, đã đối chiếu tiêu đề, kênh và mô tả hoặc trang giới thiệu của bên xuất bản qua nguồn công khai. Chưa xem toàn bộ video hoặc kiểm chứng toàn bộ mã trong từng hướng dẫn. Các mốc thời gian được ghi riêng khi có trong mô tả của tác giả.

## english

### post:

#### EP1. Bài báo gốc ViT — nguồn đối chiếu chính

**[An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)** — Alexey Dosovitskiy và cộng sự, ICLR 2021.

- **Ưu tiên:** bắt buộc; mức trung cấp.
- **Đọc trọng tâm:** Figure 1; mục 3.1 về kiến trúc; mục 3.2 về fine-tuning; mục 4.3 về dữ liệu; Appendix A về attention.
- **Cần rút ra:** ảnh thành token như thế nào, vai trò CLS và vị trí, công thức Pre-LN, điều kiện của các kết luận so sánh với CNN.
- **Đầu ra học tập:** tự vẽ sơ đồ ViT có nguồn; viết lại bốn công thức kiến trúc bằng lời của mình; ghi rõ cấu hình mà bài đang bàn tới.
- **Dùng cho:** slide 2–12, phần giải thích kiến trúc trong notebook. Có thể đọc [bản HTML của bài báo](https://arxiv.org/html/2010.11929v2) để tìm mục thuận tiện hơn.

#### EP2. Dive into Deep Learning — học kiến trúc kèm mã

**[11.8. Transformers for Vision](https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html)** — giáo trình Dive into Deep Learning.

- **Ưu tiên:** bắt buộc; mức cơ bản đến trung cấp khi đã biết tensor và CNN.
- **Đọc trọng tâm:** Model, Patch Embedding, Vision Transformer Encoder, Putting It All Together; chọn phần mã PyTorch.
- **Cần rút ra:** cách nối các module, phép chiếu patch bằng convolution và cách biểu diễn CLS/positional embedding trong code.
- **Đầu ra học tập:** bảng tên module → shape đầu vào → shape đầu ra; một bản vẽ encoder có hai residual connection.
- **Dùng cho:** notebook tensor trace và slide 3–9. Thông số trong ví dụ giáo trình cần quy đổi về 224×224, P=16, D=768 của đề; không chép nguyên cấu hình minh họa.

#### EP3. TorchVision — chọn pretrained model và preprocessing

**[vit_b_16: model, weights và inference transforms](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html)** — tài liệu chính thức PyTorch/TorchVision.

- **Ưu tiên:** bắt buộc trước khi làm demo; mức thực hành.
- **Đọc trọng tâm:** `ViT_B_16_Weights.IMAGENET1K_V1`, `transforms`, danh sách lớp và mô tả weights.
- **Cần rút ra:** phải dùng preprocessing gắn với checkpoint; phân biệt tên kiến trúc với weights cụ thể.
- **Đầu ra học tập:** phiếu cấu hình demo gồm model, weights, image size, resize/crop, normalization, nhãn lớp và phiên bản thư viện thực tế.
- **Dùng cho:** notebook inference, Top-5 và slide 13–18. Số liệu benchmark trên trang là kết quả của nhà cung cấp, không phải kết quả tập ảnh demo của mình.

#### EP4. TorchVision source — kiểm tra kiến trúc thực tế

**[Mã nguồn torchvision.models.vision_transformer](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/vision_transformer.html)** — mã triển khai chính thức.

- **Ưu tiên:** bắt buộc khi theo dõi tensor; mức trung cấp.
- **Đọc trọng tâm:** `EncoderBlock`, `Encoder`, `VisionTransformer`, `conv_proj`, `_process_input`, `forward` và hàm tạo `vit_b_16`.
- **Cần rút ra:** thứ tự ghép CLS, cộng vị trí, encoder, final LayerNorm, chọn CLS và head trong phiên bản dùng cho demo.
- **Đầu ra học tập:** đánh dấu các điểm cần quan sát shape; lập sơ đồ forward rồi đối chiếu với notebook.
- **Dùng cho:** kiểm chứng slide 5–9, 11. Khi triển khai, đọc đúng phiên bản thư viện đang cài vì đường dẫn `stable` có thể đổi theo thời gian.

#### EP5. Keras — bài thực hành bổ sung để so sánh cách triển khai

**[Image classification with Vision Transformer](https://keras.io/examples/vision/image_classification_with_vision_transformer/)** — Khalid Salama, Keras.

- **Ưu tiên:** tùy chọn, sau khi đã hiểu EP1–EP4.
- **Đọc trọng tâm:** tạo patch, PatchEncoder, các Transformer block và classifier.
- **Cần rút ra:** cách chuyển công thức thành module và tổ chức một thử nghiệm phân loại ảnh.
- **Khác biệt cần ghi nhớ:** ví dụ này flatten toàn bộ đầu ra token để đưa vào classifier, thay vì dùng CLS như cấu hình ViT gốc của bài tập. Đây là khác biệt được chính hướng dẫn nêu rõ.
- **Đầu ra học tập:** bảng đối chiếu “giống / khác ViT-B/16 dùng trong demo”. Không đổi notebook sang Keras chỉ để chạy theo nguồn bổ sung này.

### youtube:

#### EY1. Học trực giác attention trước khi đi vào ViT

**[Attention in transformers, step-by-step | Deep Learning Chapter 6](https://www.youtube.com/watch?v=eMlx5fFNoYc)** — 3Blue1Brown.

- **Ưu tiên:** xem trước nếu chưa rõ Q, K, V; có thể bỏ qua nếu đã nắm attention.
- **Mục tiêu xem:** hiểu cách tính mức liên hệ giữa token, softmax, trộn value và nhiều attention head.
- **Bài tập sau khi xem:** tự vẽ Q/K/V cho một chuỗi nhỏ, rồi thử thay token văn bản bằng patch token trong lời giải thích.
- **Liên hệ bài ViT:** kiến thức nền cho slide 7–8. Video bàn về Transformer/ngôn ngữ, không phải hướng dẫn ViT đầy đủ; không bê nguyên causal mask của mô hình sinh văn bản sang ViT encoder.
- **Nguồn chữ cùng tác giả:** [bài học Attention của 3Blue1Brown](https://www.3blue1brown.com/lessons/attention/), tiện đối chiếu thuật ngữ.

#### EY2. Đọc bài báo ViT cùng phần giải thích

**[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Paper Explained)](https://www.youtube.com/watch?v=TrdevFK_am4)** — Yannic Kilcher.

- **Ưu tiên:** chính; xem sau bài nhập môn và trước hoặc song song EP1.
- **Mục tiêu xem:** theo dõi cách người trình bày diễn giải kiến trúc và kết quả trong bài báo.
- **Cách học:** mở Figure 1 của EP1 bên cạnh; dừng sau từng thành phần để tự diễn giải patch embedding, CLS, position và encoder.
- **Đầu ra học tập:** bản tóm tắt một trang và danh sách câu hỏi còn chưa hiểu. Khi trích dẫn kết quả nghiên cứu trên slide, dẫn EP1 và điều kiện thí nghiệm, không chỉ dẫn video.

#### EY3. Đọc mã ViT từ đầu bằng PyTorch

**[Building a Vision Transformer Model from Scratch with PyTorch](https://www.youtube.com/watch?v=7o1jpvapaT0)** — freeCodeCamp.org; khóa học do Mohammed Al Abrah xây dựng.

- **Ưu tiên:** thực hành bổ sung; xem chọn lọc sau EP2.
- **Mục tiêu xem:** ghép patch embedding và encoder thành mô hình, tổ chức dữ liệu, huấn luyện và dự đoán. [Trang giới thiệu chính thức](https://www.freecodecamp.org/news/build-your-own-vit-model-from-scratch) xác nhận khóa học dùng PyTorch và CIFAR-10.
- **Bài tập sau khi xem:** tìm chỗ đổi shape trong từng module và viết chú thích giải thích tại sao cần thao tác đó.
- **Liên hệ bài tập:** dùng để hiểu code; demo cần nộp vẫn là inference pretrained theo EP3. Không lấy accuracy huấn luyện trong video làm kết quả của notebook của mình.

## vietnamese

### post:

#### VP1. Nhập môn bằng tiếng Việt, có ví dụ tensor

**[Vision Transformer (ViT): Biểu diễn hình ảnh dưới dạng patch](https://www.ultralytics.com/vi/glossary/vision-transformer-vit)** — Ultralytics, trang tiếng Việt.

- **Ưu tiên:** đọc đầu tiên; mức cơ bản.
- **Đọc trọng tâm:** luồng xử lý ảnh, token, encoder và đoạn code minh họa.
- **Cần rút ra:** bộ từ vựng Việt–Anh cho patch, embedding, CLS, position và attention; nhận diện các bước trong sơ đồ ViT.
- **Đầu ra học tập:** giải thích bằng 5–7 câu “ảnh đi qua ViT như thế nào”; tự ghi shape của chuỗi trước encoder.
- **Dùng cho:** hiểu nhanh và chuẩn bị lời thuyết trình. Các chi tiết dùng trong bài nộp cần đối chiếu EP1/EP4.

#### VP2. Bài tổng quan về kiến trúc và ứng dụng

**[Từ A-Z về Vision Transformer: Kiến trúc, cơ chế hoạt động, ứng dụng](https://vnptai.io/vi/blog/detail/vision-transformer)** — VNPT AI.

- **Ưu tiên:** đọc bổ sung; mức cơ bản.
- **Đọc trọng tâm:** cấu trúc mô hình, cơ chế hoạt động, so sánh với CNN và giới hạn.
- **Cần rút ra:** cách mở đầu bài trình bày và các câu hỏi về lý do dùng ViT.
- **Đầu ra học tập:** ba ý cho phần đặt vấn đề; một danh sách nhận định về ViT/CNN cần kiểm tra lại bằng nguồn gốc.
- **Lưu ý khi dùng:** bảng tổng quan có tính khái quát; khi viết slide, bổ sung điều kiện dữ liệu, checkpoint và chi phí thực tế thay cho kết luận tuyệt đối.

#### VP3. Đi từ bài báo đến code bằng tiếng Việt

**[From Vision Transformer Paper to code](https://viblo.asia/p/from-vision-transformer-paper-to-code-PAoJekjkV1j)** — Đạt Nguyễn, Viblo; đăng ngày 18/02/2024.

- **Ưu tiên:** chính cho phần đọc mã tiếng Việt; mức trung cấp. Tiêu đề tiếng Anh nhưng phần diễn giải trong bài là tiếng Việt.
- **Đọc trọng tâm:** phần tổng quan ViT, các công thức, patch/position embedding và các khối mô hình; đọc chọn lọc phần thiết lập dữ liệu.
- **Cần rút ra:** cách nối khái niệm trong paper với tensor và module PyTorch.
- **Đầu ra học tập:** bảng thuật ngữ → công thức → module tương ứng; đánh dấu các bước có thể dùng làm hình minh họa trong notebook.
- **Lưu ý khi dùng:** bài hướng dẫn dùng cấu hình và pipeline dữ liệu riêng. Với pretrained checkpoint của bài tập, dùng preprocessing chính thức EP3; kiểm tra lại thuật ngữ và kích thước bằng EP1/EP4 trước khi đưa vào slide.

### youtube:

#### VY1. Tổng quan ViT bằng tiếng Việt

**[AI Talk Việt | Ep21 – Tại Sao Vision Transformer Đang Dần Thay Thế CNN Trong AI?](https://www.youtube.com/watch?v=VmppMy79r24)** — AI Talk Việt; đăng ngày 06/11/2025.

- **Ưu tiên:** nghe/xem nhập môn để làm quen thuật ngữ và mạch trình bày.
- **Mốc được tác giả liệt kê trong mô tả:** 04:09 về cách hoạt động của ViT; 08:10 về so sánh ViT/CNN; 10:32 về huấn luyện và mở rộng. Chưa đối chiếu các mốc này với transcript đầy đủ.
- **Bài tập sau khi xem:** ghi năm thuật ngữ chưa hiểu, viết hai điểm khác nhau giữa ViT và CNN rồi kiểm tra lại bằng EP1.
- **Lưu ý:** tiêu đề mang tính khái quát. Trong bài tập, không chuyển thành khẳng định “ViT luôn thay thế CNN”; mọi so sánh cần gắn với tác vụ và điều kiện huấn luyện.

Hiện danh sách chọn **một video tiếng Việt có tiêu đề, kênh và mô tả xác định được**. Phần học sâu và đọc code dùng VP3, EP2–EP4 và video tiếng Anh. Có thể tìm thêm theo các truy vấn phía dưới; video mới chỉ bổ sung sau khi xác nhận đúng ngôn ngữ và nội dung.

## Lộ trình học đề xuất: 6 buổi

Thời lượng dưới đây là ngân sách học, ghi chú và làm bài tập nhỏ; không phải thời lượng chính xác của video. Ước lượng tổng cộng **7,5–10 giờ**, chưa tính triển khai notebook hoàn chỉnh hoặc xem hết mọi nguồn tùy chọn.

| Buổi | Mục tiêu | Nguồn sử dụng | Việc cần làm | Đầu ra kiểm tra | Thời gian |
|---|---|---|---|---|---|
| 1 | Hiểu ViT giải quyết vấn đề gì | VP1, VP2; VY1 nếu thích nghe | Đọc tổng quan, ghi thuật ngữ, phác thảo ảnh → chuỗi | Một đoạn giải thích ngắn và sơ đồ tổng quan | 60–90 phút |
| 2 | Hiểu attention và kiến trúc | EY1 nếu cần; EY2, EP1 mục 3.1 | Đọc/xem chọn lọc, tự vẽ Q/K/V và encoder Pre-LN | Phân biệt patch embedding, vị trí, CLS; vẽ đúng hai residual | 90–120 phút |
| 3 | Nắm kích thước tensor | EP2, VP3 | Tự tính ví dụ 224×224; lần theo patch projection và token sequence | Bảng tensor hoàn chỉnh, giải thích 196 → 197 | 90–120 phút |
| 4 | Chuẩn bị notebook pretrained | EP3, EP4; EY3 tùy chọn | Chốt weights/transform; chỉ ra vị trí quan sát tensor và lấy logits | Phiếu cấu hình và khung notebook khớp task.md | 90–120 phút |
| 5 | So sánh và thiết kế phân tích | EP1 mục 4.3, EP2; đối chiếu VP2/VY1 | Lập bảng ViT/CNN; tính tác động giảm P; thiết kế che vùng | Bảng so sánh có điều kiện và giao thức ít nhất 3 case | 60–90 phút |
| 6 | Chuyển kiến thức thành nội dung slide | Ghi chú từ các nguồn trên, task.md | Chọn hình, viết thông điệp mỗi slide, gắn nguồn; luyện giải thích | Dàn ý có trích dẫn cho đủ 10 phần và danh sách việc cần chạy demo | 60 phút |

**Đường học ngắn nhất:** VP1 → EP1 mục 3.1 → EP2 → EP3 → EP4. Xem EY1 khi chưa vững attention và dùng VP3 khi cần phần diễn giải code tiếng Việt. EP5 là tài liệu mở rộng, không bắt buộc trong lộ trình.

## Kế hoạch tìm thêm nội dung

Giữ đúng bốn nhóm khi bổ sung tài liệu. Tìm theo phần kiến thức còn thiếu, ưu tiên đọc nguồn gốc hoặc chính trang của người viết/người dạy.

| Nhóm | Truy vấn gợi ý | Tiêu chí chọn |
|---|---|---|
| english / post | `vision transformer patch embedding tensor shapes`; `ViT pre layer normalization CLS token`; `site:pytorch.org vit_b_16 weights transforms` | Có công thức hoặc code, nêu cấu hình, có nguồn bài báo hoặc tài liệu thư viện |
| english / youtube | `vision transformer paper explained`; `vision transformer PyTorch patch embedding from scratch` | Có mô tả/chapter, sơ đồ hoặc code; xác định rõ biến thể và dữ liệu dùng |
| vietnamese / post | `Vision Transformer chia patch CLS token`; `Vision Transformer kích thước tensor`; `Vision Transformer từ paper đến code` | Diễn giải tiếng Việt, chỉ ra phép biến đổi tensor, có nguồn để đối chiếu |
| vietnamese / youtube | `Vision Transformer tiếng Việt`; `ViT patch embedding bài giảng`; `Transformer thị giác máy tính tiếng Việt` | Mở kiểm tra ngôn ngữ nói; xem thử phần kiến trúc; không chỉ dựa vào tiêu đề tiếng Việt |

Khi gặp nội dung mới, ghi tên tác giả/kênh, đường dẫn trực tiếp, mục cần đọc hoặc mốc cần xem, thông tin lấy được và điểm khác cấu hình của bài tập. Không cần gom nhiều nguồn nói cùng một ý; ưu tiên lấp phần còn thiếu như Pre-LN, tensor trace hoặc preprocessing.

## Mẫu ghi chú sau mỗi nguồn

```text
Mã nguồn / tiêu đề:
Tác giả hoặc kênh:
URL trực tiếp:
Mục / trang / timestamp đã đọc hoặc xem:
Ba ý chính bằng lời của mình:
Công thức hoặc tensor shape cần nhớ:
Cấu hình trong nguồn và điểm khác bài tập:
Hình / đoạn code muốn tham khảo và nguồn gốc:
Phần notebook / số slide sẽ sử dụng:
Câu hỏi chưa hiểu và nguồn sẽ dùng để kiểm tra:
```

Ghi “chưa kiểm chứng” cho nhận định mới chưa đối chiếu. Hình tự vẽ lại cần dẫn nguồn kiến trúc; ảnh, hình chụp hoặc code lấy từ nơi khác cần lưu nguồn và điều kiện sử dụng.

## Checklist hiểu bài trước khi dựng slide

- [ ] Giải thích được vì sao ảnh cần chuyển thành chuỗi patch token.
- [ ] Tự tính N=196, L=197 và bảng tensor với ảnh 224×224×3, P=16, D=768.
- [ ] Phân biệt thao tác ghép CLS với cộng positional embedding.
- [ ] Vẽ được encoder Pre-LN, hai residual connection và final LayerNorm.
- [ ] Giải thích được Q/K/V, attention matrix và nhiều head ở mức phù hợp bài học.
- [ ] Phân tích được khi P giảm từ 16 xuống 8: N tăng 4 lần, thành phần attention bậc hai tăng gần 16 lần; không đồng nhất với thời gian toàn mô hình.
- [ ] Chỉ ra cách từ biểu diễn CLS tạo logits, vai trò cross-entropy, pretraining và fine-tuning.
- [ ] So sánh ViT/CNN theo locality, weight sharing, inductive bias, dữ liệu và chi phí.
- [ ] Biết chọn đúng weights, transform và danh sách lớp cho demo pretrained.
- [ ] Có kế hoạch Top-5, che vùng và ít nhất 3 trường hợp phân tích; hiểu giới hạn của attention heatmap theo đề.
- [ ] Mỗi ý kỹ thuật quan trọng trên slide có nguồn; mọi kết quả demo là số liệu notebook đã chạy.

Kết thúc kế hoạch học này mới chuyển sang thực hiện notebook và dựng slide theo [task.md](task.md). Tài liệu hiện tại là danh mục nguồn và kế hoạch tự học; các ô checklist chưa được đánh dấu hoàn thành thay cho người học.
