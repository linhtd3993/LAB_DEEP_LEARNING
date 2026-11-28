# Nghiệm thu Lab 2 · CIFAR-10

## Chọn checkpoint và kiểm tra kết quả

Trong tám run hoàn chỉnh, EXP_02 (ResNet-18, mở layer4, AdamW, LR 0,0001) đứng đầu theo Validation Accuracy, đạt 94,35% tại epoch 13. EXP_06 đứng kế tiếp với 94,23%. Notebook đã ghim checkpoint và CSV của EXP_02, chạy lại trên toàn bộ Test và đối chiếu ID, nhãn và xác suất mười lớp với lần chạy gốc. Trên 10.000 ảnh, EXP_02 đúng 9.363 ảnh, sai 637 ảnh, loss 0,2304 và macro-F1 0,9360. EXP_06 đạt 93,86% Test so với 93,63% của EXP_02. Thứ hạng Test được báo cáo sau khi chọn checkpoint. [Bảng tám cấu hình](../README.md#9-experimental-results--benchmark-table) · [Report EXP_02](../outputs/experiments/resnet18_unfreeze_last_block_20261121_115234_018455_f40df826/report.json) · [Notebook đã chạy](../notebooks/02_model_evaluation.ipynb)

## Lớp và lỗi có hệ thống

Mỗi lớp có 1.000 ảnh Test. Tỷ lệ nhận đúng trong từng lớp (recall) là airplane 97,6%, frog 96,8%, automobile 96,4%, ship 96,1%, deer 95,6%, horse 95,4%, truck 95,0%, bird 90,8%, dog 90,1% và cat 82,5%. Trong 175 ảnh cat bị sai, có 89 ảnh được đoán là dog. Chiều dog → cat có 44 ảnh. Hai chiều chiếm 133/637 lỗi (20,9%). Các chiều thường gặp tiếp theo là bird → deer (32), truck → automobile (31), cat → frog và ship → airplane (cùng 26). [Ma trận nhầm lẫn](../outputs/model_analysis/best_resnet18_unfreeze_last_block_20261121_115234_018455_f40df826/confusion_matrices.png) · [Dự đoán từng ảnh](../outputs/experiments/resnet18_unfreeze_last_block_20261121_115234_018455_f40df826/predictions.csv)

Bốn ví dụ dưới đây là các cực trị confidence trong nhóm đúng và sai. Xác suất đã làm tròn. Cột cuối đo xác suất của **chính lớp dự đoán** sau khi che 20% pixel có Grad-CAM cao nhất ở layer4.

| ID Test | Nhãn thật → dự đoán | Bằng chứng trên ảnh và heatmap | Xác suất trước → sau che |
|---|---|---|---|
| [4](../outputs/model_analysis/best_resnet18_unfreeze_last_block_20261121_115234_018455_f40df826/gradcam_4.png) | frog → frog | Vùng sáng phủ phần thân nâu giữa nền lá. Dự đoán đúng và rất nhạy với việc che vùng này. | 100,0% → 0,2% |
| [5708](../outputs/model_analysis/best_resnet18_unfreeze_last_block_20261121_115234_018455_f40df826/gradcam_5708.png) | bird → bird | Vùng sáng ở phần dưới mỏ đang mở. Bird 29,9%, dog 29,4%, horse 26,8%. Model chỉ chọn đúng sát nút. Che vùng sáng còn làm xác suất bird tăng. | 29,9% → 55,3% |
| [2405](../outputs/model_analysis/best_resnet18_unfreeze_last_block_20261121_115234_018455_f40df826/gradcam_2405.png) | cat → frog | Ảnh có vật thể xanh trông giống ếch. CAM frog phủ thân, CAM cat sáng gần góc dưới trái. Đã kiểm tra trực tiếp nhãn gốc trong CIFAR-10 test_batch là 3 (cat). Đây là trường hợp đáng rà soát nhãn, vẫn được tính sai theo dữ liệu gốc. | 100,0% → 0,1% |
| [4208](../outputs/model_analysis/best_resnet18_unfreeze_last_block_20261121_115234_018455_f40df826/gradcam_4208.png) | bird → dog | Ảnh chim nhỏ trên cành. CAM dog sáng quanh thân và vùng tiếp giáp cành, CAM bird lệch sang phải. Dog 32,0%, deer 27,8%, frog 15,3%. Model chọn sai với khoảng cách Top-1/Top-2 chỉ 4,2 điểm phần trăm. | 32,0% → 0,0% |

Heatmap cho thấy vùng liên quan tới điểm lớp trên từng ảnh. Việc che pixel cũng làm thay đổi phân phối đầu vào. Bốn trường hợp này chưa đủ để kết luận đặc trưng gây ra quyết định hoặc nguyên nhân chung của mọi lỗi cat/dog.

## Đối chiếu thiết kế

| Tiêu chí trong project_design.md | Kết quả và bằng chứng |
|---|---|
| M1 · cấu hình, shape và split | Đạt. EDA chạy đủ 7 cell, tái tạo seed 42 và các tập 40.000/10.000/10.000. Train/Val rời chỉ số. Validation/Test cho tensor giống nhau khi áp transform nhiều lần. Unit test kiểm tra shape và miền nhãn. [EDA](../notebooks/01_cifar10_eda.ipynb) · [Hình EDA](../outputs/eda/) |
| M2 · factory, head, hai strategy | Đạt ở mức chức năng. ResNet-18 có 5.130 tham số trainable khi đóng băng và 8.398.858 khi mở layer4. Test kỹ thuật chạy bốn kiến trúc × hai strategy với trọng số khởi tạo. Benchmark có sáu tổ hợp model/strategy đã định trong README. [Test factory](../tests/test_model_factory.py) · [Test tám tổ hợp](../tests/test_model_analysis.py) |
| M3 · AMP, checkpoint tốt nhất, thời gian, TensorBoard | Đạt. Đối chiếu epoch đầu có Validation Accuracy cực đại với best_epoch cho cả tám run. Mỗi run có checkpoint và năm chuỗi scalar, mỗi chuỗi đủ 15 epoch. CLI một epoch dry-run kết thúc thành công trong đợt nghiệm thu. [Log CLI](../outputs/acceptance_smoke.log) · [Bảng đường dẫn artifact](../outputs/experiment_summary.csv) |
| M4 · CLI, Test, per-class, confusion matrix, ba hình 300 DPI | Đạt đầu ra chức năng. Đã đọc metadata DPI và kiểm tra ba PNG của cả tám run, 10.000 ID Test theo thứ tự và accuracy từ CSV. CLI hai epoch với subset 0,05 cũng hoàn tất trong thư mục tạm riêng. Các hình chính thức nằm ở outputs/&lt;run_id&gt;/ thay cho đường dẫn dùng chung trong đặc tả. [Log subset](../outputs/acceptance_subset.log) · [Ba hình và report](../outputs/experiments/resnet18_unfreeze_last_block_20261121_115234_018455_f40df826/) |
| Chi tiết thiết kế khác | Windows dùng num_workers=0 thay cho 4. Trainer ghi giây/epoch, chưa ghi trực tiếp samples/s. Run ID dùng model, strategy, timestamp và hậu tố riêng; LR và batch lưu trong report. [Trainer](../src/trainer.py) · [Config](../src/config.py) |

## Đối chiếu quy trình phát triển

| Bước trong development_workflow.md | Kết quả và bằng chứng |
|---|---|
| 1 · Describe | Có đặc tả chữ ký hàm, kiểu dữ liệu, tensor và giới hạn phần cứng. Chưa có hồ sơ xác nhận thời điểm chốt đặc tả cho từng module. [Thiết kế §2–5](project_design.md) |
| 2 · Generate | Đạt tiêu chí đầu ra về vị trí file, cú pháp và import. Chưa chứng nhận 100% type hints/PEP8 hay trình tự sinh từng module. [Mã nguồn](../src/) · [Log hồi quy](../outputs/regression_tests.log) |
| 3 · Review | Đạt năm mục kỹ thuật. Factory dùng weights API; loader tách transform; optimizer lọc requires_grad; trainer và evaluator dùng đúng train/eval, no_grad và device. BatchNorm đóng băng có test kiểm tra running statistics. [Entry point](../main.py) · [Trainer](../src/trainer.py) · [Evaluator](../src/evaluate.py) · [Test](../tests/test_main.py) |
| 4 · Run | CLI dry-run exit 0. Chưa đạt đúng mô tả một batch train vì 128 ảnh train ở batch 64 tạo hai batch, validation một batch. Chưa có phép đo xác nhận toàn lệnh dưới 10 giây hay bằng chứng smoke từng module trước full training. [Log smoke](../outputs/acceptance_smoke.log) |
| 5 · Test | Đạt 26/26. Có kiểm tra shape, nhãn, split, logits, số tham số, loss ban đầu, loss giảm trên 50% sau 50 bước trên một batch, checkpoint, log, đồ thị và dữ liệu từng ảnh. [Log nghiệm thu](../outputs/acceptance_tests.log) · [Các test](../tests/) |
| 6 · Refactor | Đạt phần hồi quy. Bộ test đã qua ở Giai đoạn 4 và chạy lại thành công khi nghiệm thu. Có ngoại lệ workers trên Windows nêu trên. Chưa có bằng chứng lịch sử cho từng lần refactor riêng lẻ. [Log Giai đoạn 4](../outputs/regression_tests.log) · [Log cuối](../outputs/acceptance_tests.log) |
| 7 · Commit | Chưa đạt. git rev-parse tại Lab 2 báo thư mục không thuộc Git repository. Chưa xác nhận được nhánh lab2, commit nguyên tử theo milestone hoặc working tree sạch. |

Các log smoke/subset phục vụ kiểm tra CLI. Tám dòng benchmark trong README lấy từ tám run toàn dữ liệu của Giai đoạn 5–6. Log console của EXP_01 dùng [tên file demo ban đầu](../outputs/full_demo_resnet18_freeze_all_20261121_111822.log), khác với tên quy ước được ghi ở cột log của bảng trung gian.
