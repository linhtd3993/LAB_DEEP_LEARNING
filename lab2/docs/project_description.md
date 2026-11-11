# Project Description: PyTorch Pre-trained CNN Benchmark & Fine-Tuning Pipeline

## 1. Project Name

**PyTorch Pre-trained CNN Benchmark & Fine-Tuning Pipeline**
*(Hệ thống Thực nghiệm và Đánh giá Kiến trúc Mạng Nơ-ron Tiền huấn luyện trên PyTorch)*

---

## 2. Problem Statement

### 2.1. Bai toan Tong quan
Thuc hien bai toan phan loai anh da lop (Multi-class Image Classification) tren bo du lieu tieu chuan CIFAR-10.
- Tong so mau: 60,000 anh mau RGB kich thuoc 32x32 pixel thuoc 10 lop doi tuong (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
- Phan chia du lieu doc lap:
  - Tap Huan luyen (Train set): 40,000 anh.
  - Tap Kiem dinh (Validation set): 10,000 anh (tach doc lap tu 50,000 anh goc, khong su dung de tinh gradient).
  - Tap Kiem thu (Test set): 10,000 anh (chuan hoa doc lap, chi danh gia duy nhat mot lan sau khi hoan tat huan luyen).

### 2.2. Van de Cot loi (Core Problem)
Thiet lap mot pipeline thuc nghiem chuan hoa, mang tinh he thong de khao sat su danh doi (trade-off) giua:
- Toc do hoi tu (Convergence Speed).
- Chi phi tai nguyen tinh toan (Computational Cost): Tong tham so mo hinh, so tham so can huan luyen (trainable parameters), bo nho VRAM tieu thu, va thoi gian huan luyen tren moi epoch.
- Do chinh xac phan loai (Top-1 Accuracy va Per-class Accuracy).

Thuc nghiem duoc thiet ke de so sanh 3 ho kien truc Convolutional Neural Network (CNN) tien huan luyen tren ImageNet-1K:
1. Residual Networks (ResNet): ResNet-18, ResNet-50.
2. Visual Geometry Group (VGG): VGG-16 voi Batch Normalization (`vgg16_bn`).
3. Densely Connected Convolutional Networks (DenseNet): DenseNet-121.

Khao sat duoi 2 chien luoc chuyen giao tri thuc (Transfer Learning Strategies):
- **Feature Extraction (`freeze_all`)**: Dong bang toan bo cac tang trich xuat dac trung (backbone convolutional layers), chi khoi tao lai va huan luyen duy nhat tang phan loai cuoi (Classification Head).
- **Fine-Tuning (`unfreeze_last_block`)**: Dong bang cac khoi tang dau, mo khoa co chon loc khoi tich chap cuoi cung (e.g., `layer4` cua ResNet, khoi dense/transition cuoi cua DenseNet, hoac cac tang conv cuoi cua VGG) ket hop cung tang phan loai moi de tinh chinh trong so phu hop voi tap du lieu dich.

Dong thoi, he thong danh gia tac dong da chieu khi bien thien cac sieu tham so: Learning Rate ($10^{-2}, 10^{-3}, 10^{-4}$), Batch Size (32, 64, 128), Optimizer (SGD with Momentum vs. AdamW), va so luong Epochs.

---

## 3. Target User

- Sinh vien nganh Khoa hoc Du lieu, Tri tue Nhan tao va Ky su Machine Learning / Deep Learning can mot bo khung (framework / template) thuc nghiem Transfer Learning tieu chuan cong nghiep.
- Dac tinh cot loi dap ung cho nguoi dung:
  - Tinh tai lap cao (High Reproducibility): Co dinh seed ngau nhien cho PyTorch, CuDNN, NumPy va Python built-in random.
  - Ma nguon sach va lap trinh huong mo-dun (Clean, Modular Codebase): Tach biet ro rang giua Data Loader, Model Factory, Trainer, Evaluator va Visualizer.
  - Tuong tac linh hoat qua Giao dien Dong lenh (CLI Driven): Cho phep cau hinh moi thu nghiem ma khong can sua truc tiep ma nguon.

---

## 4. Input & Output Specification

### 4.1. Input Specification
1. **Dataset**: CIFAR-10 tu dong tai xuong va luu tru tai thu muc `data/` thong qua `torchvision.datasets.CIFAR10`.
2. **CLI Parameters**:
   - `--model`: Kien truc mo hinh (`resnet18`, `resnet50`, `vgg16_bn`, `densenet121`). Mac dinh: `resnet18`.
   - `--strategy`: Chien luoc hoc chuyen giao (`freeze_all`, `unfreeze_last_block`). Mac dinh: `freeze_all`.
   - `--lr`: Toc do hoc ban dau (Learning Rate). Mac dinh: `1e-3`.
   - `--batch-size`: Kich thuoc mini-batch (tu dong thich ung voi phan cung). Mac dinh: `64`.
   - `--epochs`: So luong chu ky huan luyen. Mac dinh: `15`.
   - `--optimizer`: Thuat toan toi uu (`adamw`, `sgd`). Mac dinh: `adamw`.
   - `--weight-decay`: He so dieu hoa L2. Mac dinh: `1e-4`.
   - `--num-workers`: So luong luong doc du lieu song song cua DataLoader. Mac dinh thich ung: `4` (tren Linux).
   - `--subset-ratio`: Ty le tap du lieu su dung de chay thu nghiem nhanh (0.0 < ratio <= 1.0). Mac dinh: `1.0`.
   - `--dry-run`: Co kiem thu nhanh (chay 1 epoch tren tap du lieu nho) de xac thuc pipeline truoc khi huan luyen quy mo lon.
   - `--seed`: Gia tri hat giong ngau nhien de tai lap ket qua. Mac dinh: `42`.

### 4.2. Output Specification & Directory Layout
Cau truc thu muc luu tru dau ra duoc phan bo thong minh, tach biet ro rang:

```
LAB_DEEP_LEARNING/
|-- data/                       # Du lieu tho va du lieu tai ve (CIFAR-10)
|-- checkpoints/                # Trong so mo hinh tot nhat theo Validation Accuracy
|   `-- best_<model>_<strategy>.pth
|-- runs/                       # Nhat ky TensorBoard phuc vu giam sat truc quan
|   `-- <timestamp>_<model>_<strategy>/
|-- outputs/
|   |-- figures/                # Do hoa danh gia chat luong cao (300 DPI)
|   |   |-- training_history.png
|   |   |-- class_accuracy.png
|   |   `-- confusion_matrix.png
|   `-- reports/                # Bang ket qua danh gia dang van ban va CSV
|       |-- evaluation_metrics.json
|       `-- per_class_report.csv
|-- docs/                       # Tai lieu dac ta kien truc va huong dan su dung
|   `-- project_description.md
`-- src/                        # Ma nguon he thong duoc module hoa
```

Chi tiet cac tep tin dau ra:
- **Model Checkpoint** (`checkpoints/best_<model>_<strategy>.pth`): Chua `model_state_dict`, `optimizer_state_dict`, `epoch`, `val_acc`, va cau hinh thuc nghiem.
- **TensorBoard Event Logs** (`runs/`): Ghi nhan scalar theo tung step va epoch: `Loss/train`, `Loss/val`, `Accuracy/train`, `Accuracy/val`, `LearningRate`.
- **Test Metrics Report**:
  - `outputs/reports/evaluation_metrics.json`: Overall Test Loss, Overall Test Accuracy, Inference Latency per Image, Tong tham so va Trainable Parameters.
  - `outputs/reports/per_class_report.csv`: Precision, Recall, F1-Score va Accuracy cho tung lop trong 10 lop doi tuong.
- **Figures (300 DPI PNG)**:
  - `training_history.png`: Bieu do 2 cot the hien duong cong Loss va Accuracy giua Train va Validation qua tung epoch.
  - `class_accuracy.png`: Bieu do cot so sanh do chinh xac giua 10 lop tren tap Test, danh dau nguong do chinh xac trung binh.
  - `confusion_matrix.png`: Ma tran nham lan duoc chuan hoa (normalized heatmap) the hien ty le du doan dung va cac cap lop de bi nham lan.

---

## 5. Core Features

### 5.1. Model Architecture Inspection (Kham pha va Phan tich Kien truc)
- Tu dong in cau truc toan bo mang va trich xuat phan bo tham so.
- Tinh toan chinh xac:
  - $N_{\text{total}}$: Tong so luong tham so cua mo hinh.
  - $N_{\text{trainable}}$: So luong tham so dang bat `requires_grad=True`.
  - $N_{\text{frozen}}$: So luong tham so bi dong bang (`requires_grad=False`).
- Cung cap bang tong hop tham so truoc va sau khi ap dung chien luoc dong bang/mo khoa.

### 5.2. Dynamic Model Adaptation (Bien doi Tuyen tinh va Thay the Head)
- Su dung Torchvision Weights API hien dai (`weights=...`, khong dung `pretrained=True`).
- Tich hop co che anh xa tu dong den Classification Head phu hop voi tung ho kien truc:
  - **Ho ResNet**: Thay the lop Fully Connected tuyen tinh cuoi:
    ```python
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes=10)
    ```
  - **Ho VGG**: Thay the lop tuyen tinh cuoi trong bo classifier (`.classifier[6]`):
    ```python
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes=10)
    ```
  - **Ho DenseNet**: Thay the lop phan loai tuyen tinh (`.classifier`):
    ```python
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes=10)
    ```

### 5.3. Flexible Weight Freezing Management (Quan ly Trong so Linh hoat)
- Chien luoc `freeze_all`:
  - Thiet lap `param.requires_grad = False` cho toan bo cac tham so thuoc backbone.
  - Chi mo `requires_grad = True` cho Classification Head vua duoc thay the.
- Chien luoc `unfreeze_last_block`:
  - Dong bang cac khoi tang nhe ban dau de bao toan dac trung tong quat (canh, goc, van hoa).
  - Mo khoa cac tham so thuoc khoi tich chap cuoi cung (`model.layer4` o ResNet, `model.features[24:]` o VGG-16, hoac `model.features.denseblock4` va `model.features.norm5` o DenseNet-121) cung Classification Head.
- **Bo loc Optimizer chuan xac**: Chi dua cac tham so co `requires_grad=True` vao Optimizer de tiet kiem bo nho optimizer state va tranh tinh toan gradient du thua:
  ```python
  optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
  ```

### 5.4. Leak-free Data Pipeline (Xu ly Du lieu Chuan xac va Chong Ro ri)
- Su dung gia tri trung binh va do lech chuan chuan hoa cua tap du lieu ImageNet:
  - Mean: `[0.485, 0.456, 0.406]`
  - Standard Deviation: `[0.229, 0.224, 0.225]`
- Dam bao tinh chat khong ro ri du lieu (Data Leakage Prevention):
  - Tao 2 doi tuong Dataset doc lap tu `torchvision.datasets.CIFAR10`: Mot doi tuong gan tap bien doi huan luyen (`train_transforms`), mot doi tuong gan tap bien doi danh gia (`eval_transforms`).
  - Chia tap Train/Validation thong qua phan tach chi so ngau nhien co dinh seed (`torch.utils.data.Subset`), giup tap Validation hoan toan khong bi anh huong boi cac phep bien doi ngau nhien (Data Augmentation).
- Phep bien doi:
  - Train: `Resize((224, 224))` -> `RandomCrop(224, padding=4, padding_mode='reflect')` -> `RandomHorizontalFlip(p=0.5)` -> `ToTensor()` -> `Normalize(mean, std)`.
  - Validation/Test: `Resize((224, 224))` -> `ToTensor()` -> `Normalize(mean, std)`.

### 5.5. Comprehensive Monitoring & Diagnostics (Giam sat va Chan doan Toan dien)
- Theo doi thoi gian thuc tren TensorBoard (`SummaryWriter`).
- Tinh toan thoi gian thuc thi trung binh moi epoch va throughput (samples/second).
- Tinh toan ma tran nham lan day du (10x10) tren tap Test va phan tich cac cap nhan de gay loi (vi du: cat vs. dog, automobile vs. truck).

---

## 6. Technology Stack & Hardware Probing

### 6.1. Technology Stack
- **Ngon ngu lap trinh**: Python 3.10+
- **Deep Learning Framework**: PyTorch (>= 2.0), Torchvision (>= 0.15) voi modern Weights API.
- **Xy ly So hoc va Danh gia**: NumPy, Scikit-learn (tinh confusion matrix, classification report).
- **Truc quan hoa va Do hoa**: Matplotlib (>= 3.7), Seaborn (>= 0.12), TensorBoard.
- **Tien ich Tien trinh**: tqdm.

### 6.2. Thong so Phan cung Thuc te (Actual Hardware Probing Results)
Ket qua kiem tra truc tiep tu he thong thuc thi:
- **He dieu hanh**: Ubuntu Linux x86_64
- **CPU**: 13th Gen Intel(R) Core(TM) i7-13620H (10 physical cores: 6 Performance-cores + 4 Efficient-cores, 16 logic threads)
- **Bo nho RAM He thong**: 14.82 GB (~16 GB)
- **GPU Chuyen dung**: NVIDIA GeForce RTX 3050 Laptop GPU
  - Dung luong VRAM: 6,144 MiB (6.0 GB VRAM)
  - Driver Version: 595.99.02
  - Ho tro CUDA Runtime: 12.x / 13.x

### 6.3. Cau hinh De xuat Toi uu theo Phan cung (Hardware-Tailored Defaults)
Dua tren thong so 6GB VRAM va CPU 16 threads, he thong thiet lap cac thong so mac dinh toi uu nhu sau:

| Thong so | Che do CUDA GPU (RTX 3050 6GB) | Che do CPU Fallback | Giai thich ky thuat |
| :--- | :--- | :--- | :--- |
| **Thiet bi (`--device`)** | `cuda:0` | `cpu` | Tu dong nhan dien thong qua `torch.cuda.is_available()`. |
| **Batch Size (`--batch-size`)** | `64` | `32` | 64 la muc an toan cao tren 6GB VRAM khi anh duoc upscale len 224x224 (tranh loi CUDA Out Of Memory doi voi VGG-16 va DenseNet-121). Voi ResNet-18 co the tang len 128. |
| **Num Workers (`--num-workers`)** | `4` | `2` | Khai thac da luong cua CPU i7-13620H (16 threads) de doc du lieu, dong thoi tranh overhead do IPC giua cac worker. |
| **Pin Memory (`--pin-memory`)** | `True` | `False` | Khoa trang bo nho may chu giup tang toc do truyen du lieu tu RAM len VRAM qua PCIe. |
| **Mixed Precision (`--amp`)** | Khuyen nghi bat (`fp16`) | `False` | Su dung `torch.cuda.amp.autocast()` de tiet kiem 40% VRAM va tang toc do tinh toan tren Tensor Cores cua kien truc Ampere. |
| **Danh sach Model ho tro** | `resnet18`, `resnet50`, `vgg16_bn`, `densenet121` | `resnet18` | Tren CPU khuyen nghi su dung kien truc nhe (`resnet18`) ket hop co `--subset-ratio 0.1` de kiem thu nhanh. |

---

## 7. Evaluation Metrics & Experimental Protocol

### 7.1. Tap Chi so Danh gia (Evaluation Metrics)
1. **Cross-Entropy Loss**:
   $$L = -\frac{1}{N} \sum_{i=1}^{N} \log \left(\frac{e^{z_{i, y_i}}}{\sum_{j=1}^{C} e^{z_{i, j}}}\right)$$
   Ghi nhan sau moi epoch tren ca tap Train va tap Validation.
2. **Top-1 Overall Accuracy (%)**:
   $$\text{Accuracy} = \frac{\sum_{c=1}^{C} \text{TP}_c}{N_{\text{total}}} \times 100\%$$
3. **Per-class Accuracy, Precision, Recall, F1-Score**:
   $$\text{Accuracy}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c} \times 100\%$$
   The hien nang luc nhan dien rieng biet tren tung lop doi tuong.
4. **Normalized Confusion Matrix**:
   $$M_{i, j} = \frac{C_{i, j}}{\sum_{k} C_{i, k}}$$
   Bieu thi xac suat mau thuoc lop thuc te $i$ duoc du doan la lop $j$.
5. **Efficiency & Computational Overhead**:
   - Thoi gian huan luyen moi epoch ($t_{\text{epoch}}$, tinh bang giay).
   - Thoi gian suy dien trung binh moi anh tren tap Test ($t_{\text{inf}}$, tinh bang miligiay).
   - So luong tham so huan luyen va kich thuoc checkpoint (.pth) tren o dia.

### 7.2. Quy trinh Thuc nghiem Chuan (Standard Experimental Protocol)
1. **Khoi tao va Tai lap (Initialization & Seeding)**:
   - Thiet lap seed co dinh: `torch.manual_seed(42)`, `np.random.seed(42)`, `torch.backends.cudnn.deterministic = True`.
2. **Nap va Bien doi Mo hinh (Model Loading & Adaptation)**:
   - Su dung trong so mac dinh hien dai tu Torchvision (vi du: `models.ResNet18_Weights.DEFAULT`).
   - Ap dung co che dong bang theo tham so `--strategy`.
   - Thay the lop phan loai cuoi sang 10 lop dau ra.
3. **Vong lap Huan luyen va Kiem dinh (Train/Val Loop)**:
   - Moi epoch thuc hien huan luyen tren 40,000 anh Train.
   - Danh gia ngay lap tuc tren 10,000 anh Validation.
   - Ghi nhat ky vao TensorBoard.
4. **Chon lua Checkpoint Tot nhat (Model Checkpointing)**:
   - Theo doi Validation Accuracy. Chi luu de trong so vao `checkpoints/best_<model>_<strategy>.pth` khi Validation Accuracy dat ky luc moi (Early-saving policy).
5. **Danh gia Cuoi cung tren Tap Kiem thu Doc lap (Final Independent Testing)**:
   - Tai lai checkpoint tot nhat da luu.
   - Thuc hien suy dien tren 10,000 anh Test chua tung xuat hien trong qua trinh huan luyen hoac chon mo hinh.
   - Xuat toan bo bao cao (JSON, CSV) va 3 bieu do chuan xuat ban (300 DPI) vao thu muc `outputs/`.

