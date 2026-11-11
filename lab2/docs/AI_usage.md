# AI Usage Report: Academic Transparency & Spec-Driven Development

Bao cao minh bach ve pham vi, quy trinh va muc do anh huong cua cac cong cu Tri tue Nhan tao (AI) trong qua trinh phat trien bai thuc hanh **Practice 2: Hands-on practice with pre-trained neural network architectures**.

---

## 1. AI Tools Used

Du an ap dung cac he thong AI tao sinh va Agentic AI nhu cong cu ho tro ky thuat, tuan thu nguyen tac Human-in-the-loop (con nguoi giu vai tro quyet dinh va kiem chung toan dien):

| Cong cu / Mo hinh | Vai tro ky thuat | Pham vi ap dung trong Du an |
| :--- | :--- | :--- |
| **Google Gemini (Advanced / Flash)** | Co van kien truc va phan bien yeu cau (Architecture Advisor & Peer Reviewer) | - Phan tich yeu cau tu slide bai giang goc, chi ra cac cu phap da bi khai tu (deprecated).<br>- Phap hien cac diem mu ky thuat (Data Augmentation Leakage, tham so optimizer du thua).<br>- Thiet lap khung dac ta yeu cau va chien luoc viet prompt chuan xac. |
| **AI Coding Agent (IDE Integrated)** | Tro ly sinh ma nguon theo dac ta (Spec-driven Code Generator) | - Khoi tao va dinh dang cac tep tai lieu ky thuat (`docs/`).<br>- Sinh khung ma nguon (skeletons), Type Hints, va docstrings theo dung interface da duoc thiet ke.<br>- Ho tro viet cac doan script kiem thu tu dong (Unit & Smoke tests). |

Moi doan ma nguon hoac noi dung do AI sinh ra deu phai trai qua quy trinh kiem tra ky thuat nghiem ngat truoc khi duoc phep tich hop vao nhanh lam viec (`lab2`).

---

## 2. Main Workflow & Progress (Spec-Driven Development)

Du an khong thuc hien theo loi tiep can truyen thong (yeu cau AI viet toan bo code trong mot prompt duy nhat), ma tuan thu chat che quy trinh **Phat trien Dua tren Dac ta (Spec-Driven Development)** qua 5 giai doan co kiem soat:

```
[Giai doan 1: Phan tich De goc & Phat hien Diem mu]
                       |
                       v
[Giai doan 2: Thiet lap Bo Tai lieu Dac ta (docs/)]
  - project_description.md -> project_design.md -> development_workflow.md
                       |
                       v
[Giai doan 3: Tham do Phan cung Thuc te & Cau hinh Thich ung]
  - Kiem tra RTX 3050 (6GB VRAM), CPU i7-13620H (16 threads)
                       |
                       v
[Giai doan 4: Dieu phoi AI Sinh Code theo Chot chan (Milestones)]
  - M1: Data Pipeline -> M2: Model Factory -> M3: Trainer -> M4: Evaluation
  - Xac thuc ngay qua Smoke Run & Test logic tai moi chot chan
                       |
                       v
[Giai doan 5: Thuc nghiem Doc lap tren Phan cung & Phan tich Ket qua]
  - Thu thap Log TensorBoard, xuat bieu do 300 DPI, sinh vien tu phan tich
```

### Chi tiet cac giai doan:
- **Giai doan 1 (Decompilation & Problem Framing)**: Doc ky de bai tu slide thuc hanh. Phat hien slide su dung cu phap `pretrained=True` da cu, chua quy dinh tap du lieu cu the, chua co phuong an tach validation doc lap. Sinh vien dinh hinh pham vi: Tap CIFAR-10, 3 ho model (ResNet, VGG, DenseNet), 2 chien luoc (`freeze_all`, `unfreeze_last_block`).
- **Giai doan 2 (Spec Authoring)**: Xay dung day du bo tai lieu truoc khi viet code: `docs/project_description.md` (dac ta bai toan), `docs/project_design.md` (thiet ke kien truc module, luong du lieu, chu ky ham), va `docs/development_workflow.md` (quy trinh 7 buoc).
- **Giai doan 3 (Hardware Probing)**: Chay script kiem tra phan cung thuc te de lay thong so GPU/CPU, tu do rang buoc AI thiet lap tham so mac dinh toi uu cho 6GB VRAM va 16 threads CPU.
- **Giai doan 4 (Controlled Modular Generation)**: Yeu cau AI sinh ma nguon doc lap cho tung module theo dung Signature da cam ket o `project_design.md`. Chay kiem thu tai cho sau moi module.
- **Giai doan 5 (Independent Experimentation)**: Chay thuc nghiem thuc te tren may, quan sat duong cong hoi tu qua TensorBoard va danh gia ket qua thuc te ma khong phu thuoc vao du doan cua AI.

---

## 3. Student Contributions (Dong gop Cot loi cua Sinh vien)

Ranh gioi giua nang luc ky thuat cua sinh vien va su ho tro cua AI duoc phan dinh ro rang qua cac dong gop mang tinh quyet dinh kien truc sau:

1. **Dinh hinh Bai toan Thuc te tren CIFAR-10**:
   - De bai goc chi neu yeu cau tong quan ve viec thu nghiem tren pre-trained models. Sinh vien chu dong lua chon CIFAR-10, thiet lap phan chia nghiem ngat: 40,000 Train / 10,000 Validation / 10,000 Test de danh gia chuan muc khoa hoc.
2. **Loai bo Cu phap Loi thoi, Cap nhat Chuan PyTorch/Torchvision Hien dai**:
   - Phat hien tham so `pretrained=True` trong slide da bi khai tu tu Torchvision 0.13. Yeu cau AI phai chuyen doi 100% sang Weights API (`weights=ResNet18_Weights.DEFAULT`, `VGG16_BN_Weights.DEFAULT`, `DenseNet121_Weights.DEFAULT`).
3. **Thiet ke Co che Phong chong Ro ri Du lieu (Data Leakage Prevention)**:
   - AI co xu huong dung `torch.utils.data.random_split` truc tiep tren mot dataset da gan `train_transform` (chua `RandomCrop`, `RandomHorizontalFlip`).
   - Sinh vien da phat hien loi sai nay va yeu cau giai phap chuan: Khoi tao 2 doi tuong `CIFAR10(root, train=True)` rieng biet gan voi 2 bo transform khac nhau, su dung chi so ngau nhien co dinh qua `torch.randperm` voi `manual_seed(42)` roi bọc qua `torch.utils.data.Subset`.
4. **Quy dinh Kien truc Dynamic Model Factory**:
   - Yeu cau ro rang ve viec xu ly su bat dong bo trong cach dat ten tang phan loai cuoi giua cac ho mo hinh: `.fc` (ResNet), `.classifier[6]` (VGG), `.classifier` (DenseNet).
   - Bat buoc logic dong bang phai tach biet giua `freeze_all` va `unfreeze_last_block` (mo khoa `layer4` cua ResNet hoac `denseblock4` cua DenseNet).
5. **Toi uu hoa Tai nguyen va Bo loc Optimizer**:
   - Phat hien AI truyen toan bo `model.parameters()` vao Optimizer ngay ca khi cac tang da dong bang. Sinh vien yeu cau loc `filter(lambda p: p.requires_grad, model.parameters())` de tiet kiem trang thai bo nho AdamW/SGD.
6. **Kiem duyet Chat che va Danh gia Doc lap**:
   - Sinh vien truc tiep chay cac lenh kiem thu hinh hoc Tensor (Tensor Shape Verification), kiem tra do lech Loss khoi tao va phan tich ket qua tren ma tran nham lan.

---

## 4. Prompt Engineering & Proposals

Sinh vien da van dung cac ky thuat Prompt Engineering chuyen sau de kiem soat dau ra cua AI:

### 4.1. Cac Ky thuat Prompting da Su dung
- **Context Anchoring (Neo Ngu canh)**: Luon truyen duong dan hoac trich dan truc tiep cac nguyen tac tu `docs/project_description.md` va `docs/project_design.md` vao prompt de AI khong tu y sang tao kien truc moi.
- **Role Prompting**: Dinh vi ro rang vai tro cua AI trong tung giai doan (vi du: "Senior Machine Learning Engineer", "System Architect", "Quality Assurance Engineer").
- **Strict Constraint Enforcement**: Dat cac lanh cam nghiem ngat trong prompt:
  - Khong su dung emoji.
  - Khong overengineering (khong tao lop truu tuong du thua).
  - Bat buoc su dung Type Hints va docstring theo chuan PEP8.
  - Bat buoc loc `requires_grad=True` cho Optimizer.
- **Step-by-step Modular Generation**: Chia nho qua trinh viet code thanh tung file doc lap theo dung tien do Milestone thay vi sinh toan bo du an trong mot turn.

### 4.2. Khung Cau truc Prompt Mau (Prompt Template)

```
[Vai tro & Muc tieu]
Role: Senior Machine Learning Engineer
Task: Xay dung module <ten_file> tuan thu thiet ke tai docs/project_design.md.

[Ranh gioi & Rang buoc Ky thuat]
- Chi su dung Weights API hien dai tu torchvision.models.
- Bat buoc su dung Type Hints day du va xu ly ngoai le hop le.
- Tuyet doi khong dung emoji.
- Khong them bat ky thu vien nao nam ngoai requirements.txt.

[Tieu chi Nghiem thu (DoD)]
- Cung cap mot doan test script ngan de kiem chung module hoat dong doc lap qua dong lenh terminal.
```

---

## 5. Verification & Quality Control

Moi doan code hoac tai lieu do AI tao ra deu phai vuot qua 4 chot kiem soat ky thuat do sinh vien truc tiep thuc thi:

### 5.1. Kiem tra Tinh khong Ro ri cua Data Pipeline
- **Kiem chung**: Truy cap truc tiep vao `val_loader.dataset` va `test_loader.dataset` de kiem tra danh sach phep bien doi (`transforms`).
- **Tieu chuan dat**: Tap Validation va Test chi duoc chua cac phep bien doi xac dinh (`Resize(224)`, `ToTensor()`, `Normalize()`), hoan toan khong ton tai `RandomCrop` hay `RandomHorizontalFlip`.

### 5.2. Kiem tra So luong Tham so va requires_grad
- **Kiem chung**: Chay script tinh toan tham so tren tung kien truc:
  - ResNet-18 voi `freeze_all`: Chi co $512 \times 10 + 10 = 5,130$ tham so trainable (chiem $\approx 0.046\%$ tong tham so).
  - ResNet-18 voi `unfreeze_last_block`: Toan bo `layer4` va `fc` co $\approx 8.4$ trieu tham so trainable.
- **Tieu chuan dat**: So luong tham so trainable phai khop chinh xac tuyet doi voi phep tinh ly thuyet truoc khi cho phep chay huan luyen.

### 5.3. Kiem tra Che do Hoat dong cua Model (train/eval) va Gradient
- **Kiem chung**: Kiem tra ma nguon cua `trainer.py` va `evaluate.py`:
  - Trong ham `train_one_epoch`: Phai co `model.train()`.
  - Trong ham `validate` va `evaluate_test_set`: Phai co `model.eval()` va duoc bao boc trong khoi `with torch.no_grad():`.

### 5.4. Kiem tra Gia tri Loss Khoi tao (Initial Loss Sanity Check)
- **Kiem chung**: Khi mo hinh chua duoc huan luyen tren 10 lop CIFAR-10, voi phan phoi xac suat deu giua cac lop, gia tri Cross-Entropy Loss o batch dau tien phai xap xi:
  $$L_0 \approx -\ln\left(\frac{1}{C}\right) = -\ln(0.1) \approx 2.3025$$
- **Tieu chuan dat**: Neu Loss khoi tao nam ngoai khoang $[2.2, 2.5]$, pipeline lap tuc bi tam dung de ra soat lai phep tinh loss va nhan target.

---

## 6. Limitations & Manual Adjustments

Bang nhat ky ghi nhan cac sai lech cua AI trong qua trinh phat trien va cac bien phap sua loi truc tiep tu sinh vien:

| Module / Buoc | Van de / Sai sot tu phia AI (Issue Encountered) | Bien phap Xu ly & Hieu chinh cua Sinh vien (Manual Fix) | Ket qua Kiem chung (Verification Result) |
| :--- | :--- | :--- | :--- |
| **Phan tich De bai** | AI de xuat dung lai cu phap `pretrained=True` theo slide goc. | Yeu cau loai bo hoan toan va thay bang `weights=...Weights.DEFAULT`. | Mo hinh load khong con canh bao DeprecationWarning cua PyTorch. |
| **`src/dataset.py`** | AI su dung `random_split` tren mot Dataset co san Data Augmentation, lam tap Val bi nhiem phep xoay anh ngau nhien. | Khoi tao 2 thuc the Dataset rieng biet cho `train_transform` va `eval_transform`, chia tap thong qua chi so ngau nhien co dinh `torch.randperm`. | Tap Validation duoc kiem tra chi con phep Resize va Normalize tinh. |
| **`src/model_factory.py`** | AI chi thay the `.fc` ma khong tinh den cau truc `.classifier[6]` cua VGG va `.classifier` cua DenseNet. | Bo sung cau truc re nhanh tuong minh kiem tra kien truc mo hinh va thay the dung thuoc tinh phan loai. | Chay inspect thanh cong ca 4 kien truc ma khong bi loi AttributeError. |
| **`src/trainer.py`** | AI truyen tat ca `model.parameters()` vao Optimizer mac du da dong bang cac tang truoc do. | Bo sung bo loc `filter(lambda p: p.requires_grad, model.parameters())` khi khoi tao Optimizer. | Optimizer chi quan ly 5,130 tham so o che do `freeze_all`, giam tieu thu VRAM. |
| **`src/trainer.py`** | AI thieu co che don bo nho cache VRAM khi huan luyen lien tuc nhieu model. | Bo sung lenh `torch.cuda.empty_cache()` sau moi epoch va sau khi danh gia. | Khong xay ra hien tuong tran bo nho (CUDA OOM) tren card RTX 3050 6GB. |
