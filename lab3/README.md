# Lab 3: Sentiment Analysis & Fine-Tuning with Transformers

> **Trạng thái**: Thư mục chờ tích hợp (Pending Merge). Mã nguồn hiện đang lưu trữ tại nhánh `origin/LAP-3`.

## 1. Giới thiệu Bài thực hành
Bài thực hành số 3 tập trung vào xử lý ngôn ngữ tự nhiên (NLP): phân loại sắc thái cảm xúc (Sentiment Analysis) và tinh chỉnh mô hình Transformer (Fine-tuning Pre-trained Transformers).

## 2. Hướng dẫn Hợp nhất Mã nguồn vào `lab3/` (Dành cho người phụ trách Lab 3)
Khi bạn sẵn sàng đưa mã nguồn từ nhánh `origin/LAP-3` vào thư mục này, thực hiện các lệnh sau:

```bash
# 1. Chuyển sang nhánh lab3 từ origin/LAP-3
git checkout -b lab3 origin/LAP-3

# 2. Gom toàn bộ file vào thư mục lab3/
mkdir -p lab3
git mv Practice3.ipynb README.md exercise1_sentiment.py exercise2_finetune.py metrics_utils.py predict.py requirements.txt tests lab3/
git mv .gitignore lab3/.gitignore

# 3. Commit và đẩy lên nhánh lab3
git commit -m "refactor(lab3): move files into lab3/ directory"
git push origin lab3

# 4. Hợp nhất vào main
git checkout main
git merge lab3 -m "merge(lab3): integrate lab 3 into main"
git push origin main
```
