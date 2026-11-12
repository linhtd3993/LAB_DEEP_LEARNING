# Lab 1: Image Classification on Fashion-MNIST

> **Trạng thái**: Thư mục chờ tích hợp (Pending Merge). Mã nguồn hiện đang lưu trữ tại nhánh `origin/lab1`.

## 1. Giới thiệu Bài thực hành
Bài thực hành số 1 tập trung vào xây dựng và đánh giá mạng nơ-ron truyền thẳng (Multi-Layer Perceptron - MLP) trên bộ dữ liệu Fashion-MNIST sử dụng PyTorch.

## 2. Hướng dẫn Hợp nhất Mã nguồn vào `lab1/` (Dành cho người phụ trách Lab 1)
Khi bạn sẵn sàng đưa mã nguồn từ nhánh `origin/lab1` vào thư mục này, thực hiện các lệnh sau:

```bash
# 1. Chuyển sang nhánh lab1
git checkout -b lab1 origin/lab1

# 2. Gom toàn bộ file vào thư mục lab1/
mkdir -p lab1
git mv fashion_mnist_model.pth lab1_fashion_mnist.py loss_graph.png predictions_display.png slides_lab1.md README.md lab1/

# 3. Commit và đẩy lên nhánh lab1
git commit -m "refactor(lab1): move files into lab1/ directory"
git push origin lab1

# 4. Hợp nhất vào main
git checkout main
git merge lab1 -m "merge(lab1): integrate lab 1 into main"
git push origin main
```
