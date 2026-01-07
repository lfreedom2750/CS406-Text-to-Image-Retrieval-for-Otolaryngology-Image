<!-- Banner -->
<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

<h1 align="center"><b>IMAGE PROCESSING AND APPLICATION</b></h1>

---

# THÀNH VIÊN NHÓM

| STT | MSSV     | Họ và Tên                 | Chức vụ     | Email                    |
|-----|----------|---------------------------|-------------|--------------------------|
| 1   | 22520982 | Phạm Hoàng Lê Nguyên      | Nhóm trưởng | 22520982@gm.uit.edu.vn   |
| 2   | 23520899 | Nguyễn Thế Luân           | Thành viên  | 23520899@gm.uit.edu.vn   |
| 3   | 23521193 | Đinh Hoàng Phúc           | Thành viên  | 23521193@gm.uit.edu.vn   |
| 4   | 23521704 | Trần Thị Cẩm Tú           | Thành viên  | 23521704@gm.uit.edu.vn   |

---


# GIỚI THIỆU MÔN HỌC

- **Tên môn học:** Xử lý ảnh và ứng dụng
- **Mã môn học:** CS406
- **Mã lớp:** CS406.Q12
- **Năm học:** Học kỳ 1 (2025 - 2026)
- **Giảng viên:** TS. Cáp Phạm Đình Thăng

---

# GIỚI THIỆU ĐỒ ÁN

Trong bối cảnh trí tuệ nhân tạo ngày càng được ứng dụng rộng rãi trong lĩnh vực y tế,
việc truy xuất và phân tích hình ảnh y khoa dựa trên mô tả văn bản đóng vai trò quan trọng
trong hỗ trợ chẩn đoán, đào tạo và nghiên cứu lâm sàng.

Đồ án này tập trung xây dựng và đánh giá hệ thống **truy xuất ảnh y khoa dựa trên truy vấn văn bản**
(*Text-to-Image Retrieval*), sử dụng các mô hình học sâu đa phương thức hiện đại.
Hệ thống cho phép ánh xạ ảnh và văn bản vào cùng một không gian đặc trưng (embedding space),
từ đó thực hiện truy xuất dựa trên độ tương đồng ngữ nghĩa.

---

# MỤC TIÊU VÀ PHẠM VI

## Mục tiêu
- Xây dựng pipeline hoàn chỉnh cho bài toán truy xuất ảnh–văn bản trong miền y khoa.
- Áp dụng và so sánh nhiều mô hình đa phương thức khác nhau.
- Đánh giá hiệu năng các mô hình bằng các độ đo chuẩn trong bài toán retrieval.
- Thực hành tư duy tính toán thông qua việc phân rã bài toán, chuẩn hóa dữ liệu và thiết kế hệ thống.

## Phạm vi
- Bài toán được giới hạn ở **Text-to-Image Retrieval** với ánh xạ 1–1 giữa ảnh và mô tả.
- Không xét đến bài toán phân loại hay sinh văn bản.
- Dữ liệu sử dụng thuộc miền ảnh y khoa Tai–Mũi–Họng và ảnh y sinh học tổng quát.

---

# KIẾN TRÚC HỆ THỐNG

Hệ thống được thiết kế theo kiến trúc **dual-encoder**, gồm hai bộ mã hoá độc lập:

- **Image Encoder:** trích xuất đặc trưng từ ảnh y khoa.
- **Text Encoder:** trích xuất đặc trưng từ mô tả văn bản.

Hai embedding đầu ra được chuẩn hoá và huấn luyện sao cho các cặp ảnh–văn bản tương ứng
có độ tương đồng cao trong không gian đặc trưng chung.

Pipeline tổng quát gồm các bước:
1. Chuẩn hoá và tiền xử lý dữ liệu.
2. Mã hoá ảnh và văn bản.
3. Huấn luyện bằng contrastive learning.
4. Đánh giá bằng các độ đo retrieval.

Ngoài ra, sau khi lấy được list top 10 ứng với mỗi câu truy vấn tương ứng, các hình ảnh này được đưa qua các mô hình llm kết hợp cùng các hệ cơ sở tri thức để rerank lại top các hình ảnh.
---

# CÁC MÔ HÌNH SỬ DỤNG

Hệ thống triển khai và so sánh 5 mô hình:

- **NanoCLIP:** mô hình nhẹ sử dụng DINOv2 và MiniLM, phù hợp huấn luyện nhanh.
- **ALIGN:** mô hình dual-encoder cổ điển với ResNet và BERT.
- **BLIP (Retrieval-style):** mô hình vision–language hiện đại với encoder tách biệt.
- **MedCLIP:** mô hình chính thức cho ảnh y khoa, sử dụng BioClinicalBERT.
- **BioMedCLIP:** mô hình phát hành qua OpenCLIP, tiền huấn luyện trên dữ liệu y sinh học quy mô lớn.

Tất cả các mô hình được huấn luyện và đánh giá trên cùng một tập dữ liệu
để đảm bảo tính công bằng khi so sánh.

---

# DỮ LIỆU SỬ DỤNG

Dữ liệu được tổ chức thống nhất cho toàn bộ hệ thống, bao gồm 3 folder 
train/val/test được chia theo tỷ lệ 8/1/1, trong mỗi folder bao gồm:
- Tập ảnh y khoa đã được chuẩn hoá.
- Tập mô tả văn bản tương ứng.

Chúng em sử dụng 2 bộ dataset là ENTREP2025 và OCASD. 
Link dataset: https://drive.google.com/drive/folders/1hjLXlv1FiBkp7HkjoeTatdrLSMLvTWjc?usp=sharing

---

# HƯỚNG DẪN CÀI ĐẶT & THỰC THI 

## Yêu cầu cơ bản
- Python **3.8+** (khuyến nghị 3.9/3.10)
- CUDA **11.x** (nếu dùng GPU)
- Git, pip

## Cài đặt nhanh
1. Tạo và kích hoạt môi trường ảo:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Cài phụ thuộc:

```bash
pip install -r requirements.txt
# Nếu không có file requirements.txt, cài các gói chính:
pip install torch torchvision pytorch-lightning transformers scikit-learn ftfy tqdm
```

3. Cài Jupyter để chạy notebooks, có thể chạy trên Colab hoặc Kaggle (Các file ipynb nằm trong folder notebooks):
```bash
pip install jupyterlab
jupyter lab
```

## Cấu trúc dữ liệu và cấu hình
- Đặt dữ liệu huấn luyện/đánh giá vào thư mục `data/` hoặc `datasets/` theo yêu cầu của từng config.
- Các tham số chạy (đường dẫn dữ liệu, batch size, epochs, learning rate, device, ...) được cấu hình trong `configs/*.yaml` — chỉnh file tương ứng trước khi chạy.

## Chạy huấn luyện
- Ví dụ chung (thay `<model>` bằng `nanoclip`, `align`, `blip`, `medclip`, `biomedclip`):

```bash
python scripts/train_<model>.py --config configs/<model>.yaml
```

- Kiểm tra các tuỳ chọn cụ thể trong file script nếu cần (ví dụ: sử dụng `--gpus`, `--num_workers`, ...).

## Chạy đánh giá
- Ví dụ chung:

```bash
python scripts/eval_<model>.py --config configs/<model>.yaml
```

- Kết quả đánh giá sẽ được in ra terminal và/hoặc lưu vào thư mục `evaluation/` tuỳ script.

## Chạy notebook
- Mở `notebooks/*.ipynb` bằng JupyterLab và chạy từng ô (cell) để xem demo hoặc các phân tích trực quan.

## CHẠY DEMO

Demo gồm hai phần chính: **Giao diện (Streamlit)** và **Backend (FastAPI)**. Dưới đây là các bước chuẩn bị và lệnh chạy cụ thể.

### 1) Chuẩn bị mô hình và dữ liệu

- Đảm bảo thư mục ảnh demo có trong `images/` hoặc `demo/images/`.
- Các file cần có:
  - Model checkpoint: `model/nanoclip.ckpt` hoặc `model/nanoclip_v2.ckpt` (dùng cho `demo/app.py`).
  - FAISS index: `faiss_index_raw.bin`, `faiss_index_processed.bin` hoặc `faiss_index.bin` (tuỳ script).
  - Danh sách đường dẫn ảnh: `image_paths.pkl` (cho Streamlit) hoặc `image_paths.npy` (cho backend).
- (Tùy chọn) Tải Qwen2-VL (nếu sử dụng rerank local):

```bash
python demo/download_qwen.py
```

### 2) Chạy Backend (FastAPI)

- Cài phụ thuộc nếu cần:

```bash
pip install fastapi uvicorn
```

- Chạy server backend (từ thư mục gốc project):

```bash
uvicorn demo.backend:app --host 0.0.0.0 --port 8000 --reload
```

- Endpoint truy vấn:
  - POST `/search` với payload JSON, ví dụ:

```json
{ "text": "viêm amidan", "rerank": true, "top_k": 5 }
```

### 3) Chạy giao diện (Streamlit)

- Cài Streamlit nếu chưa có:

```bash
pip install streamlit
```

- Chạy app:

```bash
streamlit run demo/app.py
```

- Mở trình duyệt tại địa chỉ console cung cấp (mặc định http://localhost:8501). Trên sidebar có thể chọn phiên bản NanoCLIP và bật/tắt **LLM rerank**.

### 4) Lưu ý vận hành

- Để tận dụng GPU, đảm bảo CUDA và PyTorch được cài phù hợp.
- Nếu gặp lỗi thiếu file index/checkpoint, kiểm tra các đường dẫn như `model/` và file `image_paths.pkl` / `image_paths.npy`.
- Nếu muốn dùng LLM rerank từ dịch vụ ngoài, chỉnh `LLM_API_URL` trong `demo/app.py` thành endpoint phù hợp hoặc chạy Qwen2-VL local và trỏ tới nó.

Ví dụ cURL để thử backend:

```bash
curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{"text":"viêm amidan","rerank":true,"top_k":5}'
```

> **Lưu ý:** Để tái lập kết quả, hãy đảm bảo sử dụng cùng phiên bản thư viện, seed ngẫu nhiên và cấu hình GPU tương tự như trong file cấu hình.

---

