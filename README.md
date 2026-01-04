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
| 1   | 22520899 | Phạm Hoàng Lê Nguyên      | Nhóm trưởng | 22520899@gm.uit.edu.vn   |
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

Dữ liệu được tổ chức thống nhất cho toàn bộ hệ thống, bao gồm:
- Tập ảnh y khoa đã được chuẩn hoá.
- Tập mô tả văn bản tương ứng.
- Các tập chia train / validation / test theo chỉ số.

Cấu trúc dữ liệu đảm bảo ánh xạ 1–1 giữa ảnh và văn bản,
phù hợp với giả định trong quá trình đánh giá retrieval.


---

# PHƯƠNG PHÁP HUẤN LUYỆN

Các mô hình được huấn luyện theo phương pháp **Contrastive Learning**.
Mục tiêu của hàm mất mát là:
- Tối đa hoá độ tương đồng giữa ảnh và văn bản đúng cặp.
- Tối thiểu hoá độ tương đồng giữa các cặp không tương ứng.

Quá trình huấn luyện sử dụng:
- Batch training.
- Chuẩn hoá embedding (L2 normalization).
- Tối ưu hoá bằng AdamW.

---

# ĐÁNH GIÁ VÀ ĐỘ ĐO

Hiệu năng hệ thống được đánh giá bằng các độ đo chuẩn trong bài toán truy xuất:

- **Recall@K:** tỷ lệ truy xuất đúng trong top-K kết quả.
- **mAP (mean Average Precision):** phản ánh thứ hạng trung bình của kết quả đúng.
- **nDCG@K:** đánh giá chất lượng xếp hạng có xét đến vị trí kết quả đúng.

Đánh giá được thực hiện cho cả hai chiều:
- Text → Image
- Image → Text

---

# KẾT QUẢ THỰC NGHIỆM

Kết quả thực nghiệm cho thấy:
- Các mô hình chuyên biệt cho y khoa (MedCLIP, BioMedCLIP) đạt hiệu năng cao hơn
  trên dữ liệu y sinh học.
- NanoCLIP có ưu điểm về tốc độ và chi phí tính toán.
- BioMedCLIP thể hiện khả năng tổng quát hoá tốt nhờ tiền huấn luyện quy mô lớn.

Chi tiết kết quả được trình bày trong các bảng đánh giá và biểu đồ so sánh.

---

# KẾT LUẬN

Đồ án đã xây dựng thành công một hệ thống truy xuất ảnh y khoa dựa trên văn bản
với kiến trúc rõ ràng, pipeline thống nhất và khả năng mở rộng cho nhiều mô hình khác nhau.
Kết quả thực nghiệm cho thấy tầm quan trọng của việc lựa chọn mô hình phù hợp với miền dữ liệu.

Trong tương lai, hệ thống có thể được mở rộng theo các hướng:
- Hỗ trợ nhiều mô tả cho một ảnh.
- Kết hợp cross-encoder để reranking.
- Triển khai giao diện truy vấn trực quan.

---

# HƯỚNG DẪN CÀI ĐẶT & THỰC THI ✅

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

3. (Tuỳ chọn) Cài Jupyter để chạy notebooks:

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

> **Lưu ý:** Để tái lập kết quả, hãy đảm bảo sử dụng cùng phiên bản thư viện, seed ngẫu nhiên và cấu hình GPU tương tự như trong file cấu hình.

---

# TÀI LIỆU THAM KHẢO

[1] Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*.  
[2] Li et al., *BLIP: Bootstrapping Language-Image Pre-training*.  
[3] Wang et al., *MedCLIP: Contrastive Learning from Medical Images and Text*.  
[4] OpenCLIP & BioMedCLIP Documentation.

---

