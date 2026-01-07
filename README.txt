====================================================================
        UNIVERSITY OF INFORMATION TECHNOLOGY (UIT)
====================================================================
               IMAGE PROCESSING AND APPLICATION
====================================================================


------------------------------------------------------------
THÀNH VIÊN NHÓM
------------------------------------------------------------

STT | MSSV     | HỌ VÀ TÊN                | CHỨC VỤ     | EMAIL
----+----------+--------------------------+-------------+-------------------------
1   | 22520982 | Phạm Hoàng Lê Nguyên     | Nhóm trưởng | 22520982@gm.uit.edu.vn
2   | 23520899 | Nguyễn Thế Luân          | Thành viên  | 23520899@gm.uit.edu.vn
3   | 23521193 | Đinh Hoàng Phúc          | Thành viên  | 23521193@gm.uit.edu.vn
4   | 23521704 | Trần Thị Cẩm Tú          | Thành viên  | 23521704@gm.uit.edu.vn


------------------------------------------------------------
GIỚI THIỆU MÔN HỌC
------------------------------------------------------------

Tên môn học : Xử lý ảnh và ứng dụng
Mã môn học  : CS406
Mã lớp     : CS406.Q12
Năm học    : Học kỳ 1 (2025 - 2026)
Giảng viên : TS. Cáp Phạm Đình Thăng


------------------------------------------------------------
GIỚI THIỆU ĐỒ ÁN
------------------------------------------------------------

Trong bối cảnh trí tuệ nhân tạo ngày càng được ứng dụng rộng rãi
trong lĩnh vực y tế, việc truy xuất và phân tích hình ảnh y khoa
dựa trên mô tả văn bản đóng vai trò quan trọng trong hỗ trợ chẩn đoán,
đào tạo và nghiên cứu lâm sàng.

Đồ án này tập trung xây dựng và đánh giá hệ thống truy xuất ảnh y khoa
dựa trên truy vấn văn bản (Text-to-Image Retrieval), sử dụng các mô hình
học sâu đa phương thức hiện đại. Hệ thống cho phép ánh xạ ảnh và văn bản
vào cùng một không gian đặc trưng (embedding space), từ đó thực hiện
truy xuất dựa trên độ tương đồng ngữ nghĩa.


------------------------------------------------------------
MỤC TIÊU VÀ PHẠM VI
------------------------------------------------------------

MỤC TIÊU:
- Xây dựng pipeline hoàn chỉnh cho bài toán truy xuất ảnh – văn bản
  trong miền y khoa.
- Áp dụng và so sánh nhiều mô hình đa phương thức khác nhau.
- Đánh giá hiệu năng các mô hình bằng các độ đo chuẩn trong bài toán
  retrieval.
- Thực hành tư duy tính toán thông qua việc phân rã bài toán, chuẩn hoá
  dữ liệu và thiết kế hệ thống.

PHẠM VI:
- Bài toán được giới hạn ở truy xuất ảnh từ văn bản (Text-to-Image
  Retrieval) với ánh xạ 1–1 giữa ảnh và mô tả.
- Không xét đến bài toán phân loại hay sinh văn bản.
- Dữ liệu sử dụng thuộc miền ảnh y khoa Tai–Mũi–Họng và ảnh y sinh học
  tổng quát.


------------------------------------------------------------
KIẾN TRÚC HỆ THỐNG
------------------------------------------------------------

Hệ thống được thiết kế theo kiến trúc dual-encoder, gồm hai bộ mã hoá
độc lập:

- Image Encoder:
  Trích xuất đặc trưng từ ảnh y khoa.

- Text Encoder:
  Trích xuất đặc trưng từ mô tả văn bản.

Hai embedding đầu ra được chuẩn hoá và huấn luyện sao cho các cặp
ảnh – văn bản tương ứng có độ tương đồng cao trong cùng một không gian
đặc trưng.

Pipeline tổng quát gồm các bước:
1. Chuẩn hoá và tiền xử lý dữ liệu.
2. Mã hoá ảnh và văn bản.
3. Huấn luyện bằng contrastive learning.
4. Đánh giá bằng các độ đo retrieval.

Ngoài ra, sau khi lấy được danh sách top-10 kết quả cho mỗi truy vấn,
các hình ảnh này được đưa qua các mô hình LLM kết hợp với hệ cơ sở tri
thức để thực hiện bước rerank.


------------------------------------------------------------
CÁC MÔ HÌNH SỬ DỤNG
------------------------------------------------------------

Hệ thống triển khai và so sánh 5 mô hình:

- NanoCLIP:
  Mô hình nhẹ sử dụng DINOv2 và MiniLM, phù hợp cho huấn luyện nhanh.

- ALIGN:
  Mô hình dual-encoder cổ điển với ResNet và BERT.

- BLIP (Retrieval-style):
  Mô hình vision–language hiện đại với encoder tách biệt.

- MedCLIP:
  Mô hình chính thức cho ảnh y khoa, sử dụng BioClinicalBERT.

- BioMedCLIP:
  Mô hình phát hành qua OpenCLIP, tiền huấn luyện trên dữ liệu y sinh học
  quy mô lớn.

Tất cả các mô hình được huấn luyện và đánh giá trên cùng một tập dữ liệu
để đảm bảo tính công bằng khi so sánh.


------------------------------------------------------------
DỮ LIỆU SỬ DỤNG
------------------------------------------------------------

Dữ liệu được tổ chức thống nhất cho toàn bộ hệ thống, bao gồm 3 folder 
train/val/test được chia theo tỷ lệ 8/1/1, trong mỗi folder bao gồm:
- Tập ảnh y khoa đã được chuẩn hoá.
- Tập mô tả văn bản tương ứng.

Chúng em sử dụng 2 bộ dataset là ENTREP2025 và OCASD. 
Link dataset: https://drive.google.com/drive/folders/1hjLXlv1FiBkp7HkjoeTatdrLSMLvTWjc?usp=sharing


------------------------------------------------------------
PHƯƠNG PHÁP HUẤN LUYỆN
------------------------------------------------------------

Các mô hình được huấn luyện theo phương pháp Contrastive Learning.

Mục tiêu của hàm mất mát:
- Tối đa hoá độ tương đồng giữa ảnh và văn bản đúng cặp.
- Tối thiểu hoá độ tương đồng giữa các cặp không tương ứng.

Quá trình huấn luyện sử dụng:
- Batch training.
- Chuẩn hoá embedding (L2 normalization).
- Tối ưu hoá bằng AdamW.


------------------------------------------------------------
HƯỚNG DẪN CÀI ĐẶT VÀ THỰC THI
------------------------------------------------------------

YÊU CẦU CƠ BẢN:
- Python 3.8+ (khuyến nghị 3.9 / 3.10)
- CUDA 11.x (nếu sử dụng GPU)
- Git, pip

CÀI ĐẶT NHANH:

1. Tạo và kích hoạt môi trường ảo:

   python -m venv .venv

   Trên Windows:
   .venv\Scripts\activate

   Trên macOS / Linux:
   source .venv/bin/activate

2. Cài đặt các thư viện phụ thuộc:

   pip install -r requirements.txt

   Nếu không có file requirements.txt:
   pip install torch torchvision pytorch-lightning transformers
   scikit-learn ftfy tqdm

3. Cài Jupyter để chạy notebook (tuỳ chọn):

   pip install jupyterlab
   jupyter lab


CẤU TRÚC DỮ LIỆU VÀ CẤU HÌNH:
- Dữ liệu huấn luyện và đánh giá được đặt trong thư mục data/ hoặc
  datasets/ theo yêu cầu của từng file cấu hình.
- Các tham số chạy (đường dẫn dữ liệu, batch size, epochs, learning rate,
  device, ...) được cấu hình trong thư mục configs/*.yaml.


CHẠY HUẤN LUYỆN:

Ví dụ chung (thay <model> bằng nanoclip, align, blip, medclip, biomedclip):

   python scripts/train_<model>.py --config configs/<model>.yaml


CHẠY ĐÁNH GIÁ:

Ví dụ chung:

   python scripts/eval_<model>.py --config configs/<model>.yaml

Kết quả đánh giá sẽ được in ra terminal và/hoặc lưu trong thư mục
evaluation/ tuỳ theo từng script.


CHẠY NOTEBOOK:

- Mở các file notebooks/*.ipynb bằng JupyterLab.
- Chạy từng cell để xem demo hoặc các phân tích trực quan.


CHẠY DEMO:

Demo gồm hai phần chính: **Giao diện (Streamlit)** và **Backend (FastAPI)**. Dưới đây là các bước chuẩn bị và lệnh chạy cụ thể.

1. Chuẩn bị mô hình và dữ liệu

- Đảm bảo thư mục ảnh demo có trong `images/` hoặc `demo/images/`.
- Các file cần có:
  - Model checkpoint: `model/nanoclip.ckpt` hoặc `model/nanoclip_v2.ckpt` (dùng cho `demo/app.py`).
  - FAISS index: `faiss_index_raw.bin`, `faiss_index_processed.bin` hoặc `faiss_index.bin` (tuỳ script).
  - Danh sách đường dẫn ảnh: `image_paths.pkl` (cho Streamlit) hoặc `image_paths.npy` (cho backend).
- (Tùy chọn) Tải Qwen2-VL (nếu sử dụng rerank local):

  ```
  python demo/download_qwen.py
  ```

2. Chạy Backend (FastAPI)

- Cài phụ thuộc nếu cần:

  ```
  pip install fastapi uvicorn
  ```

- Chạy server backend (từ thư mục gốc project):

  ```
  uvicorn demo.backend:app --host 0.0.0.0 --port 8000 --reload
  ```

- Endpoint truy vấn:
  - POST `/search` với payload JSON, ví dụ:

  ```json
  { "text": "viêm amidan", "rerank": true, "top_k": 5 }
  ```

3. Chạy giao diện (Streamlit)

- Cài Streamlit nếu chưa có:

  ```
  pip install streamlit
  ```

- Chạy app:

  ```
  streamlit run demo/app.py
  ```

- Mở trình duyệt tại địa chỉ console cung cấp (mặc định http://localhost:8501). Trên sidebar có thể chọn phiên bản NanoCLIP và bật/tắt **LLM rerank**.

4. Lưu ý vận hành

- Để tận dụng GPU, đảm bảo CUDA và PyTorch được cài phù hợp.
- Nếu gặp lỗi thiếu file index/checkpoint, kiểm tra các đường dẫn như `model/` và file `image_paths.pkl` / `image_paths.npy`.
- Nếu muốn dùng LLM rerank từ dịch vụ ngoài, chỉnh `LLM_API_URL` trong `demo/app.py` thành endpoint phù hợp hoặc chạy Qwen2-VL local và trỏ tới nó.

Ví dụ cURL để thử backend:

```bash
curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{"text":"viêm amidan","rerank":true,"top_k":5}'
```


LƯU Ý:
- Để tái lập kết quả, cần đảm bảo cùng phiên bản thư viện,
  cùng seed ngẫu nhiên và cấu hình GPU tương tự.
