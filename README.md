# fuzzy-mlp-demo

Demo trực quan cho đề tài **MLP + Fuzzy Logic trong phân loại ảnh phân cấp trên CIFAR-100**.

Mô hình có hai đầu ra: một đoán **nhãn con** (100 lớp, vd `apple`), một đoán **nhãn cha** (20 nhóm, vd `fruit_and_vegetables`). Hai đầu này có thể **mâu thuẫn nhau** — app cho thấy chuyện đó xảy ra khi nào và ba cách xử lý khác nhau cho kết quả ra sao.

---

## 🚀 Bắt đầu nhanh (5 bước)

Dành cho người chưa biết gì về dự án. Mỗi bước có phần giải thích chi tiết ở các mục bên dưới nếu cần.

**Bước 1 — Cài Python + thư viện** (cần Python 3.14, không cần GPU)

```bash
python -m venv .venv
.venv\Scripts\activate                          # Windows — macOS/Linux: source .venv/bin/activate
pip install torch==2.13.0+cpu torchvision==0.28.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**Bước 2 — Chuẩn bị dữ liệu**: chép 3 thứ vào project này — checkpoint (`.pt`), ảnh mẫu, kết quả lưới. Nếu đang chạy chung máy với notebook, một lệnh là đủ:

```bash
S=../../train_model_colab/drive_FuzzyMLP
mkdir -p results
cp $S/checkpoints/ckpt_lam0_seed0.pt $S/checkpoints/ckpt_lam2_seed0.pt checkpoints/
cp $S/samples/*.png $S/samples/index.json assets/samples/
cp $S/results/results.csv $S/results/bc_indices.json results/
```

Khác máy? Xem mục *Chuẩn bị dữ liệu* bên dưới — có hướng dẫn tải từ Google Drive.

**Bước 3 — Kiểm tra**

```bash
python scripts/smoke_test.py
```

Phải thấy `✅ PASS — 12/12 kiểm tra xanh`. Nếu FAIL, đọc thông điệp lỗi — nó sẽ nói rõ thiếu file gì.

**Bước 4 — Chạy app**

```bash
streamlit run app.py
```

**Bước 5 — Mở trình duyệt** ở địa chỉ Streamlit vừa in ra (mặc định `http://localhost:8501`). Thử lần lượt 3 tab: **Một ảnh** → **So sánh B0 vs M** → **Tổng quan**.

---

## Cài đặt (chi tiết)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

pip install torch==2.13.0+cpu torchvision==0.28.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Yêu cầu **Python 3.14** (khớp môi trường huấn luyện). Không cần GPU.

---

## Chuẩn bị dữ liệu

App cần hai thứ, đều do notebook `train_model_colab/MLP_FuzzyLogic_CIFAR10.ipynb` sinh ra:

### 1. Checkpoint → `checkpoints/`

Chép từ `train_model_colab/drive_FuzzyMLP/checkpoints/` (hoặc tải từ Google Drive nếu làm việc khác máy). Tối thiểu cần **một mốc B0 và một mốc M**:

| File                       | Là gì                                           |
| -------------------------- | ----------------------------------------------- |
| `ckpt_lam0_seed0.pt`       | **B0** — MLP thuần, không có ràng buộc logic    |
| `ckpt_lam2_seed0.pt`       | **M** — huấn luyện với fuzzy logic loss ở λ\*=2 |
| `ckpt_lam0_seed0_pilot.pt` | B0 bản chạy thử 15 epoch                        |
| `ckpt_lam1_seed0_pilot.pt` | M bản chạy thử 15 epoch                         |

Mỗi file ~7 MB. Chép thêm checkpoint khác cũng được — app tự quét và tự sinh nhãn hiển thị từ metadata bên trong file, không dựa vào tên file.

### 2. Ảnh mẫu → `assets/samples/`

Chép toàn bộ `train_model_colab/drive_FuzzyMLP/samples/` (20 ảnh PNG + `index.json`).

`index.json` giữ **chỉ số của từng ảnh trong tập test** — cần cho tab so sánh ở Step 2.4 để tô màu những mẫu mà M sửa được / phá hỏng.

### 3. Kết quả lưới → `results/`

Tab **Tổng quan** (Step 2.5) đọc `results/results.csv` — bảng kết quả đầy đủ của lưới λ×seed, do PHẦN 8 notebook sinh ra. Tab **So sánh** đọc thêm `results/bc_indices.json` (không bắt buộc — thiếu thì tab So sánh vẫn chạy, chỉ mất phần "bối cảnh trên toàn tập test").

| File | Là gì | Bắt buộc? |
|---|---|---|
| `results/results.csv` | Kết quả lưới: acc/consistency mỗi (λ, seed, mode, split) | Có, cho tab Tổng quan |
| `results/bc_indices.json` | Chỉ số ảnh ở ô b/ô c của flip matrix (PHẦN 9) | Không |

**Không commit hai file này** (xem `.gitignore`) — chúng là dữ liệu sinh ra, không phải mã nguồn; tải/chép lại mỗi khi cần.

### Chép nhanh (khi làm chung máy với notebook)

```bash
cd ui_python/fuzzy-mlp-demo
S=../../train_model_colab/drive_FuzzyMLP
mkdir -p results
cp $S/checkpoints/ckpt_lam0_seed0.pt $S/checkpoints/ckpt_lam2_seed0.pt checkpoints/
cp $S/samples/*.png $S/samples/index.json assets/samples/
cp $S/results/results.csv $S/results/bc_indices.json results/
```

### Tải từ Google Drive (khi làm việc khác máy với notebook)

1. Mở thư mục `FuzzyMLP/` trên Drive (đường link do A chia sẻ).
2. Tải `checkpoints/` → chép các file `.pt` cần dùng vào `checkpoints/` của project này.
3. Tải `samples/` → chép toàn bộ (20 PNG + `index.json`) vào `assets/samples/`.
4. Tải `results/results.csv` và `results/bc_indices.json` → chép vào `results/` của project này.

---

## Kiểm tra trước khi chạy

```bash
python scripts/smoke_test.py
```

Phải thấy `✅ PASS — 12/12 kiểm tra xanh`. Nếu FAIL, thông điệp lỗi sẽ nói rõ thiếu gì; thêm `-v` để xem traceback đầy đủ.

Smoke test kiểm tra cả **ngữ nghĩa** chứ không chỉ "chạy không lỗi" — ví dụ: chế độ `marginal` có đúng bằng tổng xác suất các nhãn con trong nhóm không, chế độ `hard` có thật sự nhất quán 100 % không, và checkpoint lệch phiên bản kiến trúc có bị **từ chối** thay vì nạp im lặng không.

---

## Chạy app

```bash
streamlit run app.py
```

Mở trình duyệt ở địa chỉ Streamlit in ra (mặc định http://localhost:8501).

---

## Ba chế độ suy luận

Đây là trọng tâm của demo. Cùng một mô hình, ba cách suy ra nhãn cha:

| Chế độ       | Cách làm                             | Tính nhất quán                |
| ------------ | ------------------------------------ | ----------------------------- |
| **raw**      | Lấy thẳng argmax của đầu ra 2        | Có thể mâu thuẫn với đầu ra 1 |
| **hard**     | Bỏ đầu ra 2, tra bảng từ nhãn con    | **100 % theo cách xây dựng**  |
| **marginal** | Cộng xác suất các nhãn con cùng nhóm | Nhất quán "mềm" tuyệt đối     |

---

## Độ bền — các điểm đã kiểm tra và vá (Step 3.3)

App được thiết kế để **không bao giờ hiện màn hình lỗi đỏ của Streamlit** khi trình diễn — mọi lỗi có thể lường trước đều thành thông điệp tiếng Việt kèm hướng xử lý. Mỗi dòng dưới đây đã được **tái hiện thật và kiểm chứng bằng test**, không phải liệt kê suy đoán:

| Tình huống | Đã kiểm chứng bằng | Xử lý |
|---|---|---|
| Thiếu checkpoint hoàn toàn | thư mục rỗng | Báo lỗi + dừng có kiểm soát, trỏ tới mục *Chuẩn bị dữ liệu* |
| File checkpoint 0 byte / tải dở | file rỗng thật | `CheckpointError` rõ nguyên nhân |
| File checkpoint là dữ liệu rác | file text giả `.pt` | `CheckpointError` rõ nguyên nhân |
| Checkpoint lệch `ARCH_VERSION` | sửa tay arch_version | Từ chối nạp, không dùng nhầm trọng số |
| **Metadata checkpoint méo** (vd `fine_classes` bị cắt ngắn so với `config.n_fine`) | cắt tay `fine_classes` còn 5/100 | `CheckpointError` — trước đó gây `IndexError` không kiểm soát ở `predict()`, đã vá tại `load_ckpt()` |
| **Ảnh (gallery hoặc upload) bị cắt cụt/hỏng** | ghi đè thật 1 ảnh mẫu bằng dữ liệu cắt cụt, chạy toàn bộ app | Tab **Một ảnh**: báo lỗi thân thiện, không crash. Tab **So sánh**: tự bỏ qua ảnh hỏng, cảnh báo số ảnh bị bỏ qua, 19 ảnh còn lại vẫn hoạt động |
| Ảnh RGBA / xám / bảng màu (P) | `smoke_test.py` kiểm tra cả 3 mode | Tự chuyển sang RGB trước khi xử lý |
| `results.csv` thiếu hoặc thiếu cột | — | Tab Tổng quan báo lỗi rõ tên cột thiếu, không crash |
| Máy không có GPU | môi trường phát triển vốn không có GPU khả dụng | App CPU-only theo thiết kế (không có lệnh `.cuda()` nào trong code) |
| Phiên bản Python/torch lệch máy huấn luyện | cài trong virtualenv sạch, tách biệt hoàn toàn khỏi máy A | Xem kết quả kiểm thử bên dưới |

### Đã thử trên virtualenv sạch

Test thật (không suy đoán): tạo virtualenv mới hoàn toàn tách biệt, cài đúng `requirements.txt`, chạy `smoke_test.py` — mô phỏng đúng tình huống "máy B tải project về lần đầu".

```
$ python -m venv .venv_clean && .venv_clean\Scripts\activate
$ pip install torch==2.13.0+cpu torchvision==0.28.0+cpu --index-url https://download.pytorch.org/whl/cpu
$ pip install -r requirements.txt
$ python scripts/smoke_test.py
```

**Kết quả (20/7, virtualenv hoàn toàn tách biệt, không set biến môi trường thủ công):**

```
✅ PASS — 12/12 kiểm tra xanh
```

Lần chạy đầu tiên trong venv sạch phát hiện một lỗi thật không thấy khi phát triển: `UnicodeEncodeError` khi in ✅/❌ trên console Windows mặc định (mã hoá `cp1252`, không phải UTF-8). Đã vá bằng cách ép `sys.stdout`/`sys.stderr` sang UTF-8 ngay đầu `scripts/smoke_test.py` — không bắt người dùng phải tự set `PYTHONIOENCODING`.

---

## ⚠️ Lưu ý khi trình diễn

**Mô hình chỉ đạt ~25 % độ chính xác nhãn con.** Đó là giới hạn của MLP thuần trên CIFAR-100 (100 lớp) — không phải lỗi cài đặt.

Vì vậy khi demo:

- **Luôn mở màn bằng gallery ảnh mẫu** (đúng phân bố dữ liệu huấn luyện).
- Ảnh tải từ Internet sẽ sai rất thường xuyên — ảnh ngoài phân bố, độ phân giải khác, và bị ép về 32×32.
- Điểm mạnh của demo **không phải độ chính xác** mà là **minh hoạ cơ chế**: khi nào hai tầng mâu thuẫn, và ba cách xử lý khác nhau ở đâu.

---

## Cấu trúc thư mục

```
fuzzy-mlp-demo/
├── app.py                  # entry Streamlit
├── core/
│   ├── model_def.py        # BẢN SAO NGUYÊN VĂN kiến trúc từ notebook PHẦN 2
│   ├── inference.py        # load_ckpt / preprocess / predict
│   └── registry.py         # quét checkpoints/, sinh nhãn hiển thị
├── ui/                     # 3 tab: single_view / compare_view / dashboard_view
├── scripts/smoke_test.py   # 12 kiểm tra, exit 0/1
├── checkpoints/            # .pt — KHÔNG commit (xem .gitignore)
├── results/                # results.csv + bc_indices.json — KHÔNG commit
└── assets/samples/         # 20 ảnh test + index.json
```

**Quy tắc bất di bất dịch:** `core/model_def.py` là bản chép y hệt từ notebook. Sửa kiến trúc ở notebook thì phải chép lại xuống đây **và tăng `ARCH_VERSION` ở cả hai nơi** — nếu quên, app sẽ báo lỗi lệch phiên bản thay vì nạp sai âm thầm.

Mọi tham số phụ thuộc bộ dữ liệu (số lớp, tên lớp, mean/std chuẩn hoá) đều đọc **từ checkpoint** lúc chạy, không hard-code trong app.

## Lưu ý khi code và viết document

Không suy đoán, luôn hỏi lại khi bị mờ hồ, không chắc chắn
Có thể tìm cách tối ưu hơn
