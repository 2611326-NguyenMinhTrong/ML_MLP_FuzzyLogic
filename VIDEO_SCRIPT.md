# Kịch bản video demo (3–5 phút) — Step 3.3

Theo đúng mạch kế hoạch: bài toán → tab 1 (một ảnh vi phạm) → tab 2 (mẫu M sửa được) → tab 3 (trade-off). Số liệu trong lời thoại lấy từ kết quả thật đã chạy (mục 11 kế hoạch) — khi quay, kiểm tra lại con số hiện trên màn hình khớp với đây; nếu λ\* hoặc số liệu đổi do chạy lại lưới, sửa lời thoại theo, không đọc số cũ.

**Chuẩn bị trước khi quay:** chạy `python scripts/smoke_test.py` phải xanh 12/12; mở sẵn app ở tab **Một ảnh**, đã chọn sẵn một ảnh vi phạm (gallery, không phải upload — mở màn bằng gallery để tránh rủi ro ảnh ngoài phân bố).

---

## Cảnh 1 — Bài toán (0:00–0:45)

**Hình:** slide tĩnh hoặc màn hình app ở tab Một ảnh, chưa bấm gì.

**Lời thoại:**
> "Đây là mô hình MLP phân loại ảnh CIFAR-100 theo hai tầng nhãn cùng lúc: một đầu ra đoán *nhãn con* — một trong 100 lớp như 'apple', 'bicycle' — đầu ra kia đoán *nhãn cha*, một trong 20 nhóm như 'fruit and vegetables', 'vehicles'.
>
> Vấn đề: hai đầu ra này **độc lập với nhau**, nên có thể tự mâu thuẫn — đầu 1 nói 'quả táo', đầu 2 nói 'phương tiện giao thông'. Chúng tôi thử nghiệm liệu thêm một *fuzzy logic loss* vào lúc huấn luyện có làm mô hình nhất quán hơn không, và so với hai cách xử lý hoàn toàn miễn phí: hậu xử lý cứng, và marginalization."

---

## Cảnh 2 — Tab Một ảnh: một ca vi phạm (0:45–2:00)

**Hình:** chọn ảnh gallery, trỏ chuột vào badge đỏ VI PHẠM, rồi vào 2 bar chart, rồi vào bảng 3 chế độ.

**Lời thoại:**
> "Đây là ảnh con bus. Mô hình đoán nhãn con là 'train' — sai rồi — và tầng cha nói 'vehicles_2'. Nhưng 'train' thực ra thuộc nhóm 'vehicles_1'. Hai đầu ra tự mâu thuẫn nhau — badge đỏ báo VI PHẠM.
>
> [trỏ vào bảng 3 chế độ] Đây chính là chỗ ba cách xử lý khác nhau: chế độ *raw* — đầu ra thô — thì vi phạm như vừa thấy. Chế độ *hard* tra bảng từ nhãn con nên luôn nhất quán 100% — nhưng phải bỏ qua hoàn toàn đầu ra tầng cha. Chế độ *marginal* cộng xác suất các nhãn con trong nhóm lại."

---

## Cảnh 3 — Tab So sánh: mẫu M sửa được / phá hỏng (2:00–3:15)

**Hình:** tab So sánh, để nguyên cặp mặc định (B0 vs M), trỏ vào lưới 20 ảnh, click một ảnh viền xanh rồi một ảnh viền đỏ.

**Lời thoại:**
> "Tab này so trực tiếp mô hình gốc B0 với mô hình M đã huấn luyện thêm fuzzy logic loss, trên cùng 20 ảnh. Viền xanh là ảnh M *sửa được* — B0 sai, M đúng. Viền đỏ là ảnh M *phá hỏng* — B0 đúng, M lại sai.
>
> [click ảnh viền xanh] Đây là một ca M sửa đúng... [click ảnh viền đỏ] ...và đây là một ca M làm hỏng.
>
> Nhưng nhìn tổng thể trên toàn bộ 10.000 ảnh test, M sửa được 702 ảnh và phá hỏng 798 ảnh — phá nhiều hơn sửa. Và khi so với nhóm chứng — so B0 với chính B0 ở các lần khởi tạo ngẫu nhiên khác nhau — chênh lệch đó nằm gọn trong mức dao động bình thường. Nói cách khác, dữ liệu **không đủ bằng chứng** để nói fuzzy logic loss thật sự có tác dụng."

---

## Cảnh 4 — Tab Tổng quan: trade-off (3:15–4:15)

**Hình:** tab Tổng quan, đồ thị trade-off, trỏ vào điểm sao λ\*=2 và hai điểm mờ bị Pareto-dominate.

**Lời thoại:**
> "Đồ thị này cho thấy đánh đổi cốt lõi: trục ngang là độ chính xác nhãn con, trục dọc là độ nhất quán. Càng tăng λ — mức phạt logic — độ nhất quán càng cao, nhưng độ chính xác càng giảm. Không có điểm nào vừa tăng cả hai.
>
> Ngôi sao là λ=2, cấu hình chúng tôi chọn theo quy tắc đã đăng ký trước: nhất quán cao nhất mà độ chính xác không giảm quá 1 điểm phần trăm.
>
> [trỏ vào bảng bốn mốc] Nhưng đây là điểm mấu chốt: so với hai cách xử lý *miễn phí* — hậu xử lý cứng đạt nhất quán 100%, marginalization đạt gần 83% — mô hình M tốn công huấn luyện thêm chỉ đạt 79% và độ chính xác còn thấp hơn cả hai. Trên bài toán này, fuzzy logic loss **không đáng công** so với hai mốc miễn phí."

---

## Cảnh 5 — Kết (4:15–4:45)

**Hình:** quay lại slide hoặc màn hình tổng quan.

**Lời thoại:**
> "Đây là một kết quả âm tính, nhưng có giá trị — đề cương của chúng tôi đã đóng khung trước khả năng này. Đóng góp thực sự nằm ở **quy trình đo**: so với mốc miễn phí, dùng nhóm chứng để loại nhiễu ngẫu nhiên, và tách bạch được liệu cải thiện có thật hay chỉ là dao động giữa các lần huấn luyện. Đó là thứ nhiều nghiên cứu bỏ qua, và là thứ khiến kết luận của chúng tôi đáng tin."

---

## Ghi chú kỹ thuật khi quay

- Ghi màn hình ở độ phân giải tối thiểu 1280×720; phóng to trình duyệt 100–110% để chữ trong bar chart không bị nhỏ.
- Nếu λ\* hoặc số liệu 702/798 đổi sau khi chạy lại lưới, các số này lấy từ `results/report_assets.md` (đường dẫn `train_model_colab/drive_FuzzyMLP/results/`) — cập nhật lại lời thoại Cảnh 3–4.
- Dự phòng: nếu mạng/máy demo trực tiếp gặp sự cố lúc bảo vệ, phát video này thay thế — vì vậy chuẩn bị sẵn bản xuất `.mp4` kèm trong link nộp bài (mục 3.5 kế hoạch).
