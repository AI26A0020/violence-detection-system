# LỊCH SỬ CÂU LỆNH VÀ HÌNH ẢNH MINH CHỨNG THỰC NGHIỆM

## CUỘC THI SÁNG TẠO CÔNG NGHỆ TRÍ TUỆ NHÂN TẠO 2026 - BẢNG A

- Mã đội thi: AI26A0020
- Tên dự án: Phần mềm phát hiện hành vi bạo lực học đường (SchoolGuard AI)
- Mục đích tài liệu: Cung cấp đầy đủ nhật ký câu lệnh, log thực thi từ terminal và danh mục hình ảnh bằng chứng thực nghiệm để Ban giám khảo kiểm tra, đối chiếu tính xác thực của đề tài.

---

## I. NHẬT KÝ CÁC CÂU LỆNH THỰC THI CHÍNH (COMMAND LOG)

### 1. Khởi tạo môi trường và cài đặt các thư viện phụ thuộc

Các câu lệnh đã thực hiện trên môi trường máy trạm để thiết lập hệ thống:

```powershell
# Di chuyển vào thư mục dự án
cd C:\Users\"   "\."   "\"   "\"   "\violence_detection


# Cài đặt các gói thư viện cốt lõi
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Lệnh khởi chạy kiểm thử trên các tệp video thực tế

Các câu lệnh được thực thi trực tiếp để kiểm tra khả năng nhận diện trên từng tình huống:

### LƯU Ý: file test_videos ở trong thư mục nên phải dùng câu lệnh

```powershell
cd C:\Users\"   "\."   "\"   "\"   "\test_videos
```

```powershell
# Chạy kiểm thử video check9.mp4 (Hành lang, bắt tay, múa võ, đánh nhau hiệp 1 và hiệp 2)
python main.py --source check9.mp4

# Chạy kiểm thử video check10.mp4 (Nhảy nhót từ xa và cú đấm trực diện)
python main.py --source check10.mp4

# Chạy kiểm thử video check8.mp4 (Học sinh đi lại bình thường và ẩu đả trong lớp học)
python main.py --source check8.mp4

# Chạy kiểm thử video check3.mp4 (Biên tập viên ngồi trường quay và đột nhập tấn công)
python main.py --source check3.mp4

# Chạy kiểm thử chế độ đa camera (Multi-Camera Surveillance)
python main.py --multi
```

### 3. Lệnh chạy bộ kiểm thử tự động hóa toàn diện 19 kịch bản

Câu lệnh thực thi kiểm thử hồi quy độc lập không can thiệp giao diện:

```powershell
python verify_all.py
```

---

## II. BẰNG CHỨNG THỰC NGHIỆM: LOG KIỂM THỬ THỰC TẾ TRÊN TERMINAL

Dưới đây là trích xuất nguyên bản từ nhật ký thực thi của lệnh `python verify_all.py` trên toàn bộ 10 video:

```text
================================================================================
STARTING COMPREHENSIVE VERIFICATION ACROSS 19 SCENARIOS
================================================================================
PASS | Check9 (OK3): Di ngang qua nhau (12:10:09) | Alerts:  0 (Exp: 0-0) | Peak Conf:   0.0%
PASS | Check9 (OK3): Danh nhau hiep 1 (12:10:21)  | Alerts: 23 (Exp: 5-999) | Peak Conf:  90.0%
PASS | Check9 (OK3): Danh vo khong khi (12:10:32) | Alerts:  0 (Exp: 0-0) | Peak Conf:   0.1%
PASS | Check9 (OK3): Danh nhau hiep 2 (12:10:44)  | Alerts: 15 (Exp: 5-999) | Peak Conf:  90.0%
PASS | Check9 (OK3): Bat tay lam hoa (12:10:59)   | Alerts:  0 (Exp: 0-0) | Peak Conf:   2.0%
PASS | Check10 (OK4): Nhay tu xa (t=2-9s)         | Alerts:  0 (Exp: 0-0) | Peak Conf:  66.3%
PASS | Check10 (OK4): Cu dam truc dien (t=15-17s) | Alerts: 24 (Exp: 5-999) | Peak Conf:  98.8%
PASS | Check1 (Video1): Nguoi di bo mot minh      | Alerts:  0 (Exp: 0-0) | Peak Conf:  18.0%
PASS | Check3 (Video3): BTV Robin Roberts studio  | Alerts:  0 (Exp: 0-0) | Peak Conf:   2.0%
PASS | Check3 (Video3): Thanh nien di mot minh ngoai cua | Alerts:  0 (Exp: 0-0) | Peak Conf:  18.0%
PASS | Check3 (Video3): Dot nhap tan cong that    | Alerts: 15 (Exp: 2-999) | Peak Conf:  99.8%
PASS | Check6 (OK): Danh nhau ban dem             | Alerts: 35 (Exp: 10-999) | Peak Conf:  90.0%
PASS | Check7 (OK1): Peaceful pre-fight           | Alerts:  0 (Exp: 0-0) | Peak Conf:  18.6%
PASS | Check7 (OK1): Real fight brawl             | Alerts: 14 (Exp: 5-999) | Peak Conf:  90.0%
PASS | Check8 (OK2): Peaceful segment             | Alerts:  0 (Exp: 0-0) | Peak Conf:  18.4%
PASS | Check8 (OK2): Real fight brawl             | Alerts: 22 (Exp: 5-999) | Peak Conf:  90.0%
PASS | Check2 (Video2): Real street brawl         | Alerts: 44 (Exp: 5-999) | Peak Conf:  98.5%
PASS | Check4 (Video4): Real street fight         | Alerts: 19 (Exp: 5-999) | Peak Conf:  90.0%
PASS | Check5 (Video5): Real fight                | Alerts: 45 (Exp: 5-999) | Peak Conf:  99.9%
================================================================================
ALL 19 SCENARIOS PASSED WITH 100% ACCURACY!
================================================================================
```

---

## III. DANH MỤC CÁC TỆP HÌNH ẢNH VÀ CHỨNG CỨ THỰC TẾ

1. Thư mục ảnh chụp bằng chứng tự động (Snapshots):
   Hệ thống tự động lưu trữ các bức ảnh bằng chứng chụp lại khoảnh khắc bạo lực xảy ra với đầy đủ mốc thời gian và định danh camera tại thư mục `snapshots/`:

- `20260917_231021_Cam-Security_VIOLENCE_ALERT.jpg`: Bằng chứng vụ ẩu đả trong văn phòng.
- `20260917_231515_Cam-Security_CRITICAL_PUNCH.jpg`: Bằng chứng cú đấm trực diện vào mặt.
- `20260917_234210_Cam-Classroom_BRAWL_DETECTED.jpg`: Bằng chứng xô xát trong lớp học.

2. Ảnh chụp màn hình giao diện Tactical HUD:

- `HUD_Normal_Secure.png`: Giao diện trạng thái bình thường với thanh tiêu đề SECURE màu xanh, các khung người PERSON [SECURE] và [ACTIVE].
- `HUD_Critical_Alert.png`: Giao diện trạng thái báo động đỏ rực CRITICAL ALERT, đối tượng tham gia xô xát viền đỏ TARGET [FIGHTING], biểu đồ sóng dựng đứng vượt ngưỡng 75%.
- `HUD_Clean_Feed_Mode.png`: Giao diện chế độ Clean Feed khi nhấn phím TAB hoặc C (ẩn hoàn toàn HUD, chỉ giữ luồng video nguyên bản).

---

## IV. THÔNG TIN HỖ TRỢ KỸ THUẬT VÀ LIÊN HỆ

Trong quá trình Ban Giám khảo tiến hành chạy thử nghiệm và đánh giá sản phẩm, nếu có bất kỳ thắc mắc kỹ thuật nào cần hỗ trợ giải đáp trực tiếp, đại diện đội thi luôn sẵn sàng tiếp nhận và hỗ trợ kỹ thuật:

- Mã đội thi: AI26A0020
- Email liên hệ đội thi: maiphuc190682@gmail.com
- Số điện thoại: 0974 507 118
- Trân trọng cảm ơn sự quan tâm và đánh giá của Ban Giám khảo!
