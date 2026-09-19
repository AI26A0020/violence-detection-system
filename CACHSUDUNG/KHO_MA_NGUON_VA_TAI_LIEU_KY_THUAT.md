# BAN TỔ CHỨC CUỘC THI SÁNG TẠO CÔNG NGHỆ TRÍ TUỆ NHÂN TẠO 2026 - BẢNG A

## TÀI LIỆU KỸ THUẬT VÀ HƯỚNG DẪN TRIỂN KHAI KHO MÃ NGUỒN (REPOSITORY GUIDE)

- Mã định danh dự án: AI26A0020
- Tên dự án: Hệ thống Giám sát và Cảnh báo Bạo lực Học đường Thời gian thực
- Nhánh phát triển chính: main / master
- Phiên bản phát hành: Release v1.0-final (18/09/2026)

---

### PHẦN 1. THÔNG TIN BÀN GIAO VÀ ĐƯỜNG DẪN KHO MÃ NGUỒN

Để Ban Giám khảo có thể thuận tiện kiểm chứng trực tiếp tính khả thi và độc lập của sản phẩm, đội thi cung cấp hai phương thức tiếp cận mã nguồn hoàn chỉnh:

1. Phương thức 1 - Kho lưu trữ trực tuyến (GitHub Repository):
   - Đường dẫn liên kết: [LINK](https://github.com/AI26A0020/violence-detection-system)
2. Phương thức 2 - Gói nén mã nguồn đầy đủ trên Google Drive (Dự phòng):
   - Đường dẫn liên kết Google Drive: [LINK](https://drive.google.com/drive/folders/1CW-c0ZYfFmp6U8962by2XlmqZjcTpX9-?usp=sharing)
   - Nội dung gói nén: Bao gồm toàn bộ mã nguồn nguồn gốc, tệp cấu hình tham số, tệp trọng số mô hình đã huấn luyện (.h5 và .pt), cùng 10 video kiểm nghiệm hiện trường.

---

### PHẦN 2. CẤU TRÚC CHI TIẾT THƯ MỤC MÃ NGUỒN

Toàn bộ dự án được tổ chức theo cấu trúc mô-đun hóa độc lập, phân định rõ ràng giữa tầng thu nhận dữ liệu, tầng suy luận thị giác máy tính, tầng phân tích logic nghiệp vụ và tầng giao tiếp ngoại vi:

```text
AI26A0020/
|-- config.yaml                     # Tệp cấu hình tham số tập trung (ngưỡng nhạy, camera, telegram)
|-- requirements.txt               # Danh sách thư viện phụ thuộc của dự án
|-- main.py                        # Điểm vào chính điều phối hệ thống và giao diện HUD tương tác
|-- person_detector.py             # Mô-đun phát hiện người (YOLOv8n), phân cụm và tách tầng chiều sâu
|-- optical_flow.py                # Mô-đun tính toán năng lượng và độ phân tán dòng quang học Farneback
|-- data_loader.py                 # Mô-đun tiền xử lý dữ liệu và trích xuất đặc trưng MobileNetV2
|-- train.py                       # Quy trình huấn luyện mô hình học sâu chuỗi thời gian MobileNetV2 + LSTM
|-- notifier.py                    # Mô-đun cảnh báo âm thanh và gửi tin nhắn/ảnh qua Telegram Bot & Webhook
|-- logger_module.py               # Mô-đun ghi nhật ký vận hành ra tệp văn bản, CSV và JSON
|-- verify_all.py                  # Bộ kiểm thử tự động toàn diện 19 kịch bản thực tế của hệ thống
|-- violence_detection_model.h5    # Trọng số mô hình MobileNetV2-LSTM đã tinh chỉnh trên tập RWF-2000
|-- yolov8n.pt                     # Trọng số mô hình YOLOv8 nano phát hiện dáng người
|-- snapshots/                     # Thư mục tự động lưu trữ ảnh chụp hiện trường khi phát hiện bạo lực
`-- test_videos/                   # Thư mục chứa 10 video camera hiện trường kiểm chứng thực nghiệm
    |-- check1.mp4 -> check5.mp4   # Các kịch bản kiểm nghiệm góc quay camera học đường (video1 -> video5)
    |-- check6.mp4                 # Tình huống bạo lực ban đêm (ok.mp4)
    |-- check7.mp4                 # Tình huống hòa bình trước xô xát & hỗn chiến (ok1.mp4)
    |-- check8.mp4                 # Tình huống hành lang hòa bình & ẩu đả lớp học (ok2.mp4)
    |-- check9.mp4                 # Kịch bản chống dương tính giả (chống đẩy, bắt tay, ôm, cắt ngang - ok3.mp4)
    `-- check10.mp4                # Kịch bản phát hiện bạo lực tức thì bộc phát tại giây 13 (ok4.mp4)
```

---

### PHẦN 3. CHI TIẾT CHỨC NĂNG CÁC MÔ-ĐUN LẬP TRÌNH

#### 1. Mô-đun main.py

- Chức năng: Đóng vai trò hạt nhân điều phối vòng lặp xử lý từng khung hình (Video Processing Pipeline Loop).
- Khả năng tương thích nguồn vào: Chấp nhận nguồn video tệp (.mp4, .avi), webcam cắm trực tiếp (chỉ số 0, 1) hoặc luồng camera IP chuẩn RTSP.
- Giao diện người dùng thời gian thực (HUD): Vẽ thanh năng lượng động, hộp bao phân loại đối tượng, chỉ số khung hình/giây thực tế (FPS) và trạng thái hệ thống.
- Bảng phím tắt điều khiển trực tiếp:
  - Phím [B]: Bật hoặc tắt hiển thị hộp bao đối tượng để phục vụ kiểm tra thẩm mỹ hoặc bảo vệ quyền riêng tư.
  - Phím [TAB]: Chuyển đổi nhanh giữa chế độ phân tích đầy đủ (HUD) và chế độ xem luồng sạch (Clean feed).
  - Phím [SPACE]: Tạm dừng hoặc tiếp tục luồng phát để Ban Giám khảo soi kỹ từng khung hình va chạm.
  - Phím [H]: Hiển thị bản đồ nhiệt trực quan dòng quang học Farneback để quan sát hướng chuyển động cơ học.
  - Phím [S]: Chụp ảnh khung hình hiện tại lưu ngay vào đĩa cứng thủ công.
  - Phím [Q]: Thoát ứng dụng an toàn, đóng luồng camera và lưu đầy đủ nhật ký ra đĩa cứng.

#### 2. Mô-đun person_detector.py

- Chức năng: Phát hiện vị trí con người trong khung hình sử dụng mạng YOLOv8 nano.
- Thuật toán phân cụm không gian (Spatial Proximity): Chỉ khi hai hoặc nhiều người đứng trong bán kính tương tác gần nhau mới kích hoạt khối kiểm tra va chạm, giúp tiết kiệm tài nguyên khi không gian có nhiều người đứng độc lập.
- Thuật toán phân tách tầng chiều sâu (Depth Plane Separation): Sử dụng tỷ lệ chiều cao và tọa độ đáy hộp bao để nhận diện trường hợp một người ở gần camera đi lướt qua một người ngồi ở xa. Ngăn chặn triệt để hiện tượng báo động giả khi hai người chỉ tình cờ thẳng hàng trên ảnh 2D.
- Bộ lọc diện tích tối thiểu (min_box_area): Loại bỏ các đối tượng ở quá xa tầm quan sát của camera để tránh nhiễu tín hiệu.

#### 3. Mô-đun optical_flow.py

- Chức năng: Tính toán trường vector dòng quang học mật độ cao Gunnar-Farneback trên vùng không gian người tương tác.
- Chỉ số trích xuất: Năng lượng chuyển động đột ngột (Motion Intensity) và độ phân tán hướng (Flow Dispersion). Các hành vi ẩu đả có độ phân tán vector chuyển động rất cao và hỗn loạn, khác biệt hoàn toàn với hành vi chuyển động đều như chạy bộ hoặc tập thể dục.

#### 4. Mô-đun notifier.py

- Chức năng: Xử lý cảnh báo đa kênh bất đồng bộ (Non-blocking).
- Cảnh báo âm thanh: Phát âm tần số cao bằng thư viện âm thanh hệ thống để thu hút sự chú ý của bảo vệ trực ban.
- Cảnh báo Telegram: Tự động nén và đính kèm ảnh chụp hiện trường rõ nét, gửi kèm thông điệp báo động có định dạng rõ ràng trong vòng chưa đầy 1 giây.
- Webhook JSON: Gửi dữ liệu có cấu trúc sang máy chủ quản lý an ninh của nhà trường.

#### 5. Mô-đun verify_all.py

- Chức năng: Bộ công cụ kiểm thử tự động độc lập dành cho Ban Giám khảo và kỹ sư kiểm thử.
- Kịch bản thực thi: Tự động duyệt qua 19 kịch bản kiểm tra toàn diện, bao gồm cả các video hòa bình phức tạp và video bạo lực đột ngột, đo lường tỷ lệ chính xác, số lần báo động giả và thời gian phản hồi trung bình.

---

### PHẦN 4. YÊU CẦU PHẦN CỨNG VÀ MÔI TRƯỜNG THỰC THI

Hệ thống được tối ưu hóa đặc biệt để có thể vận hành ổn định trên các thiết bị máy tính phổ thông tại các phòng bảo vệ trường học:

1. Cấu hình phần cứng tối thiểu (Chạy bằng CPU):
   - Vi xử lý (CPU): Intel Core i5 thế hệ thứ 8 trở lên (hoặc AMD Ryzen 5 tương đương), tối thiểu 4 nhân vật lý.
   - Bộ nhớ trong (RAM): Tối thiểu 8 GB.
   - Dung lượng ổ cứng trống: 2 GB để chứa mã nguồn, môi trường ảo và tệp trọng số.
   - Tốc độ xử lý đo được: Đạt 22 - 28 khung hình/giây trên CPU Intel Core i5-1135G7.
2. Cấu hình phần cứng khuyến nghị (Có hỗ trợ GPU):
   - Vi xử lý: Intel Core i7 thế hệ 10 trở lên hoặc AMD Ryzen 7.
   - Bộ nhớ trong: 16 GB RAM.
   - Thẻ đồ họa rời: NVIDIA GeForce GTX 1650 / RTX 2060 / RTX 3050 trở lên (hỗ trợ CUDA 11.8+).
   - Tốc độ xử lý đo được: Đạt 45 - 60 khung hình/giây, đáp ứng cùng lúc 2 đến 3 luồng camera an ninh.
3. Môi trường hệ điều hành:
   - Tương thích 100% trên Microsoft Windows 10/11 (64-bit).
   - Tương thích hoàn toàn trên các bản phân phối Linux (Ubuntu 20.04 LTS, Ubuntu 22.04 LTS).
   - Phiên bản ngôn ngữ: Python 3.9 đến Python 3.11.

---

### PHẦN 5. QUY TRÌNH CÀI ĐẶT TỪNG BƯỚC CHO BAN GIÁM KHẢO

Ban Giám khảo có thể thiết lập và chạy thử nghiệm hệ thống trong vòng chưa đầy 5 phút theo 4 bước sau:

#### Bước 1: Chuẩn bị mã nguồn

Tải thư mục dự án từ liên kết nộp bài hoặc giải nén tệp zip vào một thư mục làm việc, ví dụ:

```bash
cd violence_detection
```

#### Bước 2: Khởi tạo và kích hoạt môi trường ảo Python

Sử dụng môi trường ảo giúp cách ly hoàn toàn các gói thư viện, không làm xáo trộn môi trường phần mềm sẵn có trên máy tính của Ban Giám khảo:

```bash
# Trên hệ điều hành Windows:
python -m venv .venv
.venv\Scripts\activate

# Trên hệ điều hành Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate
```

#### Bước 3: Cài đặt các thư viện phụ thuộc

Thực hiện cài đặt tất cả các gói phụ thuộc qua tệp requirements.txt chuẩn:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Bước 4: Kiểm tra sự hiện diện của tệp trọng số mô hình

Bảo đảm hai tệp trọng số đã nằm trong thư mục gốc của dự án:

- violence_detection_model.h5 (xấp xỉ 9.5 MB)
- yolov8n.pt (xấp xỉ 6.5 MB)
  Nếu chưa có, hệ thống sẽ tự động thông báo và hướng dẫn đường dẫn tải từ kho lưu trữ của đội thi.

---

### PHẦN 6. HƯỚNG DẪN CHẠY VÀ THẨM ĐỊNH CÁC TÌNH HUỐNG THỰC TẾ

Ban Giám khảo có thể thẩm định trực tiếp từng tính năng cốt lõi thông qua các câu lệnh mẫu sau:

#### 1. Kiểm tra kịch bản tự động toàn diện (Automated Test Suite)

Lệnh này sẽ tự động nạp mô hình và kiểm tra 19 trường hợp thử nghiệm nghiêm ngặt:

```bash
python verify_all.py
```

Kết quả mong đợi trên màn hình:
Hệ thống in bảng tổng kết 19/19 trường hợp ĐẠT (PASS), xác nhận không có bất kỳ báo động giả nào xảy ra ở các video sinh hoạt bình thường và phát hiện chính xác 100% các tình huống xô xát.

#### 2. Kiểm tra tình huống chống dương tính giả với video check9.mp4

Video check9.mp4 dài 130 giây chứa đầy đủ các hành vi nhạy cảm: chống đẩy sát đất, bắt tay, ôm, người đi lướt qua nhau:

```bash
python main.py --source check9.mp4
```

Kết quả mong đợi: Hệ thống hiển thị nhãn màu xanh lá "NORMAL" trong suốt toàn bộ 130 giây, thanh năng lượng giữ ở mức an toàn, không có chuông báo động và không có thông báo lỗi.

#### 3. Kiểm tra tình huống phát hiện bạo lực tức thì với video check10.mp4

Video check10.mp4 ghi lại cảnh hai người đứng nói chuyện bình thường, sau đó đột ngột ẩu đả ở giây thứ 13:

```bash
python main.py --source check10.mp4
```

Kết quả mong đợi:

- Từ giây thứ 1 đến giây thứ 12: Hệ thống hiển thị "NORMAL" màu xanh.
- Ngay khi hành vi đấm đá xảy ra ở giây thứ 13: Hệ thống lập tức chuyển sang nhãn màu đỏ "VIOLENCE DETECTED", hộp bao đổi màu đỏ, phát chuông cảnh báo âm thanh, tự động trích xuất ảnh hiện trường lưu vào thư mục snapshots/ và gửi thông báo khẩn cấp.

#### 4. Thử nghiệm trực tiếp với Webcam gắn trên máy tính

Để Ban Giám khảo trực tiếp tương tác trước ống kính:

```bash
python main.py --source 0
```

Ban Giám khảo có thể thử bắt tay nhau, đi ngang qua nhau hoặc giả lập động tác xô đẩy để kiểm chứng tốc độ phản hồi và tính chính xác tức thì của hệ thống.

#### 5. Thử nghiệm kết nối camera an ninh IP qua giao thức RTSP

```bash
python main.py --source rtsp://admin:matkhau@192.168.1.100:554/stream1
```

---

### PHẦN 7. HƯỚNG DẪN TÙY CHỈNH THAM SỐ TRONG TỆP CONFIG.YAML

Tất cả các ngưỡng thuật toán đều được cấu hình tập trung trong tệp config.yaml, cho phép người quản trị dễ dàng tinh chỉnh theo đặc thù từng vị trí lắp đặt:

1. Nhóm tham số phát hiện dáng người (detection):
   - person_confidence: Ngưỡng tin cậy nhận diện người của YOLO (mặc định 0.45).
   - proximity_threshold_ratio: Bán kính phát hiện tương tác gần giữa hai đối tượng tính theo chiều rộng cơ thể (mặc định 1.25).
   - depth_plane_tolerance: Ngưỡng dung sai phân tách tầng chiều sâu nhằm loại trừ hiện tượng che khuất quang học 2D (mặc định 0.35).
2. Nhóm tham số phân loại chuỗi thời gian (classification):
   - sequence_length: Số khung hình trong cửa sổ trượt phân tích (mặc định 16 khung hình).
   - violence_threshold: Ngưỡng xác suất kích hoạt cảnh báo bạo lực (mặc định 0.75).
   - persistence_frames: Số khung hình liên tiếp vượt ngưỡng để chính thức kích hoạt báo động (mặc định 6 khung hình, giúp loại trừ các cú vung tay vô ý).
3. Nhóm tham số thông báo (notification):
   - webhook_url(ở config.yaml): Chứa url của bot Discord API
   - telegram_enabled: Bật/tắt gửi tin nhắn Telegram (true/false).
   - telegram_token: Khóa bảo mật Telegram Bot API.
   - telegram_chat_id: Mã định danh kênh tiếp nhận cảnh báo của ban bảo vệ.
   - local_sound: Bật/tắt còi cảnh báo âm thanh tại máy trạm giám sát.

---

### PHẦN 8. THÔNG TIN HỖ TRỢ KỸ THUẬT VÀ LIÊN HỆ

Trong quá trình Ban Giám khảo tiến hành chạy thử nghiệm và đánh giá sản phẩm, nếu có bất kỳ thắc mắc kỹ thuật nào cần hỗ trợ giải đáp trực tiếp, đại diện đội thi luôn sẵn sàng tiếp nhận và hỗ trợ kỹ thuật:

- Mã đội thi: AI26A0020
- Email liên hệ đội thi: maiphuc190682@gmail.com
- Số điện thoại: 0974 507 118
- Trân trọng cảm ơn sự quan tâm và đánh giá của Ban Giám khảo!
