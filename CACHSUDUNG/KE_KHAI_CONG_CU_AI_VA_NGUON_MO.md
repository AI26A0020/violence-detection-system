# BAN TỔ CHỨC CUỘC THI SÁNG TẠO CÔNG NGHỆ TRÍ TUỆ NHÂN TẠO 2026 - BẢNG A

## BẢN KÊ KHAI CÔNG CỤ TRÍ TUỆ NHÂN TẠO, NGUỒN DỮ LIỆU, API, THƯ VIỆN VÀ MÃ NGUỒN MỞ

- Mã định danh dự án: AI26A0020
- Tên dự án: Hệ thống Giám sát và Cảnh báo Bạo lực Học đường Thời gian thực
- Lĩnh vực ứng dụng: Thị giác máy tính và Giám sát an ninh thông minh
- Ngày hoàn thiện kê khai: 18/09/2026

---

### PHẦN 1. NGUYÊN TẮC VÀ CAM KẾT CỦA ĐỘI THI

Đội thi AI26A0020 cam kết tuân thủ đầy đủ Thể lệ Cuộc thi Sáng tạo Công nghệ Trí tuệ Nhân tạo 2026 và các quy định pháp luật hiện hành của Việt Nam, đặc biệt là Nghị định số 13/2023/NĐ-CP về bảo vệ dữ liệu cá nhân:

1. Tính trung thực trong học thuật và kỹ thuật: Mọi thành phần công nghệ sử dụng trong dự án, bao gồm mô hình học máy tiền huấn luyện, thư viện mã nguồn mở, bộ dữ liệu công khai và công cụ sinh mã/hỗ trợ, đều được kê khai minh bạch với nguồn gốc và giấy phép rõ ràng.
2. Quyền sở hữu trí tuệ: Đội thi trực tiếp thiết kế cấu trúc kết hợp đa nhánh (YOLOv8 + Optical Flow + MobileNetV2-LSTM), tự xây dựng thuật toán lọc nhiễu không gian và tách tầng chiều sâu, tự thực hiện quy trình huấn luyện tinh chỉnh (fine-tuning) và kiểm thử thực tế trên 19 kịch bản biên.
3. Không vi phạm quyền riêng tư: Hệ thống không sử dụng công nghệ nhận diện khuôn mặt, không thu thập và không lưu trữ thông tin định danh sinh trắc học của học sinh hay người xuất hiện trên camera. Dữ liệu chỉ được xử lý tạm thời trong bộ nhớ đệm phục vụ phân loại hành vi vật lý.

---

### PHẦN 2. KÊ KHAI MÔ HÌNH TRÍ TUỆ NHÂN TẠO VÀ TRỌNG SỐ TIỀN HUẤN LUYỆN

Hệ thống kết hợp ba khối xử lý AI hoạt động song song để vừa bảo đảm tốc độ xử lý trên 25 khung hình mỗi giây, vừa triệt tiêu tối đa các cảnh báo sai:

#### 1. Mô hình phát hiện dáng người (Person Detection)

- Tên mô hình: YOLOv8 Nano (YOLOv8n)
- Tổ chức phát triển: Ultralytics Inc.
- Giấy phép phân phối: GNU Affero General Public License v3.0 (AGPL-3.0) / Mã nguồn mở học thuật
- Trọng số sử dụng: yolov8n.pt (kích thước xấp xỉ 6.2 MB)
- Mục đích sử dụng: Xác định vị trí người trong từng khung hình, cung cấp tọa độ hộp bao (bounding box) và độ tin cậy để xác định vùng tương tác.
- Phần công việc do đội thi tự phát triển: Xây dựng thuật toán phân cụm khoảng cách tương tác đa người (Multi-person Spatial Proximity), bộ lọc diện tích tối thiểu chống nhiễu người ở quá xa, và thuật toán phân tách tầng chiều sâu (Depth Plane Separation) dựa trên tỷ lệ đáy hộp bao để không kích hoạt báo động khi hai đối tượng chỉ tình cờ che khuất nhau trên góc máy 2D.

#### 2. Khối trích xuất đặc trưng thị giác không gian (Spatial Feature Extractor)

- Tên kiến trúc: MobileNetV2
- Tác giả / Tổ chức phát triển: Google Research
- Giấy phép phân phối: Apache License 2.0
- Nguồn gốc trọng số: Tiền huấn luyện trên tập dữ liệu ImageNet-1k
- Mục đích sử dụng: Chuyển đổi mỗi khung hình kích thước 128x128 pixel thành một vector đặc trưng cô đọng 1280 chiều, tối ưu hóa cho các thiết bị biên có tài nguyên tính toán giới hạn.
- Cơ chế tinh chỉnh của đội: Đóng băng (freeze) 100 tầng trích xuất cạnh và cấu trúc cơ bản ban đầu để chống hiện tượng quên cục bộ, chỉ mở huấn luyện các khối thấu kính trên cùng kết hợp thêm tầng Dense 256 chiều, Batch Normalization và Dropout 0.4.

#### 3. Khối phân loại chuỗi thời gian hành vi (Temporal Action Classifier)

- Tên kiến trúc: Mạng nơ-ron hồi quy hai chiều Long Short-Term Memory (Bi-LSTM / LSTM)
- Cơ chế khởi tạo: Khởi tạo trọng số ngẫu nhiên từ phân phối Glorot Uniform, không sử dụng trọng số có sẵn từ bên thứ ba.
- Kích thước đầu vào: Chuỗi 16 vector đặc trưng liên tiếp trích xuất từ 16 khung hình gần nhất.
- Phần công việc do đội thi tự phát triển: Toàn bộ cấu trúc tầng LSTM 128 units, cơ chế kết hợp đặc trưng dòng quang học Farneback vào vector trạng thái, hàm mất mát Binary Cross-Entropy và chiến lược tối ưu hóa bằng Adam Optimizer với cơ chế giảm tốc độ học tự động khi đồ thị hàm loss chững lại (ReduceLROnPlateau).

---

### PHẦN 3. KÊ KHAI NGUỒN DỮ LIỆU HUẤN LUYỆN VÀ KIỂM THỬ

#### 1. Bộ dữ liệu dùng cho huấn luyện mô hình (Training & Validation Dataset)

- Tên bộ dữ liệu: RWF-2000 (Real-World Fight Dataset)
- Đơn vị công bố: Nhóm nghiên cứu thuộc Đại học Bách khoa Hồng Kông (Hong Kong Polytechnic University) và Đại học Tôn Trung Sơn (Sun Yat-sen University), công bố tại hội nghị quốc tế ACM MM.
- Quy mô dữ liệu: Gồm đúng 2.000 đoạn video trích xuất từ camera an ninh thực tế (CCTV), mỗi video có độ dài cố định 5 giây ở chuẩn 30 khung hình/giây.
- Phân bổ nhãn: 1.000 video ghi nhận hành vi bạo lực xô xát thực tế (Fight) và 1.000 video ghi nhận hành vi bình thường không bạo lực (Non-Fight) như chạy bộ, đá bóng, bắt tay, đi bộ, vẫy tay.
- Giấy phép sử dụng: Dữ liệu nghiên cứu phi thương mại phục vụ mục đích học thuật và cộng đồng.
- Xử lý dữ liệu của đội: Trích xuất chuỗi 16 khung hình đại diện theo phương pháp giãn cách đều, chuẩn hóa dải pixel về đoạn [0, 1], tăng cường dữ liệu bằng kỹ thuật lật ngang ngẫu nhiên (Random Horizontal Flip) và biến thiên độ sáng nhẹ (+-10%) để mô phỏng sự thay đổi ánh sáng trong ngày của trường học.

#### 2. Bộ dữ liệu thực tế dùng cho kiểm nghiệm hiện trường (Benchmark Testset)

- Nguồn gốc: 10 đoạn video thực nghiệm thực tế tại khuôn viên trường học và không gian công cộng, định dạng MP4, độ dài từ 10 giây đến 135 giây.
- Danh sách tệp kiểm thử chuyên sâu:
  - check1.mp4, check2.mp4: Tình huống ẩu đả bạo lực thực tế trong môi trường thiếu sáng và góc quay camera trên cao.
  - check3.mp4: Tình huống nhiều nhóm người di chuyển đông đúc, khoảng cách gần nhưng không có va chạm bạo lực.
  - check4.mp4, check5.mp4: Tình huống xô đẩy thể lực và xô xát trực diện giữa hai cá nhân.
  - check6.mp4, check7.mp4, check8.mp4: Các trường hợp sinh hoạt bình thường: học sinh ngồi tự học một mình, đi lại trong hành lang.
  - check9.mp4 (130 giây): Đoạn video thách thức cao nhất bao gồm người tập chống đẩy sát đất, bắt tay chào hỏi, ôm thân mật, và hai người đi lướt qua nhau ở hai vị trí xa gần khác nhau trên cùng một tầm nhìn camera.
  - check10.mp4: Tình huống đấm đá va chạm mạnh phát sinh đột ngột ở giây thứ 13 sau thời gian dài đứng nói chuyện.
- Quyền sở hữu và tính pháp lý: Các video tự thực hiện bởi nhóm tác giả với sự đồng ý tự nguyện của các thành viên tham gia mô phỏng tình huống, bảo đảm không vi phạm bản quyền và không lộ danh tính cá nhân không liên quan.

---

### PHẦN 4. KÊ KHAI THƯ VIỆN LẬP TRÌNH VÀ MÃ NGUỒN MỞ

Dự án được xây dựng trên nền tảng ngôn ngữ Python phiên bản 3.10. Tất cả các gói thư viện cài đặt qua tệp requirements.txt đều là phần mềm tự do nguồn mở với giấy phép tương thích:

1. TensorFlow (phiên bản từ 2.10.0 trở lên):
   - Giấy phép: Apache License 2.0
   - Công dụng: Xây dựng kiến trúc mô hình, nạp trọng số MobileNetV2, huấn luyện và thực thi luồng suy luận mạng nơ-ron LSTM.
2. OpenCV (opencv-python, phiên bản từ 4.7.0 trở lên):
   - Giấy phép: Apache License 2.0 (từ OpenCV 4.5)
   - Công dụng: Thu nạp luồng video tệp hoặc luồng mạng RTSP, triển khai thuật toán tính toán dòng quang học Farneback (calcOpticalFlowFarneback), xử lý đồ họa hiển thị thông tin trực tiếp trên màn hình, xuất hình ảnh chụp chứng cứ.
3. NumPy (phiên bản từ 1.23.0 trở lên):
   - Giấy phép: BSD 3-Clause License
   - Công dụng: Thao tác mảng nhiều chiều, chuẩn hóa ma trận tọa độ, tính toán góc hướng và độ lệch chuẩn của các vector chuyển động.
4. Ultralytics YOLOv8 (phiên bản từ 8.0.0 trở lên):
   - Giấy phép: GNU AGPL-3.0
   - Công dụng: Khởi tạo mô hình phát hiện đối tượng người phục vụ phân tích vùng không gian quan tâm.
5. PyYAML (phiên bản từ 6.0 trở lên):
   - Giấy phép: MIT License
   - Công dụng: Đọc và xác thực tệp cấu hình tham số tập trung config.yaml.
6. Requests (phiên bản từ 2.28.0 trở lên):
   - Giấy phép: Apache License 2.0
   - Công dụng: Xử lý giao thức truyền thông mạng gửi cảnh báo HTTP POST đến Webhook máy chủ quản lý và Telegram Bot API.

---

### PHẦN 5. KÊ KHAI CỔNG KẾT NỐI VÀ DỊCH VỤ GIAO TIẾP NGOẠI VI (API)

Hệ thống hoàn toàn độc lập về mặt tính toán AI (100% On-Premise / Edge Computing), không phụ thuộc vào bất kỳ dịch vụ đám mây xử lý ảnh nào của bên thứ ba để đảm bảo tính an toàn dữ liệu nội bộ. Chỉ sử dụng hai giao tiếp mạng sau đây cho việc chuyển tiếp thông báo:

1. Discord Bot API:
   - Dữ liệu gửi đi: Chuỗi văn bản thông báo mức độ khẩn cấp, nhãn camera, thời điểm xảy ra sự việc, và tệp ảnh chụp hiện trường dạng JPEG trích xuất ngay tại khung hình kích hoạt báo động.
2. Cổng Webhook HTTP nội bộ (Local Security Center Webhook):
   - Phương thức kết nối: Chuẩn REST JSON qua giao thức HTTP/HTTPS nội bộ
   - Dữ liệu gửi đi: Gói tin JSON chứa mã sự kiện, cấp độ cảnh báo (WARNING hoặc CRITICAL), độ tin cậy của thuật toán, tọa độ hộp bao đối tượng liên quan và đường dẫn lưu trữ tệp bằng chứng.
   - Mục đích: Dễ dàng ghép nối vào hệ thống phần mềm quản lý tòa nhà (BMS) hoặc hệ thống điều hành an ninh trường học sẵn có.

---

### PHẦN 6. KÊ KHAI CÔNG CỤ TRỢ LÝ TRÍ TUỆ NHÂN TẠO (AI COPILOT)

Nhằm nâng cao hiệu suất làm việc và tuân thủ định hướng ứng dụng AI có trách nhiệm của cuộc thi, đội thi kê khai trung thực việc sử dụng trợ lý AI trong quá trình nghiên cứu:

1. Công cụ sử dụng: Mô hình ngôn ngữ lớn Google Antigravity / Gemini 3.8 Flash
2. Phạm vi hỗ trợ của công cụ:
   - Tra cứu cú pháp tối ưu của thư viện OpenCV khi xử lý tính toán ma trận dòng quang học nhiều luồng.
   - Gợi ý cấu trúc kịch bản kiểm thử biên (Edge Case Scenarios) để rà soát các khả năng xảy ra lỗi dương tính giả đối với hành vi chống đẩy và bắt tay.
   - Hỗ trợ định dạng văn bản báo cáo theo đúng cấu trúc yêu cầu của Ban Tổ chức.
3. Phần việc cốt lõi do con người (thành viên đội thi) quyết định và chịu trách nhiệm:
   - Đưa ra giải pháp thuật toán phân tách tầng chiều sâu để loại trừ lỗi người đi ngang qua nhau.
   - Trực tiếp thu thập, gắn nhãn kiểm định trên tập 10 video camera hiện trường.
   - Trực tiếp chạy và tinh chỉnh các tham số ngưỡng trong tệp cấu hình config.yaml để đạt kết quả 19/19 kịch bản kiểm định đều thành công.
   - Toàn bộ nội dung báo cáo và mã nguồn được đội thi kiểm chứng cẩn trọng trước khi đóng gói nộp hồ sơ.

---

### PHẦN 7. KẾT LUẬN VÀ LỜI CAM ĐOAN

Bản kê khai này phản ánh chính xác và trung thực toàn bộ cấu trúc công nghệ của dự án AI26A0020. Đội thi hoàn toàn chịu trách nhiệm trước Ban Giám khảo và Ban Tổ chức Cuộc thi Sáng tạo Công nghệ Trí tuệ Nhân tạo 2026 về tính hợp pháp và nguồn gốc của tất cả các tài nguyên đã kê khai ở trên.

### PHẦN 8. THÔNG TIN HỖ TRỢ KỸ THUẬT VÀ LIÊN HỆ

Trong quá trình Ban Giám khảo tiến hành chạy thử nghiệm và đánh giá sản phẩm, nếu có bất kỳ thắc mắc kỹ thuật nào cần hỗ trợ giải đáp trực tiếp, đại diện đội thi luôn sẵn sàng tiếp nhận và hỗ trợ kỹ thuật:

- Mã đội thi: AI26A0020
- Email liên hệ đội thi: maiphuc190682@gmail.com
- Số điện thoại: 0974 507 118
- Trân trọng cảm ơn sự quan tâm và đánh giá của Ban Giám khảo!
