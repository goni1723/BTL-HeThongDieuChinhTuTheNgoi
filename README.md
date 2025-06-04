                                                    🪑 Hệ thống nhắc nhở điều chỉnh tư thế ngồi
📌 Giới thiệu
Hệ thống nhắc nhở điều chỉnh tư thế ngồi là một giải pháp sử dụng công nghệ học sâu để giám sát và nhắc nhở người dùng điều chỉnh tư thế ngồi đúng cách, đặc biệt là trong các môi trường làm việc và học tập kéo dài. Với sự kết hợp của MediaPipe và YOLOv8, hệ thống có khả năng phát hiện và phân tích tư thế ngồi của người dùng theo thời gian thực.

MediaPipe: Là một thư viện mã nguồn mở của Google, được sử dụng để phát hiện và theo dõi các điểm đặc trưng trên cơ thể người, giúp nhận diện tư thế ngồi.

YOLOv8: Là một mô hình học sâu tiên tiến, có khả năng phát hiện đối tượng trong ảnh và video với độ chính xác cao, giúp nhận diện tư thế ngồi không đúng hoặc sai lệch trong các tình huống phức tạp.

Dự án này cung cấp một hệ thống cảnh báo tự động khi người dùng ngồi sai tư thế hoặc ngồi quá lâu, giúp bảo vệ sức khỏe và giảm thiểu các vấn đề về xương khớp, đặc biệt là đau lưng và các bệnh lý liên quan đến tư thế ngồi.

🧠 Mục tiêu
Cải thiện sức khỏe người dùng: Giảm thiểu các vấn đề về cột sống và đau lưng do ngồi sai tư thế trong thời gian dài.

Tăng cường ý thức về tư thế ngồi đúng: Khuyến khích người dùng điều chỉnh tư thế ngồi đúng cách, giúp duy trì sức khỏe lâu dài.

Phát triển công nghệ tiên tiến: Áp dụng các công nghệ học máy (AI) và xử lý hình ảnh (Computer Vision) như MediaPipe và YOLOv8 để theo dõi và phân tích tư thế ngồi.

Tạo thói quen tốt: Giúp người dùng nhận ra các tư thế ngồi sai và hình thành thói quen ngồi đúng.

⚙️ Công nghệ sử dụng
Thành phần	Mô tả
MediaPipe	Thư viện của Google dùng để phát hiện và theo dõi các điểm cơ thể người (keypoints) giúp nhận diện tư thế ngồi.
YOLOv8	Mô hình phát hiện đối tượng hiện đại giúp nhận diện tư thế ngồi và cảnh báo khi người dùng ngồi sai.
Python	Ngôn ngữ lập trình chính để triển khai hệ thống, xử lý dữ liệu và kết nối các mô hình AI.
OpenCV	Thư viện xử lý hình ảnh, giúp xử lý video và kết nối với mô hình MediaPipe và YOLOv8.
TensorFlow/ PyTorch	Các framework học sâu hỗ trợ việc huấn luyện và sử dụng các mô hình YOLOv8.
Flask/ FastAPI	Cung cấp backend API để giao tiếp giữa hệ thống nhận diện và ứng dụng web/di động.

🔄 Quy trình hoạt động
Thu thập video đầu vào: Hệ thống sử dụng camera để thu thập hình ảnh hoặc video từ người dùng trong thời gian thực.

Phân tích tư thế ngồi:

MediaPipe sẽ phát hiện các điểm đặc trưng trên cơ thể người dùng và xác định các tư thế ngồi.

YOLOv8 sẽ phân tích hình ảnh và phát hiện các đối tượng liên quan đến tư thế ngồi sai.

Đánh giá tư thế: Dựa vào kết quả phân tích từ MediaPipe và YOLOv8, hệ thống sẽ xác định xem người dùng có ngồi sai tư thế hay không.

Cảnh báo nhắc nhở: Khi người dùng ngồi sai hoặc ngồi quá lâu, hệ thống sẽ gửi cảnh báo nhắc nhở dưới dạng thông báo âm thanh, hình ảnh hoặc tin nhắn để yêu cầu người dùng điều chỉnh tư thế.

Lập lịch nhắc nhở định kỳ: Hệ thống có thể lập lịch tự động để nhắc nhở người dùng thay đổi tư thế sau một khoảng thời gian nhất định.

📍 Tính năng nổi bật
Phát hiện tư thế ngồi sai: Hệ thống có khả năng nhận diện các tư thế ngồi sai lệch nhờ vào công nghệ nhận diện cơ thể từ MediaPipe và YOLOv8.

Cảnh báo thông minh: Hệ thống sẽ gửi cảnh báo khi người dùng ngồi sai tư thế hoặc ngồi quá lâu mà không thay đổi tư thế.

Theo dõi thời gian ngồi: Hệ thống sẽ theo dõi thời gian ngồi và gửi nhắc nhở sau mỗi khoảng thời gian nhất định.

Giao diện thân thiện: Cung cấp giao diện người dùng dễ sử dụng, hiển thị thông tin về tư thế ngồi và cảnh báo nhắc nhở.

Khả năng mở rộng: Hệ thống có thể dễ dàng mở rộng để hỗ trợ nhiều người dùng hoặc tích hợp với các thiết bị phần cứng khác.

🔮 Định hướng phát triển
Cải tiến nhận diện tư thế: Tích hợp các mô hình học sâu mới nhất như OpenPose hoặc các thuật toán phân tích tư thế nâng cao để tăng độ chính xác.

Tích hợp thêm tính năng chăm sóc sức khỏe: Đưa vào các tính năng theo dõi sức khỏe tổng thể như nhịp tim, mức độ căng thẳng, v.v.

Tích hợp ứng dụng di động: Phát triển ứng dụng di động để giúp người dùng dễ dàng theo dõi tư thế ngồi và nhận cảnh báo ngay trên điện thoại.

Hệ thống phân tích dữ liệu: Phát triển các công cụ phân tích dữ liệu giúp người dùng theo dõi lịch sử tư thế ngồi của họ và cải thiện thói quen.

