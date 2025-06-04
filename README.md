🪑 Hệ thống nhắc nhở điều chỉnh tư thế ngồi
📌 Giới thiệu
Hệ thống nhắc nhở điều chỉnh tư thế ngồi là một giải pháp công nghệ nhằm cải thiện sức khỏe và tư thế ngồi của người dùng, đặc biệt là trong môi trường làm việc hoặc học tập kéo dài. Hệ thống sử dụng các cảm biến để giám sát tư thế ngồi của người dùng, phát hiện khi người ngồi sai tư thế và đưa ra cảnh báo nhắc nhở để điều chỉnh. Mục tiêu của dự án là giúp giảm thiểu các vấn đề về cột sống, đau lưng và các bệnh lý liên quan đến tư thế không đúng.

Thông qua việc sử dụng các cảm biến góc và cảm biến chuyển động, dữ liệu được xử lý và truyền về hệ thống. Khi phát hiện người dùng có dấu hiệu ngồi sai tư thế trong thời gian dài, hệ thống sẽ gửi thông báo nhắc nhở tới người dùng, giúp họ duy trì tư thế ngồi đúng và khỏe mạnh.

🧠 Mục tiêu
Cải thiện sức khỏe: Giảm thiểu các vấn đề về xương khớp và cột sống do tư thế ngồi sai.

Tăng cường ý thức người dùng: Khuyến khích người dùng điều chỉnh tư thế ngồi đúng.

Tự động hóa cảnh báo: Hệ thống tự động nhận diện tư thế ngồi sai và nhắc nhở người dùng mà không cần sự can thiệp của con người.

Tạo thói quen ngồi đúng: Giúp người dùng xây dựng thói quen ngồi đúng trong thời gian dài.

⚙️ Công nghệ sử dụng
Thành phần	Mô tả
NodeMCU ESP8266	Vi điều khiển, gửi dữ liệu cảm biến qua WiFi
Cảm biến góc	Xác định góc nghiêng của cơ thể để phát hiện tư thế ngồi sai
Cảm biến chuyển động	Phát hiện khi người dùng ngồi quá lâu mà không thay đổi tư thế
Ứng dụng di động/Website	Cung cấp giao diện hiển thị trạng thái và thông báo nhắc nhở
Firebase Realtime DB	Lưu trữ và đồng bộ dữ liệu từ cảm biến
Google Maps API	(Nếu có) Theo dõi vị trí người dùng và cung cấp các thông tin hỗ trợ khác (ví dụ như có thể là thông tin văn phòng, phòng học, v.v.)

🔄 Quy trình hoạt động
Thu thập dữ liệu: Các cảm biến theo dõi tư thế ngồi của người dùng, đo góc và chuyển động.

Xử lý dữ liệu: NodeMCU nhận dữ liệu từ cảm biến và xử lý.

Thông báo nhắc nhở: Nếu tư thế ngồi sai hoặc người dùng ngồi quá lâu, hệ thống sẽ gửi cảnh báo thông qua ứng dụng hoặc đèn báo.

Lập lịch nhắc nhở: Hệ thống tự động tạo ra các nhắc nhở sau mỗi khoảng thời gian nhất định nếu người dùng vẫn duy trì tư thế sai.

📍 Tính năng nổi bật
Phát hiện tư thế ngồi sai: Hệ thống tự động nhận diện các tư thế ngồi không đúng.

Cảnh báo nhắc nhở: Người dùng nhận thông báo khi cần điều chỉnh tư thế ngồi.

Theo dõi thời gian ngồi: Cảnh báo nếu người dùng ngồi quá lâu mà không thay đổi tư thế.

Dễ dàng cấu hình và mở rộng: Có thể tùy chỉnh các ngưỡng cảnh báo và dễ dàng mở rộng để hỗ trợ nhiều người dùng.

🔮 Định hướng phát triển
Cải tiến nhận diện tư thế: Tích hợp các công nghệ nhận diện tư thế bằng AI để cải thiện độ chính xác.

Theo dõi sức khỏe tổng thể: Tích hợp thêm các cảm biến đo lường các yếu tố khác như nhịp tim, mức độ căng thẳng, để cung cấp một hệ thống chăm sóc sức khỏe toàn diện.

Tích hợp với các thiết bị thông minh khác: Kết nối hệ thống với các thiết bị gia đình thông minh như máy tính bảng, laptop, để nhận diện tư thế ngồi trong các tình huống khác nhau.
