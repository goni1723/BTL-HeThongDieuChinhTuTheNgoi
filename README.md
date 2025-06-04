# Hệ Thống Nhận Diện và Nhắc Nhở Tư Thế Ngồi

## 📌 Giới thiệu

Hệ thống nhận diện và nhắc nhở tư thế ngồi là một ứng dụng sử dụng công nghệ thị giác máy tính (Computer Vision) nhằm hỗ trợ người dùng duy trì tư thế ngồi đúng trong quá trình học tập và làm việc. Việc ngồi sai tư thế trong thời gian dài có thể dẫn đến các vấn đề sức khỏe như đau lưng, cong vẹo cột sống và giảm hiệu suất công việc. Hệ thống sẽ phát hiện tư thế sai và đưa ra cảnh báo kịp thời giúp người dùng điều chỉnh.

---

## 🎯 Mục tiêu

- Xây dựng một hệ thống có khả năng **phát hiện tư thế ngồi** thông qua hình ảnh từ camera.
- **Cảnh báo thời gian thực** khi người dùng ngồi sai tư thế.
- Cải thiện thói quen ngồi làm việc/làm bài lâu dài.
- Hướng đến khả năng mở rộng cho nhiều đối tượng sử dụng như học sinh, sinh viên, nhân viên văn phòng.

---

## 🛠️ Công nghệ sử dụng

- **Ngôn ngữ lập trình**: Python
- **Thị giác máy tính**: [MediaPipe](https://developers.google.com/mediapipe) (Pose Detection)
- **Giao diện người dùng**: Tkinter / PyQt5 (có thể tùy chọn)
- **Cảnh báo**: Âm thanh (qua thư viện `playsound`) hoặc thông báo popup
- **Môi trường phát triển**: Jupyter Notebook / Visual Studio Code

---

## 🔁 Quy trình hoạt động

1. **Mở hệ thống và kích hoạt webcam**.
2. **MediaPipe Pose** nhận diện các điểm trên cơ thể (đặc biệt là vai, cổ, lưng).
3. **Tính toán góc nghiêng** hoặc khoảng cách giữa các điểm để đánh giá tư thế.
4. **Phân loại tư thế**: đúng hoặc sai (ví dụ: khom lưng, ngả người quá mức).
5. **Gửi cảnh báo** bằng âm thanh hoặc cửa sổ nhắc nhở nếu tư thế sai kéo dài.
6. **Ghi nhận log lịch sử** nếu cần (cho các phiên bản nâng cao).

---

## 🌟 Tính năng nổi bật

- 🎯 Nhận diện chính xác tư thế ngồi theo thời gian thực.
- 🔔 Cảnh báo kịp thời giúp người dùng điều chỉnh đúng tư thế.
- 📊 Tùy chọn ghi lại dữ liệu để phân tích thói quen tư thế theo thời gian.
- 👩‍💻 Giao diện đơn giản, dễ sử dụng, thân thiện với người dùng.

---

## 🚀 Định hướng phát triển

- Hỗ trợ **nhận diện nhiều người dùng** cùng lúc.
- Thêm tùy chọn **huấn luyện mô hình học máy** để nâng cao độ chính xác.
- Đồng bộ với thiết bị **đeo cảm biến** để tăng độ tin cậy.
- Tích hợp hệ thống **báo cáo sức khỏe tư thế định kỳ** cho người dùng.
- Triển khai trên **nền tảng web hoặc ứng dụng di động**.

---

## 👤 Thông tin tác giả

- **Họ tên**: [Lê Tuấn Anh]
- **Email**: [Goni1723@gmail.com]
- **Lớp/Trường**: [CNTT-1501]
