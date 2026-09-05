# Ứng Dụng Thuật Toán Decision Tree Trong Dự Đoán Khả Năng Rời Bỏ Của Khách Hàng Bán Lẻ

## 📌 1. Tổng Quan Đề Tài (Project Overview)
Trong thị trường bán lẻ cạnh tranh cao, chi phí thu hút khách hàng mới luôn lớn hơn chi phí giữ chân khách hàng hiện tại. Dự án này ứng dụng thuật toán học máy **Cây quyết định (Decision Tree)** nhằm:
- Dự đoán khả năng rời bỏ (Customer Churn) của khách hàng dựa trên lịch sử giao dịch và hành vi tiêu dùng.
- Khám phá các nhân tố cốt lõi và ngưỡng hành vi dẫn đến quyết định ngưng sử dụng dịch vụ.
- Đề xuất các giải pháp sản phẩm/chính sách nhằm tối ưu tỷ lệ giữ chân (**Retention Rate**) và nâng cao giá trị vòng đời khách hàng (**LTV**).

---

## 🛠 2. Công Nghệ & Thư Viện Sử Dụng (Tech Stack)
- **Ngôn ngữ:** Python 3.x
- **Xử lý & Phân tích dữ liệu:** `pandas`, `numpy`
- **Mô hình học máy (Machine Learning):** `scikit-learn` (DecisionTreeClassifier)
- **Trực quan hóa:** `matplotlib`, `seaborn`

---

## ⚙️ 3. Quy Trình Thực Hiện (Workflow)

```text
[Dữ liệu thô] 
   └──> [Tiền xử lý & Feature Engineering] 
           └──> [Huấn luyện mô hình Decision Tree] 
                   └──> [Đánh giá độ chính xác & Trích xuất quy tắc] 
                           └──> [Đề xuất chiến lược sản phẩm]
