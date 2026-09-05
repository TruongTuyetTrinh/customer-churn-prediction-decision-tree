import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, recall_score
import matplotlib.pyplot as plt
import sys
import os
# Cấu hình đầu ra để hiển thị tiếng Việt chính xác
sys.stdout.reconfigure(encoding='utf-8')

# Định nghĩa đường dẫn file
FILE_PATH = r"processed_customer_churn.csv" 

PLOT_PATH = r"EDA_plots\dt_churn_model_final.png" 
os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)

# 1. TẢI DỮ LIỆU VÀ CHỌN BIẾN

try:
    data = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file tại đường dẫn {FILE_PATH}")
    sys.exit()

# Chọn 10 biến định lượng đã phân tích
FEATURES = ['Age', 'Annual_Income', 'Total_Spend', 'Years_as_Customer', 'Num_of_Purchases',
            'Average_Transaction_Amount', 'Num_of_Returns', 'Num_of_Support_Contacts',
            'Satisfaction_Score', 'Last_Purchase_Days_Ago']
TARGET = 'Target_Churn'

X = data[FEATURES]
y = data[TARGET]

# 2. CHIA DỮ LIỆU (Giữ tỷ lệ Churn đều ở tập train/test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# 3. GRID SEARCH VÀ TỐI ƯU HÓA (Mục tiêu: Tăng khả năng phát hiện nguy cơ - RECALL)
# Sử dụng class_weight='balanced' để mô hình không bỏ qua lớp thiểu số (nếu có)
dt_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')

# Phạm vi tham số để tìm kiếm (tối ưu hóa chống overfitting và tăng Recall)
param_grid = {
    'criterion': ['gini', 'entropy'],
    # max_depth được giới hạn để cây không quá phức tạp (chống overfitting)
    'max_depth': [5, 7, 9], 
    # Giới hạn kích thước nút lá tối thiểu
    'min_samples_leaf': [10, 20], 
    # Giới hạn kích thước phân tách tối thiểu
    'min_samples_split': [20, 40] 
}

# Tối ưu hóa cho RECALL
print("Bắt đầu Grid Search (Tối ưu hóa cho Recall)...")
grid = GridSearchCV(dt_model, param_grid, cv=5, scoring='recall', verbose=0)
grid.fit(X_train, y_train)

# 4. ĐÁNH GIÁ MÔ HÌNH VÀ KẾT QUẢ
best_dt = grid.best_estimator_
y_pred = best_dt.predict(X_test)

print("\n" + "="*50)
print("=== KẾT QUẢ MÔ HÌNH DECISION TREE TỐI ƯU CHO CHURN ===")
print("="*50)
print("Best params (Tham số tối ưu):", grid.best_params_)
print(f"CV Score (Recall tối ưu): {grid.best_score_:.4f}")
print(f"Test Accuracy (Độ chính xác Test): {accuracy_score(y_test, y_pred):.4f}")
print(f"Test Recall (Khả năng phát hiện Churn): {recall_score(y_test, y_pred):.4f}")
print("\nBÁO CÁO PHÂN LOẠI CHI TIẾT:")
print(classification_report(y_test, y_pred, target_names=['Trung thành', 'Rời bỏ']))

# 5. VẼ CÂY QUYẾT ĐỊNH
# =========================================================================
plt.figure(figsize=(25, 15))
plot_tree(best_dt, feature_names=X.columns, class_names=['Trung thành', 'Rời bỏ'],
          filled=True, rounded=True, fontsize=8)
plt.title(f'Cây Quyết định Dự đoán Churn (Recall: {recall_score(y_test, y_pred):.2f})')
plt.savefig(PLOT_PATH, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nHoàn tất. Cây quyết định đã được lưu vào {PLOT_PATH}")