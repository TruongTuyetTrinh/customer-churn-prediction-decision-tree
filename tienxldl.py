import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Đọc file dữ liệu
data = pd.read_csv(r"online_retail_customer_churn.csv")

# Bước 1: Xác định vấn đề của dữ liệu
print("=== BƯỚC 1: XÁC ĐỊNH VẤN ĐỀ DỮ LIỆU ===")

# 1.1. Kiểu dữ liệu
print("\n1.1. Kiểu dữ liệu của các cột:")
print(data.dtypes)

# 1.2. Thông tin chi tiết (bao gồm kiểu dữ liệu và giá trị thiếu)
print("\n1.2. Thông tin chi tiết về dữ liệu:")
print(data.info())

# 1.3. Giá trị thiếu
print("\n1.3. Giá trị thiếu trong mỗi cột:")
print(data.isnull().sum())

# 1.4. Bản sao
duplicates = data.duplicated().sum()
print("\n1.4. Số lượng bản sao (duplicates):", duplicates)

# 1.5. Giá trị bất thường (ví dụ: giá trị âm hoặc ngoài phạm vi)
print("\n1.5. Giá trị bất thường:")
print("Age < 18 hoặc > 80:", ((data['Age'] < 18) | (data['Age'] > 80)).sum())
print("Annual_Income < 0:", (data['Annual_Income'] < 0).sum())
print("Total_Spend < 0:", (data['Total_Spend'] < 0).sum())
print("Average_Transaction_Amount < 0:", (data['Average_Transaction_Amount'] < 0).sum())
print("Num_of_Returns < 0:", (data['Num_of_Returns'] < 0).sum())
print("Num_of_Support_Contacts < 0:", (data['Num_of_Support_Contacts'] < 0).sum())
print("Satisfaction_Score ngoài 1-5:", ((data['Satisfaction_Score'] < 1) | (data['Satisfaction_Score'] > 5)).sum())
print("Last_Purchase_Days_Ago < 0 hoặc > 365:", ((data['Last_Purchase_Days_Ago'] < 0) | (data['Last_Purchase_Days_Ago'] > 365)).sum())

# 1.6. Kiểm tra ngoại lai bằng IQR
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    return len(outliers), lower_bound, upper_bound

print("\n1.6. Kiểm tra ngoại lai (outliers) bằng IQR:")
columns = ['Age', 'Annual_Income', 'Total_Spend', 'Average_Transaction_Amount', 'Satisfaction_Score', 'Last_Purchase_Days_Ago']
for col in columns:
    count, lower, upper = detect_outliers(data, col)
    print(f"{col}: {count} ngoại lai (giới hạn: {lower:.2f} - {upper:.2f})")

# 1.7. Phân bố lớp Target_Churn (kiểm tra mất cân bằng)
print("\n1.7. Phân bố Target_Churn (kiểm tra mất cân bằng):")
print(data['Target_Churn'].value_counts(normalize=True))

# 1.8. Giá trị duy nhất trong biến phân loại
print("\n1.8. Giá trị duy nhất trong Gender:")
print(data['Gender'].unique())
print("\nGiá trị duy nhất trong Promotion_Response:")
print(data['Promotion_Response'].unique())

# 1.9. Thống kê mô tả để phát hiện bất thường thêm
print("\n1.9. Thống kê mô tả:")
print(data.describe())

# 1.10. Vẽ biểu đồ kiểm tra (boxplot để phát hiện ngoại lai)
plt.figure(figsize=(15, 6))
sns.boxplot(data=data[columns])
plt.title('Boxplot các biến quan trọng để phát hiện ngoại lai')
plt.savefig(r"E:\PTDLPYTHON\boxplot_outliers.png")
plt.show()

# Bước 2: Xử lý dữ liệu
print("\n=== BƯỚC 2: XỬ LÝ DỮ LIỆU ===")

# 2.1. Xử lý giá trị thiếu (dù không có)
for col in ['Annual_Income', 'Total_Spend', 'Average_Transaction_Amount']:
    data[col] = data[col].fillna(data[col].mean())
for col in ['Gender', 'Promotion_Response']:
    data[col] = data[col].fillna(data[col].mode()[0])

# 2.2. Xử lý bản sao
data = data.drop_duplicates()

# 2.3. Xử lý giá trị bất thường
data['Age'] = data['Age'].astype(float)
data.loc[data['Age'] < 18, 'Age'] = data['Age'].mean()
data.loc[data['Age'] > 80, 'Age'] = data['Age'].mean()
data['Age'] = data['Age'].astype(int)

for col in ['Annual_Income', 'Total_Spend', 'Average_Transaction_Amount']:
    data.loc[data[col] < 0, col] = data[col].mean()

data.loc[data['Satisfaction_Score'] < 1, 'Satisfaction_Score'] = 1
data.loc[data['Satisfaction_Score'] > 5, 'Satisfaction_Score'] = 5

data['Last_Purchase_Days_Ago'] = data['Last_Purchase_Days_Ago'].astype(float)
data.loc[data['Last_Purchase_Days_Ago'] < 0, 'Last_Purchase_Days_Ago'] = data['Last_Purchase_Days_Ago'].mean()
data.loc[data['Last_Purchase_Days_Ago'] > 365, 'Last_Purchase_Days_Ago'] = data['Last_Purchase_Days_Ago'].mean()
data['Last_Purchase_Days_Ago'] = data['Last_Purchase_Days_Ago'].astype(int)

# 2.4. Xử lý ngoại lai bằng IQR
for col in columns:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    data.loc[data[col] < lower, col] = lower
    data.loc[data[col] > upper, col] = upper

# 2.5. Mã hóa biến phân loại
data = pd.get_dummies(data, columns=['Gender', 'Promotion_Response'], dtype=int)
data['Email_Opt_In'] = data['Email_Opt_In'].astype(int)
data['Target_Churn'] = data['Target_Churn'].astype(int)

# 2.6. Loại bỏ cột không cần thiết
data = data.drop(['Customer_ID'], axis=1, errors='ignore')

# Lưu dữ liệu đã xử lý
data.to_csv(r"E:\PTDLPYTHON\processed_customer_churn.csv", index=False)

# Kiểm tra sau xử lý
print("\nDữ liệu sau xử lý (info):")
print(data.info())
print("\nPhân bố Target_Churn sau xử lý:")
print(data['Target_Churn'].value_counts(normalize=True))
print("\n5 dòng đầu sau xử lý:")
print(data.head())

print("\nDữ liệu đã xử lý lưu tại: processed_customer_churn.csv")