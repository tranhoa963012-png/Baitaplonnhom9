import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier

# 1. Đọc dữ liệu chuẩn
df = pd.read_csv("Financial Statement Anomaly Dataset.csv")
df.fillna(0, inplace=True) 

# Bộ tên cột chuẩn để cả ML và Web cùng hiểu
cols = ["Total_Assets", "Total_Liabilities", "Revenue", "Operating_Expenses", 
        "Net_Income", "Cash_Flow_Operating", "Current_Ratio", "Debt_to_Equity"]

X = df[cols]
y = df['Financial_Status']

# 2. Huấn luyện
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X, y)
joblib.dump(model, "financial_model.pkl")

print("=> Đã huấn luyện xong! File 'financial_model.pkl' đã sẵn sàng.")