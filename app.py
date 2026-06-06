import joblib
from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)

# Tải mô hình học máy và bẫy lỗi an toàn
try:
    model = joblib.load("financial_model.pkl")
except Exception as e:
    print(f"Cảnh báo: Chưa tìm thấy file mô hình .pkl ({e}). Hệ thống sẽ dùng logic cây quyết định dự phòng.")
    model = None

def get_data():
    # Đọc file dữ liệu đầu vào
    df = pd.read_csv("Financial Statement Anomaly Dataset.csv")
    df.fillna(0, inplace=True)
    return df

@app.route('/')
def index():
    try:
        df = get_data()
        # Thống kê tổng số doanh nghiệp trong bộ dữ liệu để hiển thị lên thẻ thống kê
        total_companies = len(df)
        return render_template('index.html', companies=df.index.tolist(), total_companies=total_companies)
    except Exception as e:
        return f"Lỗi khởi động trang chủ: {str(e)}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy ID doanh nghiệp từ form (bọc lót nhiều tên biến tránh lỗi cache)
        company_idx_raw = request.form.get('company_index') or request.form.get('company') or request.form.get('id')
        if company_idx_raw is None:
            return "Lỗi: Không nhận được thông tin ID doanh nghiệp. Vui lòng quay lại trang chủ và thử lại."

        idx = int(company_idx_raw)
        df = get_data()
        
        if idx >= len(df) or idx < 0:
            return f"Lỗi: Doanh nghiệp với ID {idx} không tồn tại trong hệ thống."
            
        row = df.iloc[idx]
        
        # 1. Trích xuất chỉ số để dự đoán và vẽ biểu đồ
        try: debt_to_equity = round(float(row.get('Debt_to_Equity', 0)), 2)
        except: debt_to_equity = 0.0
            
        try: current_ratio = round(float(row.get('Current_Ratio', 0)), 2)
        except: current_ratio = 0.0

        try: roa = round(float(row.get('ROA', row.get('Net_Income', 0)/row.get('Total_Assets', 1) * 100)), 2)
        except: roa = 0.0

        try: roe = round(float(row.get('ROE', row.get('Net_Income', 0)/row.get('Total_Liabilities', 1) * 100)), 2)
        except: roe = 0.0

        # 2. Chạy mô hình dự đoán AI (Decision Tree)
        cols = ["Total_Assets", "Total_Liabilities", "Revenue", "Operating_Expenses", "Net_Income", "Cash_Flow_Operating", "Current_Ratio", "Debt_to_Equity"]
        if model:
            features = df[cols].iloc[[idx]]
            prediction = model.predict(features)[0]
            status = "Bình thường" if prediction == 'Normal' else "Rủi ro cao"
        else:
            # Thuật toán cây quyết định dự phòng nếu chưa có file .pkl chuyên dụng
            status = "Bình thường" if current_ratio >= 1.0 and debt_to_equity <= 2.0 else "Rủi ro cao"

        # 3. Tính toán AI Risk Score (%) & Tạo đánh giá/khuyến nghị tự động theo các chỉ số thực tế
        recommendations = []
        if status == "Bình thường":
            # Tính toán điểm rủi ro nền thấp dựa trên sự mất cân đối nhẹ nếu có
            risk_score = int(min(45, max(10, (debt_to_equity * 15) + (20 if current_ratio < 1.5 else 0))))
            assessment = "Doanh nghiệp có cơ cấu tài chính ổn định. Các chỉ số thanh khoản và nợ phải trả nằm trong phạm vi kiểm soát an toàn của mô hình Decision Tree."
            recommendations.append("Duy trì cấu trúc vốn hiện tại và tiếp tục tối ưu hóa dòng tiền hoạt động.")
            recommendations.append("Theo dõi biến động thị trường để đưa ra kế hoạch tái đầu tư hợp lý.")
        else:
            # Rủi ro cao: Điểm rủi ro cao từ 60% - 98%
            risk_score = int(min(98, max(60, (debt_to_equity * 20) + (30 if current_ratio < 1.0 else 10))))
            assessment = "Hệ thống phát hiện dấu hiệu bất thường nghiêm trọng. Doanh nghiệp rơi vào nhóm rủi ro cao chủ yếu do mất cân đối dòng tiền, tỷ lệ nợ vượt ngưỡng an toàn hoặc khả năng thanh toán ngắn hạn bị suy giảm sâu."
            
            # Đưa ra các khuyến nghị hành động thực tế dựa trên chỉ số lỗi
            if debt_to_equity > 2.0:
                recommendations.append("Cần lên kế hoạch giảm tỷ lệ nợ (Debt to Equity) bằng cách cơ cấu lại các khoản vay ngắn hạn và dài hạn.")
            if current_ratio < 1.0:
                recommendations.append("Cấp bách gia tăng tài sản ngắn hạn hoặc bổ sung nguồn vốn lưu động để nâng cao tỷ số thanh khoản (Current Ratio) lên mức an toàn ≥ 1.0.")
            if float(row.get('Net_Income', 0)) < 0:
                recommendations.append("Thắt chặt và kiểm soát chặt chẽ chi phí hoạt động (Operating Expenses) để cải thiện chỉ số thu nhập ròng đang bị âm.")
            if float(row.get('Cash_Flow_Operating', 0)) < 0:
                recommendations.append("Cải thiện chu kỳ thu tiền khách hàng và rà soát các hoạt động kinh doanh cốt lõi để kéo dòng tiền hoạt động về mức dương.")
            if len(recommendations) < 3:
                recommendations.append("Thực hiện rà soát, kiểm toán nội bộ đối với toàn bộ các tài khoản báo cáo tài chính có biến động đột biến.")

        # 4. Định dạng tiền tệ cho bảng số liệu chi tiết ở Frontend
        info_formatted = {}
        for col in df.columns:
            try:
                val = float(row[col])
                info_formatted[col] = "{:,.2f}".format(val)
            except:
                info_formatted[col] = str(row[col])

        return render_template('result.html', 
                               status=status, 
                               idx=idx,
                               info=info_formatted, 
                               risk_score=risk_score,
                               assessment=assessment,
                               recommendations=recommendations,
                               debt_to_equity=debt_to_equity,
                               current_ratio=current_ratio,
                               roa=roa,
                               roe=roe)
    except Exception as e:
        return f"Lỗi hệ thống tại trang kết quả: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)