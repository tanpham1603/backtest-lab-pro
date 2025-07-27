import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib
import os

# 1. Lấy dữ liệu và tạo đặc trưng (features)
print("Đang tải dữ liệu từ Yahoo Finance...")
data = yf.download("SPY", start="2018-01-01")
print("Tải dữ liệu hoàn tất.")

# --- KIỂM TRA DỮ LIỆU ---
# Kiểm tra xem dữ liệu có được tải về thành công không
if data.empty:
    print("Lỗi: Không tải được dữ liệu. Vui lòng kiểm tra lại mã cổ phiếu hoặc kết nối mạng.")
else:
    # --- TOÀN BỘ LOGIC XỬ LÝ SẼ NẰM TRONG KHỐI ELSE NÀY ---
    
    # --- TÍNH TOÁN RSI BẰNG PANDAS ---
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(com=14 - 1, min_periods=14).mean()
    avg_loss = loss.ewm(com=14 - 1, min_periods=14).mean()

    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    # -----------------------------------------------------------

    data["MA20"] = data["Close"].rolling(20).mean()

    # 2. Tạo nhãn (label)
    data["label"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

    # Xóa các dòng có giá trị NaN
    data.dropna(inplace=True)

    # 3. Chuẩn bị dữ liệu X và y
    X = data[["RSI", "MA20"]]
    y = data["label"]

    # 4. Chia dữ liệu thành 80% cho training và 20% cho testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # 5. Huấn luyện mô hình
    print("Đang huấn luyện mô hình...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    print("Huấn luyện hoàn tất.")

    # 6. Đánh giá mô hình
    print("\n--- Báo cáo hiệu suất trên dữ liệu thực tế (Test Set) ---")
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

    # 7. Lưu lại mô hình
    output_dir = "app/ml_signals"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "rf_signal.pkl")

    joblib.dump(model, model_path)
    print(f"\nĐã lưu mô hình vào file: {model_path}")

