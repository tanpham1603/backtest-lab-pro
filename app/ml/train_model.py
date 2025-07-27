import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib
import os
import pandas_ta as ta # Import thư viện mới

# 1. Lấy dữ liệu và tạo đặc trưng (features)
print("Đang tải dữ liệu từ Yahoo Finance...")
data = yf.download("SPY", start="2018-01-01")
print("Tải dữ liệu hoàn tất.")

if not data.empty:
    # --- SỬA LỖI Ở ĐÂY: Dùng pandas-ta ---
    data.ta.rsi(length=14, append=True) # Tự động thêm cột 'RSI_14'
    data.ta.sma(length=20, append=True) # Tự động thêm cột 'SMA_20'
    
    # Đổi tên cột cho nhất quán
    data.rename(columns={'RSI_14': 'RSI', 'SMA_20': 'MA20'}, inplace=True)
    # ------------------------------------

    # 2. Tạo nhãn (label)
    data["label"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    data.dropna(inplace=True)

    # 3. Chuẩn bị dữ liệu
    X = data[["RSI", "MA20"]]
    y = data["label"]

    # 4. Chia dữ liệu
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
else:
    print("Lỗi: Không tải được dữ liệu.")
