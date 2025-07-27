from strategies.dca_strategy import run_dca_backtest
from strategies.ma_strategy import quick_ma_test
from data.data_loader import DataLoader
import pandas as pd

def compare_strategies(symbol='AAPL', start_date='2023-01-01'):
    """So sánh DCA vs Moving Average strategy"""
    
    print(f"\n🔍 SO SÁNH CHIẾN LƯỢC CHO {symbol}")
    print("="*60)
    
    # 1. Chạy DCA Strategy
    print("\n📈 1. CHIẾN LƯỢC DCA (Dollar Cost Averaging)")
    print("-" * 40)
    dca_results = run_dca_backtest(symbol, start_date)
    
    # 2. Chạy Moving Average Strategy  
    print("\n📊 2. CHIẾN LƯỢC MOVING AVERAGE")
    print("-" * 40)
    ma_strategy = quick_ma_test(symbol, fast=20, slow=50)
    
    # 3. Tổng kết so sánh
    print("\n🏆 TỔNG KẾT SO SÁNH")
    print("="*60)
    print("DCA: Đầu tư đều đặn, phù hợp nhà đầu tư dài hạn")
    print("MA:  Giao dịch theo tín hiệu, phù hợp trader ngắn hạn")
    print("="*60)

if __name__ == "__main__":
    # Test với nhiều mã khác nhau
    symbols = ['AAPL', 'SPY', 'GOOGL']
    
    for symbol in symbols:
        try:
            compare_strategies(symbol)
            print("\n" + "🎯"*20 + "\n")
        except Exception as e:
            print(f"Lỗi với {symbol}: {e}")
