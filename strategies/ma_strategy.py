import vectorbt as vbt
import datetime as dt
import pandas as pd
import numpy as np

class MovingAverageStrategy:
    """
    Chiến lược Moving Average Crossover sử dụng VectorBT
    Mua khi MA ngắn hạn cắt lên MA dài hạn, bán khi ngược lại
    """
    
    def __init__(self, symbol='SPY', fast_window=20, slow_window=50, 
                 initial_cash=10000, commission=0.001):
        self.symbol = symbol
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.initial_cash = initial_cash
        self.commission = commission
        self.portfolio = None
        self.price_data = None
        
    def load_data(self, start_days=730):
        """Tải dữ liệu từ Yahoo Finance"""
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=start_days)
        
        print(f"Tải dữ liệu {self.symbol} từ {start_date.date()}")
        
        try:
            # Sử dụng VectorBT để tải dữ liệu
            self.price_data = vbt.YFData.download(
                self.symbol, 
                start=start_date, 
                end=end_date,
                interval='1d'
            ).get('Close')
            
            print(f"Tải thành công {len(self.price_data)} ngày dữ liệu")
            return True
            
        except Exception as e:
            print(f"Lỗi tải dữ liệu: {e}")
            return False
    
    def calculate_signals(self):
        """Tính toán tín hiệu mua/bán"""
        if self.price_data is None:
            raise ValueError("Chưa tải dữ liệu. Chạy load_data() trước.")
        
        # Tính Moving Average
        self.fast_ma = vbt.MA.run(self.price_data, window=self.fast_window, short_name=f'MA{self.fast_window}')
        self.slow_ma = vbt.MA.run(self.price_data, window=self.slow_window, short_name=f'MA{self.slow_window}')
        
        # Tạo tín hiệu
        self.entries = self.fast_ma.ma_crossed_above(self.slow_ma)  # Golden Cross
        self.exits = self.fast_ma.ma_crossed_below(self.slow_ma)    # Death Cross
        
        print(f"Số tín hiệu mua: {self.entries.sum()}")
        print(f"Số tín hiệu bán: {self.exits.sum()}")
        
    def run_backtest(self):
        """Chạy backtest"""
        self.calculate_signals()
        
        # Tạo portfolio với VectorBT
        self.portfolio = vbt.Portfolio.from_signals(
            close=self.price_data,
            entries=self.entries,
            exits=self.exits,
            init_cash=self.initial_cash,
            fees=self.commission,
            freq='1D'
        )
        
        print("Backtest hoàn tất!")
        
    def get_performance_metrics(self):
        """Lấy các chỉ số hiệu suất"""
        if self.portfolio is None:
            raise ValueError("Chưa chạy backtest. Chạy run_backtest() trước.")
        
        stats = {
            'Total Return': f"{self.portfolio.total_return():.2%}",
            'Sharpe Ratio': f"{self.portfolio.sharpe_ratio():.2f}",
            'Max Drawdown': f"{self.portfolio.max_drawdown():.2%}",
            'Win Rate': f"{self.portfolio.trades.win_rate():.2%}",
            'Total Trades': self.portfolio.trades.count,
            'Avg Trade Return': f"{self.portfolio.trades.returns.mean():.2%}",
            'Final Value': f"${self.portfolio.value().iloc[-1]:,.2f}"
        }
        
        return stats
    
    def print_results(self):
        """In kết quả chi tiết"""
        stats = self.get_performance_metrics()
        
        print("\n" + "="*50)
        print(f"KẾT QUẢ MOVING AVERAGE STRATEGY ({self.symbol})")
        print(f"MA Fast: {self.fast_window}, MA Slow: {self.slow_window}")
        print("="*50)
        
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        print("="*50)
    
    def plot_results(self):
        """Vẽ biểu đồ kết quả"""
        if self.portfolio is None:
            raise ValueError("Chưa chạy backtest")
            
        # Plot với VectorBT
        fig = self.portfolio.plot(
            title=f'{self.symbol} Moving Average Strategy Results',
            subplots=['price', 'entries', 'exits', 'pnl']
        )
        fig.show()

# Hàm tiện ích để chạy nhanh
def quick_ma_test(symbol='AAPL', fast=20, slow=50):
    """Chạy test nhanh MA strategy"""
    strategy = MovingAverageStrategy(symbol=symbol, fast_window=fast, slow_window=slow)
    
    if strategy.load_data():
        strategy.run_backtest()
        strategy.print_results()
        return strategy
    else:
        print("Không thể tải dữ liệu")
        return None

# Test chạy strategy
if __name__ == "__main__":
    print("Testing Moving Average Strategy...")
    strategy = quick_ma_test('AAPL', 20, 50)
    
    if strategy:
        # Uncomment để xem biểu đồ
        # strategy.plot_results()
        pass
