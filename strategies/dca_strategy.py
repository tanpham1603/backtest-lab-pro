import backtrader as bt
from datetime import datetime
import pandas as pd

class DCAStrategy(bt.Strategy):
    """
    Chiến lược Dollar Cost Averaging (DCA)
    Đầu tư một số tiền cố định theo chu kỳ đều đặn
    """
    params = dict(
        investment_amount=1000,  # Số tiền đầu tư mỗi lần
        buy_frequency=30,        # Mua mỗi 30 ngày (hàng tháng)
        verbose=True             # In log chi tiết
    )
    
    def __init__(self):
        # Khởi tạo biến tracking
        self.day_counter = 0
        self.total_invested = 0
        self.total_shares = 0
        self.buy_dates = []
        self.buy_prices = []
        
    def log(self, txt, dt=None):
        """Logging function để theo dõi giao dịch"""
        if self.p.verbose:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}: {txt}')
    
    def next(self):
        """Logic chạy mỗi ngày"""
        self.day_counter += 1
        
        # Kiểm tra nếu đến ngày mua (mỗi buy_frequency ngày)
        if self.day_counter % self.p.buy_frequency == 0:
            current_price = self.data.close[0]
            shares_to_buy = self.p.investment_amount / current_price
            
            # Đặt lệnh mua
            order = self.buy(size=shares_to_buy)
            
            # Cập nhật tracking
            self.total_invested += self.p.investment_amount
            self.total_shares += shares_to_buy
            self.buy_dates.append(self.datas[0].datetime.date(0))
            self.buy_prices.append(current_price)
            
            self.log(f'DCA BUY: {shares_to_buy:.2f} cổ phiếu với giá ${current_price:.2f}')
    
    def stop(self):
        """Kết thúc backtest - tính toán kết quả"""
        final_price = self.data.close[0]
        portfolio_value = self.total_shares * final_price
        profit_loss = portfolio_value - self.total_invested
        profit_percent = (profit_loss / self.total_invested) * 100 if self.total_invested > 0 else 0
        
        self.log('=' * 50)
        self.log('KẾT QUẢ DCA STRATEGY')
        self.log(f'Tổng số tiền đầu tư: ${self.total_invested:,.2f}')
        self.log(f'Tổng số cổ phiếu: {self.total_shares:.2f}')
        self.log(f'Giá cuối kỳ: ${final_price:.2f}')
        self.log(f'Giá trị danh mục: ${portfolio_value:,.2f}')
        self.log(f'Lãi/Lỗ: ${profit_loss:,.2f} ({profit_percent:.2f}%)')
        self.log(f'Số lần mua: {len(self.buy_dates)}')
        self.log('=' * 50)

def run_dca_backtest(symbol='AAPL', start_date='2023-01-01', end_date='2024-01-01'):
    """
    Hàm chạy backtest DCA strategy
    """
    import yfinance as yf
    
    print(f"Bắt đầu backtest DCA cho {symbol}")
    
    # 1. Tạo Cerebro engine
    cerebro = bt.Cerebro()
    
    # 2. Thêm strategy
    cerebro.addstrategy(DCAStrategy, 
                       investment_amount=1000,
                       buy_frequency=30)
    
    # 3. Tải dữ liệu
    data_df = yf.download(symbol, start=start_date, end=end_date)
    data_df.columns = [col[0].lower() for col in data_df.columns]
    data = bt.feeds.PandasData(dataname=data_df)
    cerebro.adddata(data)
    
    # 4. Thiết lập broker (môi giới)
    cerebro.broker.setcash(100000)  # Vốn ban đầu $100K
    cerebro.broker.setcommission(commission=0.001)  # Phí 0.1%
    
    # 5. Chạy backtest
    print(f'Vốn ban đầu: ${cerebro.broker.getvalue():,.2f}')
    results = cerebro.run()
    print(f'Vốn cuối kỳ: ${cerebro.broker.getvalue():,.2f}')
    
    # 6. Vẽ biểu đồ (tùy chọn)
    # cerebro.plot(style='candlestick')
    
    return results

# Test chạy strategy
if __name__ == "__main__":
    run_dca_backtest('AAPL', '2023-01-01', '2024-07-01')
