from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.common.exceptions import APIError

class AlpacaTrader:
    """
    Lớp (class) này đóng vai trò là một "wrapper" - một công cụ chuyên dụng 
    để tương tác với Alpaca Trading API một cách đơn giản và có tổ chức.
    Nó sử dụng thư viện chính thức và mới nhất: alpaca-py.
    """
    def __init__(self, api_key, api_secret, paper=True):
        """
        Khởi tạo kết nối đến Alpaca.
        
        Args:
            api_key (str): Your Alpaca API Key.
            api_secret (str): Your Alpaca API Secret.
            paper (bool): True để kết nối đến môi trường paper trading (tiền ảo).
        """
        try:
            self.client = TradingClient(api_key, api_secret, paper=paper)
            self.account = self.client.get_account()
            self.connected = True
            print("✅ Kết nối đến Alpaca thành công!")
        except APIError as e:
            print(f"❌ Lỗi kết nối Alpaca: {e}")
            self.client = None
            self.account = None
            self.connected = False

    def get_account_info(self):
        """Lấy thông tin tài khoản."""
        if not self.connected:
            return None
        # .__dict__ để chuyển đối tượng account thành dictionary dễ sử dụng
        return self.account.__dict__

    def place_market_order(self, symbol, qty, side="buy"):
        """
        Đặt một lệnh thị trường (Market Order).
        
        Args:
            symbol (str): Mã cổ phiếu, ví dụ: "SPY".
            qty (float): Số lượng cổ phiếu.
            side (str): "buy" hoặc "sell".
        """
        if not self.connected:
            return None
            
        market_order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.GTC  # Good 'til Canceled
        )
        
        try:
            market_order = self.client.submit_order(order_data=market_order_data)
            print(f"Đã đặt lệnh Market {side.upper()} {qty} {symbol}.")
            return market_order.__dict__
        except APIError as e:
            print(f"Lỗi khi đặt lệnh Market: {e}")
            return None

    def get_all_positions(self):
        """Lấy danh sách tất cả các vị thế đang mở."""
        if not self.connected:
            return []
        positions = self.client.get_all_positions()
        # Chuyển danh sách các đối tượng position thành danh sách các dictionary
        return [p.__dict__ for p in positions]

    def get_all_orders(self):
        """Lấy danh sách tất cả các lệnh (đã khớp, đang chờ, đã hủy)."""
        if not self.connected:
            return []
            
        request_params = GetOrdersRequest(status=QueryOrderStatus.ALL)
        orders = self.client.get_orders(filter=request_params)
        # Chuyển danh sách các đối tượng order thành danh sách các dictionary
        return [o.__dict__ for o in orders]

