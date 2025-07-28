import pandas as pd
import pandas_ta as ta

class TechnicalIndicators:
    def __init__(self, data):
        self.data = data
        self.close = data["Close"]
        self.open = data["Open"]
        self.high = data["High"]
        self.low = data["Low"]
        self.volume = data["Volume"]

    def rsi(self, period=14):
        return ta.rsi(self.close, length=period)

    def macd(self, fast=12, slow=26, signal=9):
        return ta.macd(self.close, fast=fast, slow=slow, signal=signal)

    def bollinger_bands(self, period=20, std=2):
        return ta.bbands(self.close, length=period, std=std)

    def stochastic(self, k_period=14, d_period=3):
        return ta.stoch(high=self.high, low=self.low, close=self.close, k=k_period, d=d_period)

    def atr(self, period=14):
        return ta.atr(high=self.high, low=self.low, close=self.close, length=period)

    def williams_r(self, period=14):
        return ta.willr(high=self.high, low=self.low, close=self.close, length=period)

    def cci(self, period=20):
        return ta.cci(high=self.high, low=self.low, close=self.close, length=period)
