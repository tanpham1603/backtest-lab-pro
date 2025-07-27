import numpy as np
import pandas as pd
import vectorbt as vbt
import talib

class TechnicalIndicators:
    """Lớp tính toán các chỉ báo kỹ thuật"""
    
    def __init__(self, data):
        self.data = data
        self.high = data['High'] if 'High' in data.columns else data['Close']
        self.low = data['Low'] if 'Low' in data.columns else data['Close'] 
        self.close = data['Close']
        self.volume = data['Volume'] if 'Volume' in data.columns else None
        
    def sma(self, period=20):
        """Simple Moving Average"""
        return self.close.rolling(window=period).mean()
    
    def ema(self, period=20):
        """Exponential Moving Average"""
        return self.close.ewm(span=period).mean()
    
    def rsi(self, period=14):
        """Relative Strength Index"""
        try:
            # Use talib if available
            return pd.Series(talib.RSI(self.close.values, timeperiod=period), 
                           index=self.close.index)
        except:
            # Fallback calculation
            delta = self.close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
    
    def macd(self, fast=12, slow=26, signal=9):
        """MACD Oscillator"""
        try:
            macd_line, macd_signal, macd_hist = talib.MACD(
                self.close.values, fastperiod=fast, 
                slowperiod=slow, signalperiod=signal
            )
            return pd.DataFrame({
                'MACD': pd.Series(macd_line, index=self.close.index),
                'Signal': pd.Series(macd_signal, index=self.close.index),
                'Histogram': pd.Series(macd_hist, index=self.close.index)
            })
        except:
            # Fallback calculation
            ema_fast = self.ema(fast)
            ema_slow = self.ema(slow)
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            return pd.DataFrame({
                'MACD': macd_line,
                'Signal': signal_line,
                'Histogram': histogram
            })
    
    def bollinger_bands(self, period=20, std=2):
        """Bollinger Bands"""
        sma = self.sma(period)
        rolling_std = self.close.rolling(window=period).std()
        
        return pd.DataFrame({
            'Middle': sma,
            'Upper': sma + (rolling_std * std),
            'Lower': sma - (rolling_std * std),
            'Width': (sma + (rolling_std * std)) - (sma - (rolling_std * std)),
            'Percent_B': (self.close - (sma - (rolling_std * std))) / 
                        ((sma + (rolling_std * std)) - (sma - (rolling_std * std)))
        })
    
    def stochastic(self, k_period=14, d_period=3):
        """Stochastic Oscillator"""
        try:
            slowk, slowd = talib.STOCH(
                self.high.values, self.low.values, self.close.values,
                fastk_period=k_period, slowk_period=d_period, 
                slowk_matype=0, slowd_period=d_period, slowd_matype=0
            )
            return pd.DataFrame({
                'K': pd.Series(slowk, index=self.close.index),
                'D': pd.Series(slowd, index=self.close.index)
            })
        except:
            # Fallback calculation
            lowest_low = self.low.rolling(window=k_period).min()
            highest_high = self.high.rolling(window=k_period).max()
            k_percent = 100 * ((self.close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()
            
            return pd.DataFrame({
                'K': k_percent,
                'D': d_percent
            })
    
    def atr(self, period=14):
        """Average True Range"""
        try:
            atr_values = talib.ATR(self.high.values, self.low.values, 
                                 self.close.values, timeperiod=period)
            return pd.Series(atr_values, index=self.close.index)
        except:
            # Fallback calculation
            high_low = self.high - self.low
            high_close_prev = np.abs(self.high - self.close.shift())
            low_close_prev = np.abs(self.low - self.close.shift())
            
            tr = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
            return tr.rolling(window=period).mean()
    
    def williams_r(self, period=14):
        """Williams %R"""
        try:
            willr = talib.WILLR(self.high.values, self.low.values, 
                              self.close.values, timeperiod=period)
            return pd.Series(willr, index=self.close.index)
        except:
            # Fallback calculation
            highest_high = self.high.rolling(window=period).max()
            lowest_low = self.low.rolling(window=period).min()
            return -100 * ((highest_high - self.close) / (highest_high - lowest_low))
    
    def cci(self, period=20):
        """Commodity Channel Index"""
        try:
            cci_values = talib.CCI(self.high.values, self.low.values, 
                                 self.close.values, timeperiod=period)
            return pd.Series(cci_values, index=self.close.index)
        except:
            # Fallback calculation
            typical_price = (self.high + self.low + self.close) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mad = typical_price.rolling(window=period).apply(
                lambda x: np.abs(x - x.mean()).mean()
            )
            return (typical_price - sma_tp) / (0.015 * mad)

class AdvancedIndicators:
    """Các chỉ báo nâng cao và tùy chỉnh"""
    
    def __init__(self, data):
        self.indicators = TechnicalIndicators(data)
        self.data = data
        
    def multi_timeframe_rsi(self, periods=[14, 28, 50]):
        """RSI đa khung thời gian"""
        rsi_data = {}
        for period in periods:
            rsi_data[f'RSI_{period}'] = self.indicators.rsi(period)
        return pd.DataFrame(rsi_data)
    
    def rsi_divergence_signals(self, period=14):
        """Tín hiệu phân kỳ RSI"""
        rsi = self.indicators.rsi(period)
        price = self.data['Close']
        
        # Tìm peaks và troughs
        from scipy.signal import argrelextrema
        
        price_peaks = argrelextrema(price.values, np.greater, order=5)
        price_troughs = argrelextrema(price.values, np.less, order=5)
        rsi_peaks = argrelextrema(rsi.values, np.greater, order=5)
        rsi_troughs = argrelextrema(rsi.values, np.less, order=5)
        
        signals = pd.Series(0, index=price.index)
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        # Bearish divergence: price makes higher high, RSI makes lower high
        
        return signals
    
    def composite_momentum(self):
        """Momentum tổng hợp từ nhiều chỉ báo"""
        rsi = self.indicators.rsi(14)
        macd = self.indicators.macd()
        stoch = self.indicators.stochastic()
        williams = self.indicators.williams_r(14)
        
        # Chuẩn hóa các chỉ báo về scale 0-100
        rsi_norm = rsi
        macd_norm = ((macd['MACD'] - macd['MACD'].rolling(50).min()) / 
                    (macd['MACD'].rolling(50).max() - macd['MACD'].rolling(50).min())) * 100
        stoch_norm = stoch['K']
        williams_norm = (williams + 100)  # Williams %R is negative
        
        # Tính trung bình có trọng số
        composite = (rsi_norm * 0.3 + macd_norm * 0.25 + 
                    stoch_norm * 0.25 + williams_norm * 0.2)
        
        return composite
    
    def trend_strength(self, period=20):
        """Độ mạnh xu hướng"""
        close = self.data['Close']
        sma = close.rolling(period).mean()
        
        # Tính độ lệch chuẩn
        std = close.rolling(period).std()
        
        # Trend strength = (price - sma) / std
        trend_strength = (close - sma) / std
        
        return trend_strength
    
    def volatility_bands(self, period=20, multiplier=2):
        """Dải biến động dựa trên ATR"""
        atr = self.indicators.atr(period)
        close = self.data['Close']
        sma = close.rolling(period).mean()
        
        upper_band = sma + (atr * multiplier)
        lower_band = sma - (atr * multiplier)
        
        return pd.DataFrame({
            'Middle': sma,
            'Upper': upper_band,
            'Lower': lower_band,
            'ATR': atr
        })
