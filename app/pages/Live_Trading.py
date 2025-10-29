import streamlit as st
import pandas as pd
import yfinance as yf
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import traceback
from datetime import datetime, timedelta
import requests
import json
import base64
import os
import pandas_ta as ta
import re
from textblob import TextBlob

# --- Page Configuration ---
st.set_page_config(
    page_title="🚀 Live Trading Pro", 
    page_icon="🛰️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    .header-gradient {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
    }
    .dashboard-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #667eea;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.2);
    }
    .position-group {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .position-item {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #00ff88;
    }
    .position-item.sell {
        border-left-color: #ff4444;
    }
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: white;
        margin: 0.5rem 0;
    }
    .metric-label {
        color: #8898aa;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .badge-stock { background: linear-gradient(135deg, #28a745, #7ae582); color: white; }
    .badge-crypto { background: linear-gradient(135deg, #f7931a, #ffc46c); color: black; }
    .badge-profit { background: linear-gradient(135deg, #00ff88, #00cc6a); color: white; }
    .badge-loss { background: linear-gradient(135deg, #ff4444, #cc0000); color: white; }
    .connection-panel {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 2rem;
    }
    .tab-container {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 15px;
        padding: 2rem;
        margin-top: 1rem;
    }
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
        border: none;
    }
    .sentiment-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .sentiment-card:hover {
        transform: translateY(-3px);
        border-color: #667eea;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
    }
    .sentiment-positive {
        border-left: 4px solid #00ff88;
    }
    .sentiment-negative {
        border-left: 4px solid #ff4444;
    }
    .sentiment-neutral {
        border-left: 4px solid #667eea;
    }
    .source-badge {
        background: rgba(255, 255, 255, 0.1);
        color: #8898aa;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.7rem;
        margin: 0 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="header-gradient">🚀 Live Trading Pro</div>', unsafe_allow_html=True)

# --- ACCOUNT MANAGER ---
class AccountManager:
    def __init__(self):
        self.accounts_file = "saved_accounts.json"
    
    def save_account(self, api_key, api_secret, nickname=""):
        try:
            accounts = self.load_accounts()
            encoded_key = base64.b64encode(api_key.encode()).decode()
            encoded_secret = base64.b64encode(api_secret.encode()).decode()
            
            account_id = f"alpaca_{nickname}_{int(time.time())}"
            accounts[account_id] = {
                'api_key': encoded_key,
                'api_secret': encoded_secret,
                'nickname': nickname,
                'created_at': datetime.now().isoformat()
            }
            
            with open(self.accounts_file, 'w') as f:
                json.dump(accounts, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Error saving account: {e}")
            return False
    
    def load_accounts(self):
        try:
            if os.path.exists(self.accounts_file):
                with open(self.accounts_file, 'r') as f:
                    return json.load(f)
            return {}
        except:
            return {}
    
    def delete_account(self, account_id):
        try:
            accounts = self.load_accounts()
            if account_id in accounts:
                del accounts[account_id]
                with open(self.accounts_file, 'w') as f:
                    json.dump(accounts, f, indent=2)
                return True
        except Exception as e:
            st.error(f"Error deleting account: {e}")
        return False
    
    def get_account(self, account_id):
        accounts = self.load_accounts()
        if account_id in accounts:
            account = accounts[account_id].copy()
            try:
                account['api_key'] = base64.b64decode(account['api_key']).decode()
                account['api_secret'] = base64.b64decode(account['api_secret']).decode()
            except:
                pass
            return account
        return None

# --- ALPACA TRADING CLIENT ---
class AlpacaTradingClient:
    def __init__(self):
        self.connected = False
        self.account_info = {}
        self.base_url = "https://paper-api.alpaca.markets"
        self.headers = {}
        
    def connect(self, api_key, api_secret):
        try:
            self.headers = {
                "APCA-API-KEY-ID": api_key.strip(),
                "APCA-API-SECRET-KEY": api_secret.strip()
            }
            
            response = requests.get(f"{self.base_url}/v2/account", headers=self.headers, timeout=10)
            if response.status_code == 200:
                self.connected = True
                self._update_account_info()
                return True
            else:
                st.error(f"Connection failed: {response.json().get('message', 'Unknown error')}")
                return False
        except Exception as e:
            st.error(f"Connection error: {e}")
            return False
    
    def _update_account_info(self):
        try:
            response = requests.get(f"{self.base_url}/v2/account", headers=self.headers, timeout=10)
            if response.status_code == 200:
                account_data = response.json()
                positions_response = requests.get(f"{self.base_url}/v2/positions", headers=self.headers, timeout=10)
                positions = positions_response.json() if positions_response.status_code == 200 else []
                
                unrealized_pl = sum(float(pos['unrealized_pl']) for pos in positions)
                positions_value = sum(float(pos['market_value']) for pos in positions)
                
                self.account_info = {
                    'portfolio_value': float(account_data['portfolio_value']),
                    'buying_power': float(account_data['buying_power']),
                    'cash': float(account_data['cash']),
                    'equity': float(account_data['equity']),
                    'positions_value': positions_value,
                    'unrealized_pl': unrealized_pl
                }
        except Exception as e:
            st.error(f"Error updating account info: {e}")
    
    def get_account_info(self):
        class Account:
            def __init__(self, info):
                self.portfolio_value = info.get('portfolio_value', 0)
                self.buying_power = info.get('buying_power', 0)
                self.cash = info.get('cash', 0)
                self.equity = info.get('equity', 0)
                self.positions_value = info.get('positions_value', 0)
                self.unrealized_pl = info.get('unrealized_pl', 0)
        return Account(self.account_info)
    
    def get_positions(self):
        try:
            response = requests.get(f"{self.base_url}/v2/positions", headers=self.headers, timeout=10)
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            st.error(f"Error getting positions: {e}")
            return []
    
    def place_order(self, symbol, qty, side):
        try:
            # Validate inputs
            if not symbol or not isinstance(symbol, str) or len(symbol.strip()) == 0:
                st.error("Invalid symbol")
                return None
                
            if qty <= 0:
                st.error("Quantity must be positive")
                return None
                
            order_data = {
                "symbol": symbol.upper().strip(),
                "qty": str(int(qty)),
                "side": side.lower(),
                "type": "market",
                "time_in_force": "day"
            }
            
            response = requests.post(f"{self.base_url}/v2/orders", headers=self.headers, json=order_data, timeout=10)
            if response.status_code == 200:
                self._update_account_info()
                st.success(f"Order executed: {side.upper()} {qty} {symbol}")
                return response.json()
            else:
                error_msg = response.json().get('message', 'Unknown error')
                st.error(f"Order failed: {error_msg}")
                return None
        except Exception as e:
            st.error(f"Error placing order: {e}")
            return None

# --- PERFORMANCE ANALYTICS ---
class PerformanceAnalytics:
    def __init__(self, trader):
        self.trader = trader
        self.trades = []
    
    def get_portfolio_summary(self):
        try:
            account_info = self.trader.get_account_info()
            positions = self.trader.get_positions()
            
            return {
                'total_equity': float(account_info.equity),
                'buying_power': float(account_info.buying_power),
                'cash': float(account_info.cash),
                'positions_value': float(account_info.positions_value),
                'unrealized_pl': float(account_info.unrealized_pl),
                'total_positions': len(positions)
            }
        except:
            return {
                'total_equity': 0, 'buying_power': 0, 'cash': 0,
                'positions_value': 0, 'unrealized_pl': 0, 'total_positions': 0
            }

# --- RISK MANAGEMENT ---
class RiskManager:
    def __init__(self, trader):
        self.trader = trader
        
    def calculate_var(self, positions, confidence_level=0.95, periods=252):
        try:
            account_info = self.trader.get_account_info()
            portfolio_value = float(account_info.equity)
            
            if portfolio_value == 0:
                return {'1d': 0, '1w': 0, '1m': 0}
                
            annual_volatility = 0.20
            daily_volatility = annual_volatility / np.sqrt(periods)
            
            var_1d = portfolio_value * daily_volatility * 2.33
            var_1w = var_1d * np.sqrt(5)
            var_1m = var_1d * np.sqrt(21)
            
            return {'1d': var_1d, '1w': var_1w, '1m': var_1m}
        except Exception as e:
            return {'1d': 0, '1w': 0, '1m': 0}

# --- AI TRADING ASSISTANT (FIXED VERSION) ---
class AITradingAssistant:
    def __init__(self):
        self.analysis_history = []
    
    def comprehensive_analysis(self, symbol):
        """Phân tích toàn diện với AI insights - FIXED VERSION"""
        try:
            # Kiểm tra symbol hợp lệ
            if not symbol or len(symbol.strip()) < 1:
                return self.get_fallback_analysis(symbol, "Symbol không hợp lệ")
            
            symbol = symbol.strip().upper()
            
            # Lấy dữ liệu đa chiều với timeout
            stock_data = self.get_stock_data(symbol)
            if stock_data.empty:
                return self.get_fallback_analysis(symbol, "Không thể lấy dữ liệu giá từ Yahoo Finance")
            
            company_info = self.get_company_info(symbol)
            technicals = self.calculate_technicals(stock_data)
            market_context = self.get_market_context()
            
            # AI Analysis
            analysis = self.generate_ai_insights(
                symbol, stock_data, company_info, technicals, market_context
            )
            
            self.analysis_history.append(analysis)
            return analysis
            
        except Exception as e:
            return self.get_fallback_analysis(symbol, f"Lỗi phân tích: {str(e)}")
    
    def get_stock_data(self, symbol):
        """Lấy dữ liệu stock với fallback"""
        try:
            # Thử period dài hơn trước
            data = yf.download(symbol, period='3mo', progress=False, timeout=15)
            if data.empty or len(data) < 20:
                # Thử period ngắn hơn
                data = yf.download(symbol, period='1mo', progress=False, timeout=15)
            return data
        except Exception as e:
            st.error(f"Lỗi lấy dữ liệu {symbol}: {e}")
            return pd.DataFrame()
    
    def get_company_info(self, symbol):
        """Lấy thông tin công ty"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            # Kiểm tra xem có dữ liệu hợp lệ không
            if not info or len(info) < 5:
                return {}
            return info
        except Exception as e:
            return {}
    
    def calculate_technicals(self, data):
        """Tính toán technical indicators"""
        if data.empty or len(data) < 20:
            return {}
        
        try:
            current_price = data['Close'].iloc[-1]
            price_5d_ago = data['Close'].iloc[-5] if len(data) >= 5 else current_price
            
            return {
                'sma_20': data['Close'].rolling(20).mean().iloc[-1],
                'sma_50': data['Close'].rolling(50).mean().iloc[-1],
                'rsi': ta.rsi(data['Close'], length=14).iloc[-1],
                'volume_avg': data['Volume'].rolling(20).mean().iloc[-1],
                'current_price': current_price,
                'price_change': ((current_price - price_5d_ago) / price_5d_ago * 100) if price_5d_ago > 0 else 0
            }
        except Exception as e:
            return {}
    
    def get_market_context(self):
        """Phân tích bối cảnh thị trường với fallback"""
        try:
            # Lấy VIX for market fear/greed
            vix = yf.download('^VIX', period='5d', progress=False, timeout=10)
            current_vix = vix['Close'].iloc[-1] if len(vix) > 0 else 20
            
            # SPY for overall market trend
            spy = yf.download('SPY', period='5d', progress=False, timeout=10)
            if len(spy) > 1:
                spy_trend = "UP" if spy['Close'].iloc[-1] > spy['Close'].iloc[-2] else "DOWN"
            else:
                spy_trend = "NEUTRAL"
            
            vix_sentiment = "LOW_FEAR" if current_vix < 15 else "HIGH_FEAR" if current_vix > 25 else "NEUTRAL"
            
            return {
                'vix_level': current_vix,
                'vix_sentiment': vix_sentiment,
                'market_trend': spy_trend,
                'market_condition': self.assess_market_condition(vix_sentiment, spy_trend)
            }
        except Exception as e:
            # Fallback data
            return {
                'market_condition': 'NEUTRAL',
                'vix_sentiment': 'NEUTRAL',
                'market_trend': 'NEUTRAL',
                'vix_level': 20
            }
    
    def assess_market_condition(self, vix_sentiment, market_trend):
        """Đánh giá điều kiện thị trường"""
        if market_trend == "UNKNOWN":
            return "NEUTRAL"
            
        if vix_sentiment == "LOW_FEAR" and market_trend == "UP":
            return "BULL_MARKET"
        elif vix_sentiment == "HIGH_FEAR" and market_trend == "DOWN":
            return "BEAR_MARKET"
        elif vix_sentiment == "HIGH_FEAR" and market_trend == "UP":
            return "VOLATILE_BULL"
        elif vix_sentiment == "LOW_FEAR" and market_trend == "DOWN":
            return "CORRECTION"
        else:
            return "SIDEWAYS"
    
    def generate_ai_insights(self, symbol, data, info, technicals, market):
        """Tạo insights thông minh như chuyên gia"""
        
        # Phân tích price action
        price_analysis = self.analyze_price_action(data)
        
        # Phân tích technicals
        tech_analysis = self.analyze_technicals(technicals)
        
        # Phân tích fundamental
        fundamental_analysis = self.analyze_fundamentals(info)
        
        # Market context
        context_analysis = self.analyze_market_context(market)
        
        # Kết hợp tất cả analysis
        combined_analysis = self.combine_analyses(
            price_analysis, tech_analysis, fundamental_analysis, context_analysis
        )
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'overall_sentiment': combined_analysis['sentiment'],
            'confidence_score': combined_analysis['confidence'],
            'key_insights': combined_analysis['insights'],
            'trading_recommendation': combined_analysis['recommendation'],
            'risk_assessment': combined_analysis['risk'],
            'price_targets': combined_analysis['targets'],
            'time_horizon': combined_analysis['horizon'],
            'support_resistance': self.calculate_support_resistance(data),
            'market_context': context_analysis,
            'current_price': technicals.get('current_price', 0),
            'price_change': technicals.get('price_change', 0)
        }
    
    def analyze_price_action(self, data):
        """Phân tích price action nâng cao"""
        if data.empty or len(data) < 20:
            return {'sentiment': 'NEUTRAL', 'confidence': 50, 'trend': 'UNKNOWN', 'volume_signal': 'UNKNOWN'}
        
        try:
            current_price = data['Close'].iloc[-1]
            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
            sma_50 = data['Close'].rolling(50).mean().iloc[-1]
            
            # Trend analysis
            if pd.isna(sma_20) or pd.isna(sma_50):
                return {'sentiment': 'NEUTRAL', 'confidence': 50, 'trend': 'UNKNOWN', 'volume_signal': 'UNKNOWN'}
                
            if current_price > sma_20 > sma_50:
                trend = "STRONG_UPTREND"
                sentiment = "BULLISH"
                confidence = 75
            elif current_price < sma_20 < sma_50:
                trend = "STRONG_DOWNTREND" 
                sentiment = "BEARISH"
                confidence = 75
            else:
                trend = "CONSOLIDATION"
                sentiment = "NEUTRAL"
                confidence = 60
            
            # Volume analysis
            avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            
            if pd.isna(avg_volume) or avg_volume == 0:
                volume_signal = "NORMAL_VOLUME"
            else:
                volume_ratio = current_volume / avg_volume
                if volume_ratio > 1.5:
                    volume_signal = "HIGH_VOLUME"
                    confidence += 10
                elif volume_ratio < 0.7:
                    volume_signal = "LOW_VOLUME" 
                    confidence -= 5
                else:
                    volume_signal = "NORMAL_VOLUME"
            
            return {
                'sentiment': sentiment,
                'confidence': min(95, max(30, confidence)),  # Đảm bảo confidence trong khoảng 30-95
                'trend': trend,
                'volume_signal': volume_signal,
                'price_vs_ma': {
                    'vs_sma_20': round((current_price - sma_20) / sma_20 * 100, 2) if sma_20 > 0 else 0,
                    'vs_sma_50': round((current_price - sma_50) / sma_50 * 100, 2) if sma_50 > 0 else 0
                }
            }
        except Exception as e:
            return {'sentiment': 'NEUTRAL', 'confidence': 50, 'trend': 'UNKNOWN', 'volume_signal': 'UNKNOWN'}
    
    def analyze_technicals(self, technicals):
        """Phân tích technical indicators"""
        if not technicals:
            return {'sentiment': 'NEUTRAL'}
        
        try:
            sentiment = "NEUTRAL"
            rsi = technicals.get('rsi', 50)
            
            if pd.isna(rsi):
                return {'sentiment': 'NEUTRAL'}
                
            if rsi > 70:
                sentiment = "BEARISH"
            elif rsi < 30:
                sentiment = "BULLISH"
            
            return {'sentiment': sentiment, 'rsi': rsi}
        except:
            return {'sentiment': 'NEUTRAL'}
    
    def analyze_fundamentals(self, info):
        """Phân tích cơ bản"""
        if not info:
            return {'sentiment': 'NEUTRAL'}
        
        try:
            # Đơn giản hóa phân tích cơ bản
            pe_ratio = info.get('trailingPE', 0)
            if pe_ratio and pe_ratio > 0:
                if pe_ratio < 15:
                    return {'sentiment': 'BULLISH'}
                elif pe_ratio > 25:
                    return {'sentiment': 'BEARISH'}
            
            # Kiểm tra recommendation
            rec = info.get('recommendationKey', 'hold')
            rec_scores = {
                'strong_buy': 'BULLISH',
                'buy': 'BULLISH',
                'hold': 'NEUTRAL',
                'sell': 'BEARISH',
                'strong_sell': 'BEARISH'
            }
            return {'sentiment': rec_scores.get(rec, 'NEUTRAL')}
        except:
            return {'sentiment': 'NEUTRAL'}
    
    def analyze_market_context(self, market):
        """Phân tích market context"""
        return market
    
    def combine_analyses(self, price, tech, fundamental, context):
        """Kết hợp tất cả phân tích thành recommendation cuối cùng"""
        
        # Scoring system
        scores = {
            'BULLISH': 1,
            'NEUTRAL': 0, 
            'BEARISH': -1
        }
        
        total_score = (
            scores.get(price['sentiment'], 0) * 0.4 +
            scores.get(tech.get('sentiment', 'NEUTRAL'), 0) * 0.3 +
            scores.get(fundamental.get('sentiment', 'NEUTRAL'), 0) * 0.2 +
            self.context_score(context) * 0.1
        )
        
        # Determine final sentiment
        if total_score > 0.3:
            sentiment = "STRONG_BULLISH"
            recommendation = "🟢 STRONG BUY"
            risk = "LOW"
            horizon = "SHORT_TERM"
        elif total_score > 0.1:
            sentiment = "BULLISH" 
            recommendation = "🟢 BUY"
            risk = "MEDIUM"
            horizon = "MEDIUM_TERM"
        elif total_score > -0.1:
            sentiment = "NEUTRAL"
            recommendation = "🟡 HOLD"
            risk = "MEDIUM"
            horizon = "WAIT"
        elif total_score > -0.3:
            sentiment = "BEARISH"
            recommendation = "🔴 SELL"
            risk = "HIGH"
            horizon = "SHORT_TERM"
        else:
            sentiment = "STRONG_BEARISH"
            recommendation = "🔴 STRONG SELL"
            risk = "VERY_HIGH"
            horizon = "IMMEDIATE"
        
        # Generate insights
        insights = self.generate_insights(price, tech, fundamental, context)
        
        # Calculate price targets
        targets = self.calculate_price_targets(price, tech)
        
        return {
            'sentiment': sentiment,
            'confidence': price['confidence'],
            'insights': insights,
            'recommendation': recommendation,
            'risk': risk,
            'targets': targets,
            'horizon': horizon
        }
    
    def context_score(self, context):
        """Chuyển market context thành score"""
        condition = context.get('market_condition', 'NEUTRAL')
        scores = {
            'BULL_MARKET': 0.8,
            'VOLATILE_BULL': 0.3,
            'NEUTRAL': 0.0,
            'SIDEWAYS': 0.0,
            'CORRECTION': -0.3,
            'BEAR_MARKET': -0.8
        }
        return scores.get(condition, 0.0)
    
    def generate_insights(self, price, tech, fundamental, context):
        """Tạo insights thông minh"""
        insights = []
        
        # Price action insights
        if price.get('trend') == "STRONG_UPTREND":
            insights.append("📈 Đang trong xu hướng tăng mạnh với momentum tốt")
        elif price.get('trend') == "STRONG_DOWNTREND":
            insights.append("📉 Đang trong xu hướng giảm mạnh, cần thận trọng")
        elif price.get('trend') == "CONSOLIDATION":
            insights.append("⚖️ Giá đang trong giai đoạn tích lũy, chờ breakout")
        
        # Volume insights
        if price.get('volume_signal') == "HIGH_VOLUME":
            insights.append("🔥 Volume cao cho thấy sự quan tâm mạnh từ thị trường")
        elif price.get('volume_signal') == "LOW_VOLUME":
            insights.append("💤 Volume thấp, thiếu sự tham gia của nhà đầu tư")
        
        # RSI insights
        rsi = tech.get('rsi')
        if rsi and not pd.isna(rsi):
            if rsi > 70:
                insights.append("🚨 RSI cho thấy quá mua, cảnh báo điều chỉnh")
            elif rsi < 30:
                insights.append("💎 RSI cho thấy quá bán, cơ hội mua tiềm năng")
        
        # Market context insights
        market_condition = context.get('market_condition', 'NEUTRAL')
        if market_condition == "BULL_MARKET":
            insights.append("🐂 Thị trường tổng thể đang tích cực, hỗ trợ xu hướng tăng")
        elif market_condition == "BEAR_MARKET":
            insights.append("🐻 Thị trường tổng thể tiêu cực, quản lý rủi ro cẩn thận")
        elif market_condition == "VOLATILE_BULL":
            insights.append("⚡ Thị trường biến động mạnh nhưng vẫn trong xu hướng tăng")
        
        # Nếu không có insights, thêm insights mặc định
        if not insights:
            insights.append("🔍 Đang phân tích dữ liệu thị trường...")
            insights.append("💡 Sử dụng kết hợp nhiều chỉ báo kỹ thuật")
            insights.append("📊 Theo dõi volume và price action để xác nhận xu hướng")
        
        return insights
    
    def calculate_price_targets(self, price, tech):
        """Tính toán price targets dựa trên dữ liệu thực tế"""
        try:
            # Sử dụng current_price từ technicals nếu có
            current_price = tech.get('current_price', 100)
            
            # Dựa trên sentiment để tính targets
            sentiment = price.get('sentiment', 'NEUTRAL')
            
            if sentiment == "BULLISH":
                return {
                    'short_term': current_price * 1.05,
                    'medium_term': current_price * 1.12,
                    'long_term': current_price * 1.20
                }
            elif sentiment == "BEARISH":
                return {
                    'short_term': current_price * 0.95,
                    'medium_term': current_price * 0.88, 
                    'long_term': current_price * 0.80
                }
            else:
                return {
                    'short_term': current_price * 1.02,
                    'medium_term': current_price * 1.05,
                    'long_term': current_price * 1.08
                }
        except:
            # Fallback targets
            return {
                'short_term': 105,
                'medium_term': 112,
                'long_term': 120
            }
    
    def calculate_support_resistance(self, data):
        """Tính support và resistance levels"""
        if data.empty or len(data) < 20:
            return {'support': 95, 'resistance': 105}
        
        try:
            high_20 = data['High'].rolling(20).max().iloc[-1]
            low_20 = data['Low'].rolling(20).min().iloc[-1]
            current_price = data['Close'].iloc[-1]
            
            # Tính toán support/resistance hợp lý
            support = low_20 * 0.98  # Support dưới mức thấp 20 ngày
            resistance = high_20 * 1.02  # Resistance trên mức cao 20 ngày
            
            return {
                'support': round(support, 2),
                'resistance': round(resistance, 2),
                'current': round(current_price, 2)
            }
        except:
            return {'support': 95, 'resistance': 105, 'current': 100}
    
    def get_fallback_analysis(self, symbol, error_msg=""):
        """Fallback analysis khi có lỗi với thông tin chi tiết"""
        error_insight = f"⚠️ {error_msg}" if error_msg else "⚠️ Dữ liệu hạn chế, cần phân tích thêm"
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'overall_sentiment': 'NEUTRAL',
            'confidence_score': 50,
            'key_insights': [
                error_insight,
                "💡 Thử symbol khác như AAPL, TSLA, MSFT",
                "🔧 Đang cải thiện hệ thống phân tích"
            ],
            'trading_recommendation': '🟡 HOLD',
            'risk_assessment': 'MEDIUM',
            'price_targets': {
                'short_term': 100,
                'medium_term': 105, 
                'long_term': 110
            },
            'time_horizon': 'WAIT',
            'support_resistance': {'support': 95, 'resistance': 105, 'current': 100},
            'market_context': {
                'market_condition': 'NEUTRAL',
                'vix_sentiment': 'NEUTRAL',
                'market_trend': 'NEUTRAL'
            },
            'current_price': 100,
            'price_change': 0
        }

# UI Implementation cho AI Assistant - FIXED VERSION
def display_ai_assistant():
    st.markdown("### 🤖 AI Trading Assistant Pro")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        symbol = st.text_input("Enter Symbol:", "AAPL", key="ai_symbol").upper()
        
        if st.button("🔍 Analyze with AI", type="primary", use_container_width=True):
            with st.spinner("🤖 AI Assistant is analyzing..."):
                assistant = AITradingAssistant()
                analysis = assistant.comprehensive_analysis(symbol)
                
                # Hiển thị kết quả
                display_ai_analysis(analysis)
    
    with col2:
        st.markdown("""
        <div class="dashboard-card">
            <h4>🎯 AI Capabilities</h4>
            <p>• Price Action Analysis</p>
            <p>• Technical Analysis</p>
            <p>• Market Context</p>
            <p>• Risk Assessment</p>
            <p>• Price Targets</p>
            <p>• Real-time Data</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="dashboard-card">
            <h4>💡 Try These Symbols</h4>
            <p>• <strong>AAPL</strong> - Apple</p>
            <p>• <strong>TSLA</strong> - Tesla</p>
            <p>• <strong>MSFT</strong> - Microsoft</p>
            <p>• <strong>GOOGL</strong> - Google</p>
            <p>• <strong>SPY</strong> - S&P 500 ETF</p>
        </div>
        """, unsafe_allow_html=True)

def display_ai_analysis(analysis):
    """Hiển thị AI analysis results - FIXED VERSION"""
    
    # Recommendation Card
    sentiment_color = {
        "STRONG_BULLISH": "#00ff88",
        "BULLISH": "#7ae582", 
        "NEUTRAL": "#667eea",
        "BEARISH": "#ff6b6b",
        "STRONG_BEARISH": "#ff4444"
    }
    
    color = sentiment_color.get(analysis['overall_sentiment'], "#667eea")
    
    st.markdown(f"""
    <div class="sentiment-card" style="border-left-color: {color};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 style="margin: 0; color: {color}; font-size: 1.5rem;">
                    {analysis['trading_recommendation']}
                </h3>
                <p style="margin: 0.2rem 0 0 0; color: #8898aa;">
                    {analysis['symbol']} • Confidence: {analysis['confidence_score']}%
                </p>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 1.2rem; font-weight: bold; color: {color};">
                    {analysis['overall_sentiment'].replace('_', ' ').title()}
                </div>
                <div style="color: #8898aa; font-size: 0.8rem;">
                    Risk: {analysis['risk_assessment']} • Horizon: {analysis['time_horizon']}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Current Price và Price Change
    if analysis.get('current_price', 0) > 0:
        col_price, col_change = st.columns(2)
        with col_price:
            st.metric("Current Price", f"${analysis['current_price']:.2f}")
        with col_change:
            change = analysis.get('price_change', 0)
            st.metric("5-Day Change", f"{change:+.1f}%")
    
    # Key Insights
    st.markdown("#### 💡 AI Insights")
    for insight in analysis['key_insights']:
        st.markdown(f"- {insight}")
    
    # Price Targets
    if analysis['price_targets']['short_term'] > 0:
        st.markdown("#### 🎯 Price Targets")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Short Term", f"${analysis['price_targets']['short_term']:.2f}")
        with col2:
            st.metric("Medium Term", f"${analysis['price_targets']['medium_term']:.2f}")
        with col3:
            st.metric("Long Term", f"${analysis['price_targets']['long_term']:.2f}")
    
    # Support & Resistance
    st.markdown("#### 📊 Support & Resistance")
    col_sup, col_res = st.columns(2)
    with col_sup:
        st.metric("Support Level", f"${analysis['support_resistance']['support']:.2f}")
    with col_res:
        st.metric("Resistance Level", f"${analysis['support_resistance']['resistance']:.2f}")
    
    # Market Context
    st.markdown("#### 🌐 Market Context")
    col_ctx1, col_ctx2, col_ctx3 = st.columns(3)
    with col_ctx1:
        st.metric("Market Condition", analysis['market_context']['market_condition'])
    with col_ctx2:
        st.metric("VIX Sentiment", analysis['market_context']['vix_sentiment'])
    with col_ctx3:
        st.metric("Market Trend", analysis['market_context']['market_trend'])

# --- MACHINE LEARNING ---
@st.cache_resource
def train_model_on_the_fly(data):
    if data is None or len(data) < 100:
        return None
    
    try:
        data['SMA_20'] = data['close'].rolling(20).mean()
        data['SMA_50'] = data['close'].rolling(50).mean()
        data['RSI'] = ta.rsi(data['close'], length=14)
        data['Volume_SMA'] = data['volume'].rolling(20).mean()
        
        data['Future_Return'] = data['close'].shift(-5) / data['close'] - 1
        data['Target'] = (data['Future_Return'] > 0).astype(int)
        
        features = ['SMA_20', 'SMA_50', 'RSI', 'Volume_SMA']
        data_clean = data.dropna()
        
        if len(data_clean) < 50:
            return None
            
        X = data_clean[features]
        y = data_clean['Target']
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return model
    except Exception as e:
        return None

def get_ml_signal(data, model):
    if data is None or model is None:
        return "NO_SIGNAL", 0
    
    try:
        latest = data.iloc[-1].copy()
        latest['SMA_20'] = data['close'].rolling(20).mean().iloc[-1]
        latest['SMA_50'] = data['close'].rolling(50).mean().iloc[-1]
        latest['RSI'] = ta.rsi(data['close'], length=14).iloc[-1]
        latest['Volume_SMA'] = data['volume'].rolling(20).mean().iloc[-1]
        
        features = ['SMA_20', 'SMA_50', 'RSI', 'Volume_SMA']
        X_latest = pd.DataFrame([latest[features]])
        
        prediction = model.predict(X_latest)[0]
        probability = model.predict_proba(X_latest)[0]
        
        confidence = max(probability)
        
        if prediction == 1 and confidence > 0.6:
            return "BUY", confidence
        elif prediction == 0 and confidence > 0.6:
            return "SELL", confidence
        else:
            return "HOLD", confidence
            
    except Exception as e:
        return "NO_SIGNAL", 0

# --- DATA LOADING ---
@st.cache_data(ttl=300)
def load_stock_data(symbol, period="6mo"):
    try:
        data = yf.download(symbol, period=period, progress=False, timeout=15)
        if not data.empty:
            data.columns = [col.lower() for col in data.columns]
        return data
    except Exception as e:
        st.error(f"Error loading data for {symbol}: {e}")
        return None

@st.cache_data(ttl=300)
def load_crypto_data(symbol):
    try:
        if '/' in symbol:
            symbol = symbol.replace('/', '-')
        data = yf.download(symbol + "-USD", period="6mo", progress=False, timeout=15)
        if not data.empty:
            data.columns = [col.lower() for col in data.columns]
        return data
    except Exception as e:
        st.error(f"Error loading crypto data for {symbol}: {e}")
        return None

def get_current_price(symbol, asset_type):
    try:
        if asset_type == "Stock":
            data = load_stock_data(symbol, "1d")
        else:
            data = load_crypto_data(symbol)
        
        if data is not None and not data.empty:
            return data['close'].iloc[-1]
        return None
    except Exception as e:
        st.error(f"Error getting current price for {symbol}: {e}")
        return None

# --- HELPER FUNCTIONS ---
def display_position(position):
    symbol = position['symbol']
    qty = float(position['qty'])
    avg_entry = float(position['avg_entry_price'])
    current_price = float(position['current_price'])
    unrealized_pl = float(position['unrealized_pl'])
    pl_percent = (unrealized_pl / (avg_entry * qty)) * 100 if avg_entry * qty != 0 else 0
    
    pl_class = "" if unrealized_pl >= 0 else "sell"
    badge_class = "badge-profit" if unrealized_pl >= 0 else "badge-loss"
    
    st.markdown(f"""
    <div class="position-item {pl_class}">
        <div style="display: flex; justify-content: between; align-items: center;">
            <div>
                <strong>{symbol}</strong>
                <span class="badge {badge_class}">{qty:.0f} shares</span>
            </div>
            <div style="text-align: right;">
                <div style="color: {'#00ff88' if unrealized_pl >= 0 else '#ff4444'}; font-weight: bold;">
                    ${unrealized_pl:+.2f} ({pl_percent:+.1f}%)
                </div>
                <div style="font-size: 0.8em; color: #8898aa;">
                    Avg: ${avg_entry:.2f} • Current: ${current_price:.2f}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- SESSION STATE ---
if 'trader' not in st.session_state:
    st.session_state.trader = AlpacaTradingClient()
if 'performance_analytics' not in st.session_state:
    st.session_state.performance_analytics = None
if 'account_manager' not in st.session_state:
    st.session_state.account_manager = AccountManager()
if 'ai_assistant' not in st.session_state:
    st.session_state.ai_assistant = AITradingAssistant()
if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False

# --- CONNECTION PANEL ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="connection-panel">
        <h3>🔌 API Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    api_key = st.text_input("API Key", type="password", placeholder="Enter Alpaca API Key")
    api_secret = st.text_input("API Secret", type="password", placeholder="Enter Alpaca API Secret")
    
    col1a, col1b = st.columns(2)
    with col1a:
        if st.button("🚀 Connect", use_container_width=True, type="primary"):
            if api_key and api_secret:
                with st.spinner("Connecting to Alpaca..."):
                    if st.session_state.trader.connect(api_key, api_secret):
                        st.session_state.performance_analytics = PerformanceAnalytics(st.session_state.trader)
                        st.success("✅ Connected Successfully!")
                    else:
                        st.error("❌ Connection failed. Check your API credentials.")
            else:
                st.warning("⚠️ Please enter both API Key and API Secret")
    
    with col1b:
        if st.button("💾 Save Account", use_container_width=True):
            if api_key and api_secret:
                nickname = st.text_input("Account Name", value="My Alpaca Account", key="account_name")
                if st.session_state.account_manager.save_account(api_key, api_secret, nickname):
                    st.success("✅ Account Saved Successfully!")
                else:
                    st.error("❌ Failed to save account")
            else:
                st.warning("⚠️ Please enter API credentials first")

with col2:
    st.markdown("""
    <div class="dashboard-card">
        <h4>📋 Quick Guide</h4>
        <p><strong>1. Get API Keys:</strong><br>app.alpaca.markets</p>
        <p><strong>2. Paper Trading:</strong><br>$100,000 virtual</p>
        <p><strong>3. US Stocks:</strong><br>AAPL, TSLA, etc.</p>
        <p><strong>4. Crypto:</strong><br>BTC-USD, ETH-USD</p>
    </div>
    """, unsafe_allow_html=True)

# Load saved accounts
saved_accounts = st.session_state.account_manager.load_accounts()
if saved_accounts:
    with st.expander("💾 Saved Accounts", expanded=False):
        account_options = {k: v['nickname'] for k, v in saved_accounts.items()}
        selected_account = st.selectbox("Select account:", [""] + list(account_options.keys()), 
                                      format_func=lambda x: account_options.get(x, "Select account"))
        
        if selected_account:
            account_info = st.session_state.account_manager.get_account(selected_account)
            if account_info and st.button("Use This Account"):
                st.session_state.trader = AlpacaTradingClient()
                if st.session_state.trader.connect(account_info['api_key'], account_info['api_secret']):
                    st.session_state.performance_analytics = PerformanceAnalytics(st.session_state.trader)
                    st.success("✅ Connected with saved account!")
                    st.rerun()

# --- MAIN DASHBOARD ---
if st.session_state.trader.connected:
    trader = st.session_state.trader
    performance_analytics = st.session_state.performance_analytics
    
    # Refresh Button
    col_refresh, col_space = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 Refresh Data", use_container_width=True):
            with st.spinner("Refreshing data..."):
                trader._update_account_info()
                st.rerun()
    
    # Portfolio Overview
    st.markdown("---")
    st.subheader("📊 Portfolio Overview")
    
    portfolio_summary = performance_analytics.get_portfolio_summary()
    positions = trader.get_positions()
    
    # Main Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="feature-icon">💰</div>
            <div class="metric-value">${portfolio_summary['total_equity']:,.0f}</div>
            <div class="metric-label">Total Equity</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="feature-icon">⚡</div>
            <div class="metric-value">${portfolio_summary['buying_power']:,.0f}</div>
            <div class="metric-label">Buying Power</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="feature-icon">💵</div>
            <div class="metric-value">${portfolio_summary['cash']:,.0f}</div>
            <div class="metric-label">Available Cash</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        pl_color = "#00ff88" if portfolio_summary['unrealized_pl'] >= 0 else "#ff4444"
        st.markdown(f"""
        <div class="metric-card">
            <div class="feature-icon">📈</div>
            <div class="metric-value" style="color: {pl_color}">${portfolio_summary['unrealized_pl']:,.2f}</div>
            <div class="metric-label">Unrealized P&L</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Positions Grouped View
    st.markdown("---")
    st.subheader("📈 Positions")
    
    if positions:
        # Group positions by type (profit/loss)
        profitable_positions = [p for p in positions if float(p['unrealized_pl']) > 0]
        losing_positions = [p for p in positions if float(p['unrealized_pl']) <= 0]
        
        col_pos1, col_pos2 = st.columns(2)
        
        with col_pos1:
            if profitable_positions:
                st.markdown("""
                <div class="position-group">
                    <h4>🟢 Profitable Positions</h4>
                </div>
                """, unsafe_allow_html=True)
                for position in profitable_positions:
                    display_position(position)
            else:
                st.info("No profitable positions")
        
        with col_pos2:
            if losing_positions:
                st.markdown("""
                <div class="position-group">
                    <h4>🔴 Losing Positions</h4>
                </div>
                """, unsafe_allow_html=True)
                for position in losing_positions:
                    display_position(position)
            else:
                st.info("No losing positions")
    else:
        st.info("💰 No active positions - Start trading to see positions here!")
    
    # Tabs for Advanced Features
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Trading", "🤖 Auto Trading", "📊 Analytics", "🤖 AI Assistant"])
    
    with tab1:
        st.subheader("Manual Trading")
        
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            asset_type = st.radio("Asset Type", ["Stock", "Crypto"], horizontal=True)
            symbol = st.text_input("Symbol", value="AAPL" if asset_type == "Stock" else "BTC-USD", key="trading_symbol").upper()
            
            # Symbol validation
            if symbol:
                if len(symbol) < 1:
                    st.warning("Please enter a valid symbol")
                elif asset_type == "Stock" and not re.match(r'^[A-Z]{1,5}$', symbol):
                    st.warning("Please enter a valid stock symbol (1-5 uppercase letters)")
        
        with col_t2:
            qty = st.number_input("Quantity", min_value=1, value=10, step=1, key="trading_qty")
            
            # Price display
            current_price = get_current_price(symbol, asset_type)
            if current_price:
                st.metric("Current Price", f"${current_price:.2f}")
                total_cost = current_price * qty
                st.metric("Total Cost", f"${total_cost:,.2f}")
            else:
                st.warning("Could not fetch current price")
        
        col_buy, col_sell = st.columns(2)
        with col_buy:
            if st.button("🟢 BUY", use_container_width=True, type="primary", key="buy_btn"):
                if symbol and qty > 0 and current_price:
                    result = trader.place_order(symbol, qty, "buy")
                    if result:
                        time.sleep(2)
                        st.rerun()
                else:
                    st.error("Please check symbol and quantity")
        
        with col_sell:
            if st.button("🔴 SELL", use_container_width=True, type="primary", key="sell_btn"):
                if symbol and qty > 0 and current_price:
                    result = trader.place_order(symbol, qty, "sell")
                    if result:
                        time.sleep(2)
                        st.rerun()
                else:
                    st.error("Please check symbol and quantity")
    
    with tab2:
        st.subheader("Automated Trading")
        
        col_a1, col_a2 = st.columns(2)
        
        with col_a1:
            auto_symbol = st.text_input("Symbol", value="AAPL", key="auto_symbol").upper()
            auto_asset = st.radio("Asset", ["Stock", "Crypto"], horizontal=True, key="auto_asset")
            risk_per_trade = st.slider("Risk per Trade %", 0.1, 5.0, 1.0, 0.1)
            
            st.info(f"Risk per trade: ${portfolio_summary['total_equity'] * risk_per_trade / 100:,.2f}")
        
        with col_a2:
            ml_enabled = st.checkbox("Enable ML Signals", value=True)
            min_confidence = st.slider("Min Confidence", 0.5, 0.95, 0.7, 0.05)
            
            st.markdown("#### Bot Control")
            col_start, col_stop = st.columns(2)
            with col_start:
                if not st.session_state.bot_running:
                    if st.button("🚀 Start Bot", use_container_width=True, type="primary"):
                        st.session_state.bot_running = True
                        st.success("🤖 Trading Bot Started!")
                else:
                    if st.button("⏸️ Pause Bot", use_container_width=True):
                        st.session_state.bot_running = False
                        st.warning("⏸️ Trading Bot Paused!")
            
            with col_stop:
                if st.button("🛑 Stop Bot", use_container_width=True):
                    st.session_state.bot_running = False
                    st.error("🛑 Trading Bot Stopped!")
        
        if st.session_state.bot_running:
            st.info("🤖 Bot is running...")
            # Simulate ML trading
            data = load_stock_data(auto_symbol) if auto_asset == "Stock" else load_crypto_data(auto_symbol)
            if data is not None:
                model = train_model_on_the_fly(data)
                signal, confidence = get_ml_signal(data, model)
                
                col_sig, col_conf = st.columns(2)
                with col_sig:
                    signal_color = "#00ff88" if signal == "BUY" else "#ff4444" if signal == "SELL" else "#667eea"
                    st.markdown(f"<h3 style='color: {signal_color};'>{signal}</h3>", unsafe_allow_html=True)
                with col_conf:
                    confidence_color = "#00ff88" if confidence > 0.7 else "#ff6b6b" if confidence < 0.5 else "#667eea"
                    st.markdown(f"<h3 style='color: {confidence_color};'>{confidence:.1%}</h3>", unsafe_allow_html=True)
                
                if ml_enabled and confidence >= min_confidence and signal != "HOLD":
                    st.warning(f"🚨 Signal: {signal} {auto_symbol} (Confidence: {confidence:.1%})")
        else:
            st.info("⏸️ Trading Bot is stopped. Click 'Start Bot' to begin automated trading.")
    
    with tab3:
        st.subheader("Risk & Performance Analytics")
        
        risk_manager = RiskManager(trader)
        portfolio_var = risk_manager.calculate_var(positions)
        
        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            st.metric("1-Day VaR", f"${portfolio_var['1d']:,.0f}")
        with col_r2:
            st.metric("1-Week VaR", f"${portfolio_var['1w']:,.0f}")
        with col_r3:
            st.metric("1-Month VaR", f"${portfolio_var['1m']:,.0f}")
        
        # Performance Metrics
        st.markdown("#### Portfolio Metrics")
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            total_return = (portfolio_summary['unrealized_pl'] / portfolio_summary['total_equity'] * 100) if portfolio_summary['total_equity'] > 0 else 0
            st.metric("Total Return", f"{total_return:+.2f}%")
        
        with col_m2:
            positions_count = len(positions)
            st.metric("Active Positions", f"{positions_count}")
        
        with col_m3:
            cash_ratio = (portfolio_summary['cash'] / portfolio_summary['total_equity'] * 100) if portfolio_summary['total_equity'] > 0 else 0
            st.metric("Cash Allocation", f"{cash_ratio:.1f}%")
        
        # Performance Chart
        if st.button("Generate Performance Report", key="perf_report"):
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode = "gauge+number+delta",
                value = portfolio_summary['unrealized_pl'],
                title = {'text': "Portfolio P&L"},
                gauge = {
                    'axis': {'range': [min(-1000, portfolio_summary['unrealized_pl']), max(1000, portfolio_summary['unrealized_pl'])]},
                    'bar': {'color': "#00ff88" if portfolio_summary['unrealized_pl'] >= 0 else "#ff4444"},
                    'steps': [
                        {'range': [-1000, 0], 'color': "rgba(255, 68, 68, 0.2)"},
                        {'range': [0, 1000], 'color': "rgba(0, 255, 136, 0.2)"}
                    ]
                }
            ))
            
            fig.update_layout(
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': "white", 'family': "Arial"}
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Thay thế Social Sentiment bằng AI Assistant
        display_ai_assistant()

else:
    # Welcome Screen
    st.markdown("---")
    st.subheader("🎯 Welcome to Live Trading Pro")
    
    col_w1, col_w2, col_w3 = st.columns(3)
    
    with col_w1:
        st.markdown("""
        <div class="dashboard-card">
            <div class="feature-icon">📈</div>
            <h3>Stock Trading</h3>
            <p>Trade US stocks with real-time data and advanced analytics</p>
            <p><strong>Supported:</strong> AAPL, TSLA, MSFT, GOOGL, etc.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_w2:
        st.markdown("""
        <div class="dashboard-card">
            <div class="feature-icon">💰</div>
            <h3>Paper Trading</h3>
            <p>$100,000 virtual trading account</p>
            <p>Risk-free environment to test strategies</p>
            <p>Real-market conditions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_w3:
        st.markdown("""
        <div class="dashboard-card">
            <div class="feature-icon">⚡</div>
            <h3>Live Data</h3>
            <p>Real-time market data</p>
            <p>AI-powered analysis</p>
            <p>Advanced risk management</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Features Grid
    st.markdown("---")
    st.subheader("🚀 Advanced Features")
    
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    
    with col_f1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 2rem;">🤖</div>
            <h4>AI Assistant</h4>
            <p>Smart trading insights and recommendations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_f2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 2rem;">📊</div>
            <h4>ML Signals</h4>
            <p>Machine learning trading signals</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_f3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 2rem;">🎯</div>
            <h4>Risk Management</h4>
            <p>VaR calculations and risk assessment</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_f4:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 2rem;">⚡</div>
            <h4>Real-time</h4>
            <p>Live portfolio tracking and analytics</p>
        </div>
        """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
<div style='text-align: center; padding: 3rem; color: #8898aa;'>
    <p style='margin: 0; font-size: 0.9rem;'>Built with Streamlit • Alpaca Markets API • AI Trading Assistant</p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.8rem;'>Paper Trading Account • Real-time Market Data • Risk Management</p>
</div>
""", unsafe_allow_html=True)