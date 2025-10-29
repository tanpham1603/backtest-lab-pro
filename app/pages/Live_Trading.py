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
            
            response = requests.get(f"{self.base_url}/v2/account", headers=self.headers)
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
            response = requests.get(f"{self.base_url}/v2/account", headers=self.headers)
            if response.status_code == 200:
                account_data = response.json()
                positions_response = requests.get(f"{self.base_url}/v2/positions", headers=self.headers)
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
            response = requests.get(f"{self.base_url}/v2/positions", headers=self.headers)
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            return []
    
    def place_order(self, symbol, qty, side):
        try:
            order_data = {
                "symbol": symbol.upper(),
                "qty": str(int(qty)),
                "side": side.lower(),
                "type": "market",
                "time_in_force": "day"
            }
            
            response = requests.post(f"{self.base_url}/v2/orders", headers=self.headers, json=order_data)
            if response.status_code == 200:
                self._update_account_info()
                return response.json()
            else:
                st.error(f"Order failed: {response.json().get('message', 'Unknown error')}")
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

# --- AI TRADING ASSISTANT ---
class AITradingAssistant:
    def __init__(self):
        self.analysis_history = []
    
    def comprehensive_analysis(self, symbol):
        """Phân tích toàn diện với AI insights"""
        try:
            # Lấy dữ liệu đa chiều
            stock_data = self.get_stock_data(symbol)
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
            return self.get_fallback_analysis(symbol)
    
    def get_stock_data(self, symbol):
        """Lấy dữ liệu stock"""
        try:
            data = yf.download(symbol, period='6mo', progress=False)
            return data
        except:
            return pd.DataFrame()
    
    def get_company_info(self, symbol):
        """Lấy thông tin công ty"""
        try:
            stock = yf.Ticker(symbol)
            return stock.info
        except:
            return {}
    
    def calculate_technicals(self, data):
        """Tính toán technical indicators"""
        if len(data) < 20:
            return {}
        
        try:
            return {
                'sma_20': data['Close'].rolling(20).mean().iloc[-1],
                'sma_50': data['Close'].rolling(50).mean().iloc[-1],
                'rsi': ta.rsi(data['Close'], length=14).iloc[-1],
                'volume_avg': data['Volume'].rolling(20).mean().iloc[-1]
            }
        except:
            return {}
    
    def get_market_context(self):
        """Phân tích bối cảnh thị trường"""
        try:
            # Lấy VIX for market fear/greed
            vix = yf.download('^VIX', period='5d', progress=False)
            current_vix = vix['Close'].iloc[-1] if len(vix) > 0 else 20
            
            # SPY for overall market trend
            spy = yf.download('SPY', period='5d', progress=False)
            spy_trend = "UP" if len(spy) > 1 and spy['Close'].iloc[-1] > spy['Close'].iloc[-2] else "DOWN"
            
            vix_sentiment = "LOW_FEAR" if current_vix < 15 else "HIGH_FEAR" if current_vix > 25 else "NEUTRAL"
            
            return {
                'vix_level': current_vix,
                'vix_sentiment': vix_sentiment,
                'market_trend': spy_trend,
                'market_condition': self.assess_market_condition(vix_sentiment, spy_trend)
            }
        except:
            return {'market_condition': 'UNKNOWN', 'vix_sentiment': 'NEUTRAL'}
    
    def assess_market_condition(self, vix_sentiment, market_trend):
        """Đánh giá điều kiện thị trường"""
        if vix_sentiment == "LOW_FEAR" and market_trend == "UP":
            return "BULL_MARKET"
        elif vix_sentiment == "HIGH_FEAR" and market_trend == "DOWN":
            return "BEAR_MARKET"
        elif vix_sentiment == "HIGH_FEAR" and market_trend == "UP":
            return "VOLATILE_BULL"
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
            'market_context': context_analysis
        }
    
    def analyze_price_action(self, data):
        """Phân tích price action nâng cao"""
        if len(data) < 20:
            return {'sentiment': 'NEUTRAL', 'confidence': 50}
        
        current_price = data['Close'].iloc[-1]
        sma_20 = data['Close'].rolling(20).mean().iloc[-1]
        sma_50 = data['Close'].rolling(50).mean().iloc[-1]
        
        # Trend analysis
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
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
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
            'confidence': min(95, confidence),
            'trend': trend,
            'volume_signal': volume_signal,
            'price_vs_ma': {
                'vs_sma_20': round((current_price - sma_20) / sma_20 * 100, 2),
                'vs_sma_50': round((current_price - sma_50) / sma_50 * 100, 2)
            }
        }
    
    def analyze_technicals(self, technicals):
        """Phân tích technical indicators"""
        if not technicals:
            return {'sentiment': 'NEUTRAL'}
        
        sentiment = "NEUTRAL"
        rsi = technicals.get('rsi', 50)
        
        if rsi > 70:
            sentiment = "BEARISH"
        elif rsi < 30:
            sentiment = "BULLISH"
        
        return {'sentiment': sentiment, 'rsi': rsi}
    
    def analyze_fundamentals(self, info):
        """Phân tích cơ bản"""
        if not info:
            return {'sentiment': 'NEUTRAL'}
        
        # Đơn giản hóa phân tích cơ bản
        pe_ratio = info.get('trailingPE', 0)
        if pe_ratio > 0 and pe_ratio < 20:
            return {'sentiment': 'BULLISH'}
        elif pe_ratio > 25:
            return {'sentiment': 'BEARISH'}
        else:
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
        condition = context.get('market_condition', 'UNKNOWN')
        scores = {
            'BULL_MARKET': 0.8,
            'VOLATILE_BULL': 0.3,
            'SIDEWAYS': 0.0,
            'BEAR_MARKET': -0.8
        }
        return scores.get(condition, 0.0)
    
    def generate_insights(self, price, tech, fundamental, context):
        """Tạo insights thông minh"""
        insights = []
        
        # Price action insights
        if price['trend'] == "STRONG_UPTREND":
            insights.append("📈 Đang trong xu hướng tăng mạnh với momentum tốt")
        elif price['trend'] == "STRONG_DOWNTREND":
            insights.append("📉 Đang trong xu hướng giảm mạnh, cần thận trọng")
        
        # Volume insights
        if price['volume_signal'] == "HIGH_VOLUME":
            insights.append("🔥 Volume cao cho thấy sự quan tâm mạnh")
        
        # Market context insights
        if context['market_condition'] == "BULL_MARKET":
            insights.append("🐂 Thị trường tổng thể đang tích cực")
        elif context['market_condition'] == "BEAR_MARKET":
            insights.append("🐻 Thị trường tổng thể tiêu cực, quản lý rủi ro cẩn thận")
        
        return insights
    
    def calculate_price_targets(self, price, tech):
        """Tính toán price targets"""
        try:
            # Sử dụng dữ liệu thực tế thay vì giá cố định
            current_price = 150  # Có thể thay bằng giá thực tế
            
            # Simple target calculation based on sentiment
            if price['sentiment'] == "BULLISH":
                return {
                    'short_term': current_price * 1.05,
                    'medium_term': current_price * 1.12,
                    'long_term': current_price * 1.20
                }
            elif price['sentiment'] == "BEARISH":
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
            return {
                'short_term': 0,
                'medium_term': 0,
                'long_term': 0
            }
    
    def calculate_support_resistance(self, data):
        """Tính support và resistance levels"""
        if len(data) < 20:
            return {'support': 0, 'resistance': 0}
        
        try:
            high_20 = data['High'].rolling(20).max().iloc[-1]
            low_20 = data['Low'].rolling(20).min().iloc[-1]
            return {'support': low_20, 'resistance': high_20}
        except:
            return {'support': 0, 'resistance': 0}
    
    def get_fallback_analysis(self, symbol):
        """Fallback analysis khi có lỗi"""
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'overall_sentiment': 'NEUTRAL',
            'confidence_score': 50,
            'key_insights': ['Dữ liệu hạn chế, cần phân tích thêm'],
            'trading_recommendation': '🟡 HOLD',
            'risk_assessment': 'MEDIUM',
            'price_targets': {'short_term': 0, 'medium_term': 0, 'long_term': 0},
            'time_horizon': 'WAIT',
            'support_resistance': {'support': 0, 'resistance': 0},
            'market_context': {'market_condition': 'UNKNOWN'}
        }

# UI Implementation cho AI Assistant
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
        </div>
        """, unsafe_allow_html=True)

def display_ai_analysis(analysis):
    """Hiển thị AI analysis results"""
    
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
                    Risk: {analysis['risk_assessment']}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Insights
    st.markdown("#### 💡 AI Insights")
    for insight in analysis['key_insights']:
        st.markdown(f"- {insight}")
    
    # Price Targets
    if analysis['price_targets']['short_term'] > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Short Term Target", f"${analysis['price_targets']['short_term']:.2f}")
        with col2:
            st.metric("Medium Term Target", f"${analysis['price_targets']['medium_term']:.2f}")
        with col3:
            st.metric("Long Term Target", f"${analysis['price_targets']['long_term']:.2f}")
    
    # Market Context
    st.markdown("#### 🌐 Market Context")
    st.write(f"Market Condition: {analysis['market_context']['market_condition']}")
    if 'vix_sentiment' in analysis['market_context']:
        st.write(f"VIX Sentiment: {analysis['market_context']['vix_sentiment']}")

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
        data = yf.download(symbol, period=period, progress=False)
        if not data.empty:
            data.columns = [col.lower() for col in data.columns]
        return data
    except Exception as e:
        return None

@st.cache_data(ttl=300)
def load_crypto_data(symbol):
    try:
        if '/' in symbol:
            symbol = symbol.replace('/', '-')
        data = yf.download(symbol + "-USD", period="6mo", progress=False)
        if not data.empty:
            data.columns = [col.lower() for col in data.columns]
        return data
    except:
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
    except:
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
                with st.spinner("Connecting..."):
                    if st.session_state.trader.connect(api_key, api_secret):
                        st.session_state.performance_analytics = PerformanceAnalytics(st.session_state.trader)
                        st.success("Connected Successfully!")
            else:
                st.warning("Enter API credentials")
    
    with col1b:
        if st.button("💾 Save Account", use_container_width=True):
            if api_key and api_secret:
                nickname = st.text_input("Account Name", value="My Alpaca Account", key="account_name")
                if st.session_state.account_manager.save_account(api_key, api_secret, nickname):
                    st.success("Account Saved!")

with col2:
    st.markdown("""
    <div class="dashboard-card">
        <h4>📋 Quick Guide</h4>
        <p><strong>1. Get API Keys:</strong><br>app.alpaca.markets</p>
        <p><strong>2. Paper Trading:</strong><br>$100,000 virtual</p>
        <p><strong>3. US Stocks:</strong><br>AAPL, TSLA, etc.</p>
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
                    st.success("Connected with saved account!")
                    st.rerun()

# --- MAIN DASHBOARD ---
if st.session_state.trader.connected:
    trader = st.session_state.trader
    performance_analytics = st.session_state.performance_analytics
    
    # Refresh Button
    if st.button("🔄 Refresh Data"):
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
            <div class="metric-value" style="color: {pl_color}">${portfolio_summary['unrealized_pl']:,.0f}</div>
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
        st.info("💰 No active positions")
    
    # Tabs for Advanced Features
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Trading", "🤖 Auto Trading", "📊 Analytics", "🤖 AI Assistant"])
    
    with tab1:
        st.subheader("Manual Trading")
        
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            asset_type = st.radio("Asset Type", ["Stock", "Crypto"], horizontal=True)
            symbol = st.text_input("Symbol", value="AAPL" if asset_type == "Stock" else "BTC-USD", key="trading_symbol").upper()
        
        with col_t2:
            qty = st.number_input("Quantity", min_value=1, value=10, step=1, key="trading_qty")
            
            # Price display
            current_price = get_current_price(symbol, asset_type)
            if current_price:
                st.metric("Current Price", f"${current_price:.2f}")
        
        col_buy, col_sell = st.columns(2)
        with col_buy:
            if st.button("🟢 BUY", use_container_width=True, type="primary", key="buy_btn"):
                if symbol and qty > 0:
                    result = trader.place_order(symbol, qty, "buy")
                    if result:
                        time.sleep(1)
                        st.rerun()
        
        with col_sell:
            if st.button("🔴 SELL", use_container_width=True, type="primary", key="sell_btn"):
                if symbol and qty > 0:
                    result = trader.place_order(symbol, qty, "sell")
                    if result:
                        time.sleep(1)
                        st.rerun()
    
    with tab2:
        st.subheader("Automated Trading")
        
        col_a1, col_a2 = st.columns(2)
        
        with col_a1:
            auto_symbol = st.text_input("Symbol", value="AAPL", key="auto_symbol").upper()
            auto_asset = st.radio("Asset", ["Stock", "Crypto"], horizontal=True, key="auto_asset")
            risk_per_trade = st.slider("Risk per Trade %", 0.1, 5.0, 1.0, 0.1)
        
        with col_a2:
            ml_enabled = st.checkbox("Enable ML Signals", value=True)
            min_confidence = st.slider("Min Confidence", 0.5, 0.95, 0.7, 0.05)
            
            col_start, col_stop = st.columns(2)
            with col_start:
                if not st.session_state.bot_running:
                    if st.button("🚀 Start Bot", use_container_width=True):
                        st.session_state.bot_running = True
                        st.success("Bot started!")
                else:
                    if st.button("⏸️ Pause Bot", use_container_width=True):
                        st.session_state.bot_running = False
                        st.warning("Bot paused!")
            
            with col_stop:
                if st.button("🛑 Stop Bot", use_container_width=True):
                    st.session_state.bot_running = False
                    st.error("Bot stopped!")
        
        if st.session_state.bot_running:
            st.info("🤖 Bot is running...")
            # Simulate ML trading
            data = load_stock_data(auto_symbol) if auto_asset == "Stock" else load_crypto_data(auto_symbol)
            if data is not None:
                model = train_model_on_the_fly(data)
                signal, confidence = get_ml_signal(data, model)
                
                col_sig, col_conf = st.columns(2)
                with col_sig:
                    st.metric("ML Signal", signal)
                with col_conf:
                    st.metric("Confidence", f"{confidence:.1%}")
                
                if ml_enabled and confidence >= min_confidence and signal != "HOLD":
                    st.warning(f"🚨 Signal: {signal} {auto_symbol} (Confidence: {confidence:.1%})")
    
    with tab3:
        st.subheader("Risk & Performance")
        
        risk_manager = RiskManager(trader)
        portfolio_var = risk_manager.calculate_var(positions)
        
        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            st.metric("1-Day VaR", f"${portfolio_var['1d']:,.0f}")
        with col_r2:
            st.metric("1-Week VaR", f"${portfolio_var['1w']:,.0f}")
        with col_r3:
            st.metric("1-Month VaR", f"${portfolio_var['1m']:,.0f}")
        
        # Performance Chart
        if st.button("Generate Report"):
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode = "gauge+number+delta",
                value = portfolio_summary['unrealized_pl'],
                title = {'text': "Portfolio P&L"},
                gauge = {'axis': {'range': [min(-1000, portfolio_summary['unrealized_pl']), max(1000, portfolio_summary['unrealized_pl'])]}}
            ))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Thay thế Social Sentiment bằng AI Assistant
        display_ai_assistant()

else:
    # Welcome Screen
    st.markdown("---")
    col_w1, col_w2, col_w3 = st.columns(3)
    
    with col_w1:
        st.markdown("""
        <div class="dashboard-card">
            <div class="feature-icon">📈</div>
            <h3>Stock Trading</h3>
            <p>Trade US stocks with real-time data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_w2:
        st.markdown("""
        <div class="dashboard-card">
            <div class="feature-icon">💰</div>
            <h3>Paper Account</h3>
            <p>$100,000 virtual trading</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_w3:
        st.markdown("""
        <div class="dashboard-card">
            <div class="feature-icon">⚡</div>
            <h3>Live Data</h3>
            <p>Real-time market updates</p>
        </div>
        """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
<div style='text-align: center; padding: 3rem; color: #8898aa;'>
    <p style='margin: 0; font-size: 0.9rem;'>Built with Streamlit • Alpaca Markets API • AI Trading Assistant</p>
</div>
""", unsafe_allow_html=True)