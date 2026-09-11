import streamlit as st
import requests
import pandas as pd
import numpy as np

# Page Configuration for Mobile View
st.set_page_config(
    page_title="Trade Decision Engine",
    page_icon="📈",
    layout="centered"
)

# Custom Dark Mode & High-Contrast CSS Styling
st.markdown("""
    <style>
    .stApp { background-color: #0b0b0e; color: #ffffff; }
    h1, h2, h3 { color: #00d2ff !important; text-align: uppercase; letter-spacing: 1.5px; }
    .metric-container { background: #22222d; border-radius: 12px; padding: 14px; border: 1px solid #333342; text-align: center; }
    .decision-wait { color: #ffe600; font-size: 3rem; font-weight: 900; text-align: center; text-shadow: 0 0 20px rgba(255,230,0,0.4); }
    .decision-exit { color: #ff2d55; font-size: 3rem; font-weight: 900; text-align: center; text-shadow: 0 0 20px rgba(255,45,85,0.5); }
    .decision-reenter { color: #30d158; font-size: 3rem; font-weight: 900; text-align: center; text-shadow: 0 0 20px rgba(48,209,88,0.5); }
    .alert-box { background: rgba(255, 45, 85, 0.25); border: 2px solid #ff2d55; color: #ff2d55; padding: 12px; border-radius: 10px; font-weight: bold; text-align: center; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2>Decision Engine</h2>", unsafe_allow_html=True)

# Load API Key automatically from Streamlit Secrets
api_key = st.secrets.get("TWELVE_DATA_API_KEY", "")

# User Inputs
col1, col2 = st.columns(2)
with col1:
    ticker_input = st.text_input("Ticker", value="SOXL").upper().strip()

# Precise Indicator Engine using Wilder's Smoothing (TradingView Standard)
def compute_tradingview_style_indicators(df, window=14):
    close = df['Close']
    delta = close.diff()
    
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    ema20 = close.ewm(span=20, adjust=False).mean()
    
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    
    return rsi.iloc[-1], ema20.iloc[-1], macd.iloc[-1], histogram.iloc[-1]

# Fetch data via Twelve Data REST API
@st.cache_data(ttl=60)
def fetch_twelve_data(symbol, interval, key):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=100&apikey={key}"
    try:
        response = requests.get(url).json()
        if "values" in response:
            df = pd.DataFrame(response["values"])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
            df = df.sort_index()
            df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
            return df
        return None
    except Exception:
        return None

# Load Market Data
market_data = None
if api_key:
    df_1d = fetch_twelve_data(ticker_input, "1day", api_key)
    df_4h = fetch_twelve_data(ticker_input, "4h", api_key)
    df_1h = fetch_twelve_data(ticker_input, "1h", api_key)
    
    if df_1d is not None and df_4h is not None and df_1h is not None:
        price = float(df_1d['Close'].iloc[-1])
        rsi_1d, ema_1d, _, _ = compute_tradingview_style_indicators(df_1d)
        rsi_1h, _, _, _ = compute_tradingview_style_indicators(df_1h)
        _, _, macd_4h, hist_4h = compute_tradingview_style_indicators(df_4h)
        
        market_data = {
            "price": price,
            "rsi_1d": rsi_1d,
            "ema_1d": ema_1d,
            "macd_4h": macd_4h,
            "hist_4h": hist_4h,
            "rsi_1h": rsi_1h
        }

current_price = market_data["price"] if market_data else 115.76

with col2:
    price_input = st.number_input("Current Price ($)", value=round(current_price, 2), step=0.01)

# Default risk rule: 10% for leveraged 3x ETFs, 5% for standard stocks
leveraged_assets = ['SOXL', 'TECL', 'TQQQ', 'UPRO', 'FAS']
default_buffer = 0.10 if ticker_input in leveraged_assets else 0.05
default_stop = round(price_input * (1 - default_buffer), 2)

stop_level = st.number_input("Stop Level ($)", value=default_stop, step=0.01)

rsi_1h_val = market_data["rsi_1h"] if market_data else 42.5
hist_4h_val = market_data["hist_4h"] if market_data else -0.29
ema_1d_val = market_data["ema_1d"] if market_data else price_input

# Decision Logic & Dynamic Threshold Rule Building
decision = "WAIT"
border_color = "#ffe600"
is_alert = False

if not api_key:
    summary_markdown = "⚠️ **Error:** `TWELVE_DATA_API_KEY` not found in Streamlit Secrets. Please configure your secrets."
elif price_input < stop_level:
    decision = "EXIT"
    is_alert = True
    summary_markdown = f"""
**CRITICAL EXIT TRIGGER**
* **Trigger Event:** Current price (${price_input}) breached your stop loss floor (**${stop_level}**).
* **Action:** Leveraged volatility decay overrides oversold indicators. Execute exit immediately.
    """
    border_color = "#ff2d55"
elif market_data and price_input > ema_1d_val and hist_4h_val > 0 and ticker_input not in leveraged_assets:
    decision = "RE-ENTER"
    summary_markdown = f"""
**RE-ENTER SIGNAL CONFIRMED**
* **Condition Met:** Price (${round(price_input, 2)}) is trading above the 1-Day 20 EMA (${round(ema_1d_val, 2)}) with positive 4-Hr momentum.
    """
    border_color = "#30d158"
else:
    decision = "WAIT"
    needed_price = round(ema_1d_val, 2)
    summary_markdown = f"""
**STATUS: WAIT / HOLD** (Macro constraints active)

**Exact Thresholds Required to Change Signal:**
* **To Shift Bullish / Re-enter:** 1-Day price must close above **${needed_price}** (20 EMA) **AND** 4-Hr MACD Histogram must cross above **0.00** (currently `{round(hist_4h_val, 2)}`).
* **Execution Watch:** Monitor 1-Hr RSI (currently `{round(rsi_1h_val, 1)}`). Look for a drop below **40.0** for deep-value entries or a break above **50.0** for early momentum confirmation.
* **Capital Protection:** Active stop level floor set at **${stop_level}**.
    """
    border_color = "#ffe600"

# Render Alert Banner if Triggered
if is_alert:
    st.markdown(f'<div class="alert-box">⚠ RISK OVERRIDE: Price breached support threshold (${stop_level}).</div>', unsafe_allow_html=True)

# Render Primary Decision Card
st.markdown("---")
if decision == "EXIT":
    st.markdown(f'<div class="decision-exit">{decision}</div>', unsafe_allow_html=True)
elif decision == "RE-ENTER":
    st.markdown(f'<div class="decision-reenter">{decision}</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="decision-wait">{decision}</div>', unsafe_allow_html=True)

st.markdown("---")

# Render Clean Multi-Timeframe Breakdown (Consistent Live Values Only)
st.markdown("### Multi-Timeframe Technical Breakdown")
if market_data:
    col_a, col_b, col_c = st.columns(3)
    
    # 1-Day Trend
    d1_bullish = market_data['price'] > ema_1d_val
    d1_trend = "Bullish" if d1_bullish else "Bearish"
    d1_color = "#30d158" if d1_bullish else "#ff2d55"
    
    # 4-Hr Trend
    h4_bullish = hist_4h_val > 0
    h4_trend = "Bullish" if h4_bullish else "Bearish"
    h4_color = "#30d158" if h4_bullish else "#ff2d55"
    
    # 1-Hr Trend (Adjusted thresholds: <40 oversold, >50 bullish momentum)
    if rsi_1h_val < 40:
        h1_trend = "Oversold"
        h1_color = "#ffe600"
    elif rsi_1h_val > 50:
        h1_trend = "Bullish Momentum"
        h1_color = "#30d158"
    else:
        h1_trend = "Neutral"
        h1_color = "#a0a0b0"

    with col_a:
        st.markdown(f"""
            <div class="metric-container">
                <div style="font-size:0.7rem; color:#a0a0b0; letter-spacing:1px;">1-DAY (MACRO)</div>
                <div style="font-size:1.0rem; font-weight:bold; color:{d1_color}; margin-top:6px;">{d1_trend}</div>
                <div style="font-size:0.75rem; color:#ffffff; margin-top:6px;">RSI: {round(market_data['rsi_1d'], 1)}</div>
                <div style="font-size:0.7rem; color:#8e8e93; margin-top:2px;">EMA20: ${round(ema_1d_val, 1)}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col_b:
        st.markdown(f"""
            <div class="metric-container">
                <div style="font-size:0.7rem; color:#a0a0b0; letter-spacing:1px;">4-HR (MOMENTUM)</div>
                <div style="font-size:1.0rem; font-weight:bold; color:{h4_color}; margin-top:6px;">{h4_trend}</div>
                <div style="font-size:0.75rem; color:#ffffff; margin-top:6px;">MACD Hist: {round(hist_4h_val, 2)}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col_c:
        st.markdown(f"""
            <div class="metric-container">
                <div style="font-size:0.7rem; color:#a0a0b0; letter-spacing:1px;">1-HR (EXECUTION)</div>
                <div style="font-size:1.0rem; font-weight:bold; color:{h1_color}; margin-top:6px;">{h1_trend}</div>
                <div style="font-size:0.75rem; color:#ffffff; margin-top:6px;">RSI: {round(rsi_1h_val, 1)}</div>
            </div>
        """, unsafe_allow_html=True)
else:
    st.warning("Unable to fetch data streams. Verify your API key configuration.")

# Recommendation Box with Clean Markdown Rendering
st.markdown(f"""
    <div style="margin-top: 20px; font-size: 0.85rem; color: #aeaeb2; line-height: 1.6; padding: 15px; background: #16161c; border-radius: 12px; border-left: 4px solid {border_color};">
        {summary_markdown}
    </div>
""", unsafe_allow_html=True)
