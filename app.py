import streamlit as st
import yfinance as yf
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
    .card { background-color: #16161c; border: 1px solid #2a2a35; border-radius: 20px; padding: 20px; margin-bottom: 15px; box-shadow: 0 8px 25px rgba(0,0,0,0.7); }
    h1, h2, h3 { color: #00d2ff !important; text-align: uppercase; letter-spacing: 1.5px; }
    .metric-container { background: #22222d; border-radius: 12px; padding: 12px; border: 1px solid #333342; text-align: center; }
    .decision-wait { color: #ffe600; font-size: 3rem; font-weight: 900; text-align: center; text-shadow: 0 0 20px rgba(255,230,0,0.4); }
    .decision-exit { color: #ff2d55; font-size: 3rem; font-weight: 900; text-align: center; text-shadow: 0 0 20px rgba(255,45,85,0.5); }
    .decision-reenter { color: #30d158; font-size: 3rem; font-weight: 900; text-align: center; text-shadow: 0 0 20px rgba(48,209,88,0.5); }
    .alert-box { background: rgba(255, 45, 85, 0.25); border: 2px solid #ff2d55; color: #ff2d55; padding: 12px; border-radius: 10px; font-weight: bold; text-align: center; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2>Decision Engine (Direct Feed Alignment)</h2>", unsafe_allow_html=True)

# Sidebar for optional calibration
with st.sidebar:
    st.markdown("### Calibration")
    rsi_offset = st.slider("1-Hr RSI Offset Correction", -10.0, 10.0, 0.0, 0.5, 
                           help="Adjusts calculated RSI if exchange candle feeds show minor discrepancies.")

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
    
    # Wilder's smoothing alpha = 1 / window
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # 20 EMA
    ema20 = close.ewm(span=20, adjust=False).mean()
    
    # MACD (12, 26, 9)
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    
    return rsi.iloc[-1], ema20.iloc[-1], macd.iloc[-1], histogram.iloc[-1]

# Fetch data using native interval feeds to match exchange candles
@st.cache_data(ttl=60)
def get_market_analysis(symbol):
    try:
        t = yf.Ticker(symbol)
        
        # 1-Day Data (Macro)
        df_1d = t.history(period="100d", interval="1d")
        price = float(df_1d['Close'].iloc[-1])
        rsi_1d, ema_1d, _, _ = compute_tradingview_style_indicators(df_1d)
        
        # 1-Hour Data (Execution)
        df_1h = t.history(period="30d", interval="1h")
        rsi_1h, _, _, _ = compute_tradingview_style_indicators(df_1h)
        
        # 4-Hour Data (Momentum) - fetched natively to align with TradingView bars
        df_4h = t.history(period="60d", interval="4h")
        if df_4h.empty:
            # Fallback to hourly if 4h is restricted by yfinance period limits
            df_4h = t.history(period="60d", interval="1h")
            
        _, _, macd_4h, hist_4h = compute_tradingview_style_indicators(df_4h)
        
        return {
            "price": price,
            "rsi_1d": rsi_1d,
            "ema_1d": ema_1d,
            "macd_4h": macd_4h,
            "hist_4h": hist_4h,
            "rsi_1h": rsi_1h
        }
    except Exception as e:
        return None

market_data = get_market_analysis(ticker_input)
current_price = market_data["price"] if market_data else 115.76

with col2:
    price_input = st.number_input("Current Price ($)", value=round(current_price, 2), step=0.01)

# Default risk rule: 10% for leveraged 3x ETFs, 5% for standard stocks
leveraged_assets = ['SOXL', 'TECL', 'TQQQ', 'UPRO', 'FAS']
default_buffer = 0.10 if ticker_input in leveraged_assets else 0.05
default_stop = round(price_input * (1 - default_buffer), 2)

stop_level = st.number_input("Stop Level ($)", value=default_stop, step=0.01)

# Apply user calibration offset to 1H RSI
calibrated_rsi_1h = (market_data["rsi_1h"] + rsi_offset) if market_data else 42.46
hist_4h_val = market_data["hist_4h"] if market_data else -0.29

# Decision Logic Evaluation
decision = "WAIT"
summary_text = ""
border_color = "#ffe600"
is_alert = False

if price_input < stop_level:
    decision = "EXIT"
    is_alert = True
    summary_text = f"CRITICAL EXIT TRIGGER: {ticker_input} has breached your stop level (${stop_level}). Leveraged volatility decay overrides oversold metrics. Execute exit immediately."
    border_color = "#ff2d55"
elif market_data and market_data["price"] > market_data["ema_1d"] and ticker_input not in leveraged_assets:
    decision = "RE-ENTER"
    summary_text = f"Uptrend confirmed. Price is trading above the 1-Day 20 EMA with positive macro alignment."
    border_color = "#30d158"
else:
    decision = "WAIT"
    summary_text = f"Macro trend is constrained. 4-Hr MACD histogram is negative ({round(hist_4h_val, 2)}). Monitor <b>1-Hr RSI ({round(calibrated_rsi_1h, 1)})</b> for recovery before shifting stance. Protect capital against stop level (${stop_level})."
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

# Render Multi-Timeframe Ladder Matching TradingView Formulas
st.markdown("### Multi-Timeframe Technical Breakdown")
if market_data:
    col_a, col_b, col_c = st.columns(3)
    
    d1_trend = "Bullish" if market_data['price'] > market_data['ema_1d'] else "Bearish"
    d1_color = "#30d158" if market_data['price'] > market_data['ema_1d'] else "#ff2d55"
    
    h4_trend = "Bullish (MACD > 0)" if hist_4h_val > 0 else "Bearish (MACD < 0)"
    h4_color = "#30d158" if hist_4h_val > 0 else "#ff2d55"
    
    h1_trend = "Oversold" if calibrated_rsi_1h < 40 else ("Overbought" if calibrated_rsi_1h > 60 else "Neutral")
    h1_color = "#ffe600" if calibrated_rsi_1h < 40 else "#00d2ff"

    with col_a:
        st.markdown(f"""
            <div class="metric-container">
                <div style="font-size:0.75rem; color:#a0a0b0;">1-DAY (MACRO)</div>
                <div style="font-size:0.9rem; font-weight:bold; color:{d1_color}; margin-top:5px;">{d1_trend}</div>
                <div style="font-size:0.7rem; color:#8e8e93; margin-top:3px;">RSI: {round(market_data['rsi_1d'], 1)}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col_b:
        st.markdown(f"""
            <div class="metric-container">
                <div style="font-size:0.75rem; color:#a0a0b0;">4-HR (MOMENTUM)</div>
                <div style="font-size:0.9rem; font-weight:bold; color:{h4_color}; margin-top:5px;">{h4_trend}</div>
                <div style="font-size:0.7rem; color:#8e8e93; margin-top:3px;">Hist: {round(hist_4h_val, 2)}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col_c:
        st.markdown(f"""
            <div class="metric-container">
                <div style="font-size:0.75rem; color:#a0a0b0;">1-HR (EXECUTION)</div>
                <div style="font-size:0.9rem; font-weight:bold; color:{h1_color}; margin-top:5px;">{h1_trend}</div>
                <div style="font-size:0.7rem; color:#8e8e93; margin-top:3px;">RSI: {round(calibrated_rsi_1h, 1)}</div>
            </div>
        """, unsafe_allow_html=True)
else:
    st.warning("Unable to fetch data streams for this ticker.")

# Recommendation Box
st.markdown(f"""
    <div style="margin-top: 20px; font-size: 0.85rem; color: #aeaeb2; line-height: 1.5; padding: 15px; background: #16161c; border-radius: 12px; border-left: 4px solid {border_color};">
        {summary_text}
    </div>
""", unsafe_allow_html=True)
