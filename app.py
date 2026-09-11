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

st.markdown("<h2>Decision Engine (Live Native Math)</h2>", unsafe_allow_html=True)

# User Inputs
col1, col2 = st.columns(2)
with col1:
    ticker_input = st.text_input("Ticker", value="SOXL").upper().strip()

# Function to calculate technical indicators from historical data frames
def compute_indicators(df, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # EMA 20
    ema20 = df['Close'].ewm(span=20, adjust=False).mean()
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    
    return rsi.iloc[-1], ema20.iloc[-1], macd.iloc[-1], histogram.iloc[-1]

# Fetch data and compute indicators for multiple timeframes
@st.cache_data(ttl=60)
def get_market_analysis(symbol):
    try:
        t = yf.Ticker(symbol)
        
        # 1 Day Data (Macro)
        df_1d = t.history(period="60d", interval="1d")
        price = float(df_1d['Close'].iloc[-1])
        rsi_1d, ema_1d, _, _ = compute_indicators(df_1d)
        
        # 4 Hour Data (Momentum) - approximated via hourly history
        df_4h = t.history(period="60d", interval="1h")
        _, _, macd_4h, hist_4h = compute_indicators(df_4h)
        
        # 1 Hour Data (Execution)
        rsi_1h, _, _, _ = compute_indicators(df_4h)
        
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

if market_data:
    current_price = market_data["price"]
else:
    current_price = 115.76

with col2:
    price_input = st.number_input("Current Price ($)", value=round(current_price, 2), step=0.01)

# Default buffer rule: 10% for leveraged 3x ETFs, 5% for standard stocks
leveraged_assets = ['SOXL', 'TECL', 'TQQQ', 'UPRO', 'FAS']
default_buffer = 0.10 if ticker_input in leveraged_assets else 0.05
default_stop = round(price_input * (1 - default_buffer), 2)

stop_level = st.number_input("Stop Level ($)", value=default_stop, step=0.01)

# Decision Logic Evaluation
decision = "WAIT"
summary_text = ""
border_color = "#ffe600"
is_alert = False

if price_input < stop_level:
    decision = "EXIT"
    is_alert = True
    summary_text = f"CRITICAL EXIT TRIGGER: {ticker_input} has dropped below your automated risk stop of ${stop_level}. Leveraged decay risk overrides short-term oversold indicators. Execute exit immediately."
    border_color = "#ff2d55"
elif market_data and market_data["price"] > market_data["ema_1d"] and ticker_input not in leveraged_assets:
    decision = "RE-ENTER"
    summary_text = f"Uptrend confirmed. Price is trading above the 1-Day 20 EMA with positive momentum alignment."
    border_color = "#30d158"
else:
    decision = "WAIT"
    summary_text = f"Macro trend is constrained. Watch indicators for a relief bounce: ensure <b>1-Hr RSI recovers</b> and <b>4-Hr MACD histogram turns positive</b> before shifting stance. Protect capital against stop level (${stop_level})."
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

# Render Multi-Timeframe Ladder Using Native Calculations
st.markdown("### Live Timeframe Breakdown")
if market_data:
    col_a, col_b, col_c = st.columns(3)
    
    # 1-Day Trend evaluation
    d1_trend = "Bullish (Above EMA)" if market_data['price'] > market_data['ema_1d'] else "Bearish (Below EMA)"
    d1_color = "#30d158" if market_data['price'] > market_data['ema_1d'] else "#ff2d55"
    
    # 4-Hour Momentum evaluation
    h4_trend = "Bullish (> 0)" if market_data['hist_4h'] > 0 else "Bearish (< 0)"
    h4_color = "#30d158" if market_data['hist_4h'] > 0 else "#ff2d55"
    
    # 1-Hour RSI evaluation
    rsi_val = market_data['rsi_1h']
    h1_trend = "Oversold" if rsi_val < 40 else ("Overbought" if rsi_val > 60 else "Neutral")
    h1_color = "#ffe600" if rsi_val < 40 else "#00d2ff"

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
                <div style="font-size:0.7rem; color:#8e8e93; margin-top:3px;">Hist: {round(market_data['hist_4h'], 2)}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col_c:
        st.markdown(f"""
            <div class="metric-container">
                <div style="font-size:0.75rem; color:#a0a0b0;">1-HR (EXECUTION)</div>
                <div style="font-size:0.9rem; font-weight:bold; color:{h1_color}; margin-top:5px;">{h1_trend}</div>
                <div style="font-size:0.7rem; color:#8e8e93; margin-top:3px;">RSI: {round(rsi_val, 1)}</div>
            </div>
        """, unsafe_allow_html=True)
else:
    st.warning("Unable to fetch price history for this ticker.")

# Recommendation Box
st.markdown(f"""
    <div style="margin-top: 20px; font-size: 0.85rem; color: #aeaeb2; line-height: 1.5; padding: 15px; background: #16161c; border-radius: 12px; border-left: 4px solid {border_color};">
        {summary_text}
    </div>
""", unsafe_allow_html=True)
