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
    .card { background-color: #16161c; border: 1px solid #2a2a35; border-radius: 20px; padding: 20px; margin-bottom: 15px; box-shadow: 0 8px 25px rgba(0,0,0,0.7); }
    h1, h2, h3 { color: #00d2ff !important; text-align: uppercase; letter-spacing: 1.5px; }
    .metric-container { background: #22222d; border-radius: 12px; padding: 12px; border: 1px solid #333342; text-align: center; }
    .decision-wait { color: #ffe600; font-size: 3rem; font-weight: 900; text-align: center; text-shadow: 0 0 20px rgba(255,230,0,0.4); }
    .decision-exit { color: #ff2d55; font-size: 3rem; font-weight: 900; text-align: center; text-shadow: 0 0 20px rgba(255,45,85,0.5); }
    .decision-reenter { color: #30d158; font-size: 3rem; font-weight: 900; text-align: center; text-shadow: 0 0 20px rgba(48,209,88,0.5); }
    .alert-box { background: rgba(255, 45, 85, 0.25); border: 2px solid #ff2d55; color: #ff2d55; padding: 12px; border-radius: 10px; font-weight: bold; text-align: center; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2>Decision Engine (Threshold-Driven)</h2>", unsafe_allow_html=True)

# Sidebar for API Key (with Secrets fallback) and Calibration
with st.sidebar:
    st.markdown("### API Configuration")
    default_key = st.secrets.get("TWELVE_DATA_API_KEY", "")
    api_key = st.text_input("Twelve Data API Key", value=default_key, type="password", help="Get a free key at twelvedata.com")
    
    st.markdown("### Feed Calibration")
    rsi_offset = st.slider("1-Hr RSI Offset", -10.0, 10.0, 0.0, 0.5, 
                           help="Fine-tunes 1H RSI to match TradingView exactly.")
    macd_offset = st.slider("4-Hr MACD Hist Offset", -1.0, 1.0, 0.0, 0.05, 
                            help="Fine-tunes 4H MACD histogram value.")

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

# Apply user calibration offsets
calibrated_rsi_1h = (market_data["rsi_1h"] + rsi_offset) if market_data else 42.46
base_hist = market_data["hist_4h"] if market_data else -0.29
calibrated_hist_4h = base_hist + macd_offset
ema_1d_val = market_data["ema_1d"] if market_data else price_input

# Decision Logic & Dynamic Threshold Rule Building
decision = "WAIT"
summary_text = ""
border_color = "#ffe600"
is_alert = False

if not api_key:
    summary_text = "Please enter your Twelve Data API key in the sidebar to activate live feeds."
elif price_input < stop_level:
    decision = "EXIT"
    is_alert = True
    summary_text = f"<b>CRITICAL EXIT TRIGGER:</b> Current price (${price_input}) has breached your stop level threshold (<b>${stop_level}</b>). Leveraged volatility decay overrides oversold metrics. Execute exit immediately."
    border_color = "#ff2d55"
elif market_data and price_input > ema_1d_val and calibrated_hist_4h > 0 and ticker_input not in leveraged_assets:
    decision = "RE-ENTER"
    summary_text = f"<b>RE-ENTER SIGNAL CONFIRMED:</b> Price (${round(price_input, 2)}) is above the 1-Day 20 EMA (${round(ema_1d_val, 2)}) and 4-Hr MACD histogram is positive ({round(calibrated_hist_4h, 2)})."
    border_color = "#30d158"
else:
    decision = "WAIT"
    # Detail exact technical thresholds needed to trigger changes
    needed_price = round(ema_1d_val, 2)
    summary_text = f"""
        <b>STATUS: WAIT / HOLD.</b> Macro constraints are active.<br><br>
        <u>Explicit Thresholds Required to Change Signal:</u>
        <br>• <b>To Shift Bullish / Re-enter:</b> 1-Day Price must close above <b>${needed_price}</b> (20 EMA) AND 4-Hr MACD Histogram must cross above <b>0.00</b> (currently {round(calibrated_hist_4h, 2)}).
        <br>• <b>Execution Timing:</b> Look for 1-Hr RSI (currently {round(calibrated_rsi_1h, 1)}) to drop below <b>40.0</b> for deep-value entry or clear above <b>60.0</b> for momentum confirmation.
        <br>• <b>Capital Protection:</b> Active stop level floor set at <b>${stop_level}</b>.
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

# Render Multi-Timeframe Breakdown (Trend + Technical Value Combined)
st.markdown("### Multi-Timeframe Technical Breakdown")
if market_data:
    col_a, col_b, col_c = st.columns(3)
    
    # 1-Day Trend + Values
    d1_is_bullish = market_data['price'] > ema_1d_val
    d1_trend = "Bullish" if d1_is_bullish else "Bearish"
    d1_color = "#30d158" if d1_is_bullish else "#ff2d55"
    
    # 4-Hr Trend + Values
    h4_is_bullish = calibrated_hist_4h > 0
    h4_trend = "Bullish" if h4_is_bullish else "Bearish"
    h4_color = "#30d158" if h4_is_bullish else "#ff2d55"
    
    # 1-Hr Trend + Values
    if calibrated_rsi_1h < 40:
        h1_trend = "Oversold"
        h1_color = "#ffe600"
    elif calibrated_rsi_1h > 60:
        h1_trend = "Overbought"
        h1_color = "#00d2ff"
    else:
        h1_trend = "Neutral"
        h1_color = "#a0a0b0"

    with col_a:
        st.markdown(f"""
            <div class="metric-container">
                <div style="font-size:0.7rem; color:#a0a0b0; letter-spacing:1px;">1-DAY (MACRO)</div>
                <div style="font-size:1.0rem; font-weight:bold; color:{d1_color}; margin-top:4px;">{d1_trend}</div>
                <div style="font-size:0.75rem; color:#ffffff; margin-top:4px;">Price vs EMA20</div>
                <div style="font-size:0.7rem; color:#8e8e93;">RSI: {round(market_data['rsi_1d'], 1)}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col_b:
        st.markdown(f"""
            <div class="metric-container">
                <div style="font-size:0.7rem; color:#a0a0b0; letter-spacing:1px;">4-HR (MOMENTUM)</div>
                <div style="font-size:1.0rem; font-weight:bold; color:{h4_color}; margin-top:4px;">{h4_trend}</div>
                <div style="font-size:0.75rem; color:#ffffff; margin-top:4px;">MACD Hist</div>
                <div style="font-size:0.7rem; color:#8e8e93;">{round(calibrated_hist_4h, 2)}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col_c:
        st.markdown(f"""
            <div class="metric-container">
                <div style="font-size:0.7rem; color:#a0a0b0; letter-spacing:1px;">1-HR (EXECUTION)</div>
                <div style="font-size:1.0rem; font-weight:bold; color:{h1_color}; margin-top:4px;">{h1_trend}</div>
                <div style="font-size:0.75rem; color:#ffffff; margin-top:4px;">RSI State</div>
                <div style="font-size:0.7rem; color:#8e8e93;">RSI: {round(calibrated_rsi_1h, 1)}</div>
            </div>
        """, unsafe_allow_html=True)
else:
    st.warning("Please enter your Twelve Data API key in the sidebar to load indicators.")

# Recommendation Box with Clear Thresholds
st.markdown(f"""
    <div style="margin-top: 20px; font-size: 0.85rem; color: #aeaeb2; line-height: 1.6; padding: 15px; background: #16161c; border-radius: 12px; border-left: 4px solid {border_color};">
        {summary_text}
    </div>
""", unsafe_allow_html=True)
