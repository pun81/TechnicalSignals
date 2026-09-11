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

with col2:
    include_after_hours = st.checkbox("Include After-Hours (ETH)", value=True, help="Check to include pre-market and after-hours ticks on intraday intervals.")

# Asset Class Profile Definition
leveraged_assets = ['SOXL', 'TECL', 'TQQQ', 'UPRO', 'FAS']
is_leveraged = ticker_input in leveraged_assets

# Adaptive Thresholds Based on Asset Class
if is_leveraged:
    min_macd_hist = 0.05
    min_rsi_execution = 55.0
    profile_label = "⚡ **Asset Profile:** 3x Leveraged ETF (Stricter Filters Active)"
else:
    min_macd_hist = 0.00
    min_rsi_execution = 50.0
    profile_label = "📊 **Asset Profile:** Standard Equity / Single Stock Swing Profile"

# Precise Indicator Engine using Wilder's Smoothing (TradingView Standard)
def compute_tradingview_style_indicators(df, window=14):
    if df is None or len(df) < window:
        return 50.0, 0.0, 0.0, 0.0
        
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

# Automated 4H Structural Support Calculation
def find_structural_support(df_4h, is_leveraged):
    if df_4h is None or len(df_4h) < 10:
        return None
    recent_lows = df_4h['Low'].tail(30)
    raw_support = recent_lows.min()
    
    if is_leveraged:
        return round(raw_support * 0.985, 2)
    else:
        return round(raw_support, 2)

# Fetch historical time series safely with strict interval rules
@st.cache_data(ttl=60)
def fetch_twelve_data(symbol, interval, key, include_eth):
    # Twelve Data only permits prepost=true on intraday intervals <= 30min (1min, 5min, 15min, 30min).
    if interval == "1h":
        if include_eth:
            interval = "30min"  # Switch to 30min to lawfully accept pre/post data
            prepost_param = "true"
        else:
            prepost_param = "false"
    else:
        # Macro intervals (4h, 1day) do not support pre/post parameters on Twelve Data
        prepost_param = "false"

    if prepost_param == "true":
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=150&prepost=true&apikey={key}"
    else:
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=150&apikey={key}"
        
    try:
        response = requests.get(url).json()
        if "values" in response and len(response["values"]) > 0:
            df = pd.DataFrame(response["values"])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
            df = df.sort_index()
            df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
            
            return df, None
        err_msg = response.get("message", f"No values returned for {interval}")
        return None, err_msg
    except Exception as e:
        return None, str(e)

# Load Market Data & Capture Specific Errors
market_data = None
df_4h_data = None
live_price_val = None
api_error_log = []

if not api_key:
    api_error_log.append("`TWELVE_DATA_API_KEY` is missing from your Streamlit Secrets.")
else:
    df_execution_raw, err_exec = fetch_twelve_data(ticker_input, "1h", api_key, include_after_hours)
    df_1d, err_1d = fetch_twelve_data(ticker_input, "1day", api_key, include_after_hours)
    df_4h_data, err_4h = fetch_twelve_data(ticker_input, "4h", api_key, include_after_hours)

    if err_1d: api_error_log.append(f"1-Day Data Error: {err_1d}")
    if err_4h: api_error_log.append(f"4-Hour Data Error: {err_4h}")
    if err_exec: api_error_log.append(f"Execution Stream Data Error: {err_exec}")
    
    if df_1d is not None and df_4h_data is not None and df_execution_raw is not None and len(df_1d) > 0:
        # Extract live price dynamically from the latest tick
        live_price_val = float(df_execution_raw['Close'].iloc[-1])
        
        # Prepare indicator dataframes by dropping the unclosed live candle for alignment
        df_exec_indicators = df_execution_raw.iloc[:-1] if len(df_execution_raw) > 1 else df_execution_raw
        df_1d_indicators = df_1d.iloc[:-1] if len(df_1d) > 1 else df_1d
        df_4h_indicators = df_4h_data.iloc[:-1] if len(df_4h_data) > 1 else df_4h_data

        rsi_1d, ema_1d, _, _ = compute_tradingview_style_indicators(df_1d_indicators)
        rsi_exec, _, _, _ = compute_tradingview_style_indicators(df_exec_indicators)
        _, _, macd_4h, hist_4h = compute_tradingview_style_indicators(df_4h_indicators)
        
        market_data = {
            "price": live_price_val,
            "rsi_1d": rsi_1d,
            "ema_1d": ema_1d,
            "macd_4h": macd_4h,
            "hist_4h": hist_4h,
            "rsi_1h": rsi_exec
        }

current_price = market_data["price"] if market_data else 115.76

# Price Input Field
price_input = st.number_input("Current Price ($)", value=round(current_price, 2), step=0.01)

# Dynamically calculate default stop from structural 4H support, with fallback
calculated_support = find_structural_support(df_4h_data, is_leveraged)
default_stop = calculated_support if calculated_support else round(price_input * (0.90 if is_leveraged else 0.95), 2)

stop_level = st.number_input("Stop Level (Structural Support)", value=default_stop, step=0.01)

rsi_1h_val = market_data["rsi_1h"] if market_data else 42.5
rsi_1d_val = market_data["rsi_1d"] if market_data else 45.0
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
* **Trigger Event:** Current price (${price_input}) breached your 4H structural support floor (**${stop_level}**).
* **Action:** { "Leveraged volatility decay overrides oversold indicators. Execute exit immediately." if is_leveraged else "Support level broken. Protect capital and exit position." }
    """
    border_color = "#ff2d55"
else:
    decision = "WAIT"
    needed_price = round(ema_1d_val, 2)
    summary_markdown = f"""
{profile_label}<br><br>
**STATUS: WAIT / HOLD** (Adaptive thresholds active)<br><br>
**Exact Thresholds Required to Change Signal:**
* **To Shift Bullish / Re-enter:** 1-Day price must close above **${needed_price}** (20 EMA) **AND** 4-Hr MACD Histogram must exceed **{min_macd_hist}** (currently `{round(hist_4h_val, 2)}`).
* **Macro Confirmation:** Monitor 1-Day RSI (currently `{round(rsi_1d_val, 1)}`). Look for it to push back above **50.0** to validate daily momentum.
* **Execution Watch:** Monitor Intraday RSI (currently `{round(rsi_1h_val, 1)}`). Look for a break above **{min_rsi_execution}** to confirm momentum conviction.
* **Capital Protection:** Active stop level anchored to 4H structural support at **${stop_level}**.
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

# Render Clean Multi-Timeframe Breakdown
st.markdown("### Multi-Timeframe Technical Breakdown")
if market_data:
    col_a, col_b, col_c = st.columns(3)
    
    # 1-Day Trend
    d1_bullish = market_data['price'] > ema_1d_val
    d1_trend = "Bullish" if d1_bullish else "Bearish"
    d1_color = "#30d158" if d1_bullish else "#ff2d55"
    
    # 4-Hr Trend
    h4_bullish = hist_4h_val > min_macd_hist
    h4_trend = "Bullish Conviction" if h4_bullish else ("Transitioning" if hist_4h_val > 0 else "Bearish")
    h4_color = "#30d158" if h4_bullish else ("#ffe600" if hist_4h_val > 0 else "#ff2d55")
    
    # Execution Trend
    if rsi_1h_val < 40:
        h1_trend = "Oversold"
        h1_color = "#ffe600"
    elif rsi_1h_val >= min_rsi_execution:
        h1_trend = "Bullish Momentum"
        h1_color = "#30d158"
    else:
        h1_trend = "Neutral / Waiting"
        h1_color = "#a0a0b0"

    with col_a:
        st.markdown(f"""
            <div class="metric-container">
                <div style="font-size:0.7rem; color:#a0a0b0; letter-spacing:1px;">1-DAY (MACRO)</div>
                <div style="font-size:1.0rem; font-weight:bold; color:{d1_color}; margin-top:6px;">{d1_trend}</div>
                <div style="font-size:0.75rem; color:#ffffff; margin-top:6px;">RSI: {round(rsi_1d_val, 1)}</div>
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
                <div style="font-size:0.7rem; color:#a0a0b0; letter-spacing:1px;">EXECUTION STREAM</div>
                <div style="font-size:1.0rem; font-weight:bold; color:{h1_color}; margin-top:6px;">{h1_trend}</div>
                <div style="font-size:0.75rem; color:#ffffff; margin-top:6px;">RSI: {round(rsi_1h_val, 1)}</div>
            </div>
        """, unsafe_allow_html=True)
else:
    st.error("Unable to fetch data streams. API Error Logs:")
    for err in api_error_log:
        st.code(err)

# Recommendation Box with Clean Markdown Rendering
st.markdown(f"""
    <div style="margin-top: 20px; font-size: 0.85rem; color: #aeaeb2; line-height: 1.6; padding: 15px; background: #16161c; border-radius: 12px; border-left: 4px solid {border_color};">
        {summary_markdown}
    </div>
""", unsafe_allow_html=True)
