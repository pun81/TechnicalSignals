import streamlit as st
import requests
import pandas as pd
import numpy as np

# Page Configuration for Mobile View
st.set_page_config(
    page_title="Trade Decision Engine",
    page_icon="📈",
    layout="wide" # Switched to wide to accommodate the extra data cleanly
)

# Custom Dark Mode & High-Contrast CSS Styling
st.markdown("""
    <style>
    .stApp { background-color: #0b0b0e; color: #ffffff; }
    h1, h2, h3 { color: #00d2ff !important; text-align: uppercase; letter-spacing: 1.5px; }
    .metric-container { background: #22222d; border-radius: 12px; padding: 14px; border: 1px solid #333342; text-align: center; height: 100%; }
    .decision-wait { color: #ffe600; font-size: 3rem; font-weight: 900; text-align: center; text-shadow: 0 0 20px rgba(255,230,0,0.4); }
    .decision-exit { color: #ff2d55; font-size: 3rem; font-weight: 900; text-align: center; text-shadow: 0 0 20px rgba(255,45,85,0.5); }
    .decision-reenter { color: #30d158; font-size: 3rem; font-weight: 900; text-align: center; text-shadow: 0 0 20px rgba(48,209,88,0.5); }
    .alert-box { background: rgba(255, 45, 85, 0.25); border: 2px solid #ff2d55; color: #ff2d55; padding: 12px; border-radius: 10px; font-weight: bold; text-align: center; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2>Decision Engine</h2>", unsafe_allow_html=True)

# Load API Key automatically from Streamlit Secrets
api_key = st.secrets.get("TWELVE_DATA_API_KEY", "")

# Ticker Input
ticker_input = st.text_input("Ticker", value="SOXL").upper().strip()

# Asset Class Profile Definition
leveraged_assets = ['SOXL', 'TECL', 'TQQQ', 'UPRO', 'FAS']
is_leveraged = ticker_input in leveraged_assets

# Adaptive Thresholds Based on Asset Class
if is_leveraged:
    min_macd_hist = 0.05
    min_rsi_execution = 55.0
    min_adx = 20.0 # Chop filter for leveraged decay
    profile_label = "⚡ **Asset Profile:** 3x Leveraged ETF (Stricter Filters Active)"
else:
    min_macd_hist = 0.00
    min_rsi_execution = 50.0
    min_adx = 15.0 
    profile_label = "📊 **Asset Profile:** Standard Equity / Single Stock Swing Profile"

# Indicator Calculations
def compute_tradingview_style_indicators(df, window=14):
    if df is None or len(df) < window:
        return 50.0, 0.0, 0.0, 0.0
        
    close = df['Close']
    delta = close.diff()
    
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    ema20 = close.ewm(span=20, adjust=False).mean()
    
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    
    return rsi.iloc[-1], ema20.iloc[-1], macd.iloc[-1], histogram.iloc[-1]

def compute_adx(df, window=14):
    if df is None or len(df) < window * 2:
        return 0.0
    
    df = df.copy()
    high_diff = df['High'].diff()
    low_diff = -df['Low'].diff()
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    
    tr = np.maximum(df['High'] - df['Low'], 
         np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                    abs(df['Low'] - df['Close'].shift(1))))
    
    tr_smooth = pd.Series(tr).ewm(alpha=1/window, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/window, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/window, adjust=False).mean()
    
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.ewm(alpha=1/window, adjust=False).mean()
    
    return adx.iloc[-1]

def compute_50_sma(df):
    if df is None or len(df) < 50:
        return 0.0
    return df['Close'].rolling(window=50).mean().iloc[-1]

def find_structural_support(df_4h, is_leveraged):
    if df_4h is None or len(df_4h) < 10:
        return None
    recent_lows = df_4h['Low'].tail(30)
    raw_support = recent_lows.min()
    return round(raw_support * 0.985, 2) if is_leveraged else round(raw_support, 2)

@st.cache_data(ttl=60)
def fetch_twelve_data(symbol, interval, key):
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

# Load Market Data
market_data = None
df_4h_data = None
api_error_log = []

if not api_key:
    api_error_log.append("`TWELVE_DATA_API_KEY` is missing from your Streamlit Secrets.")
else:
    df_execution_raw, err_exec = fetch_twelve_data(ticker_input, "1h", api_key)
    df_1d, err_1d = fetch_twelve_data(ticker_input, "1day", api_key)
    df_4h_data, err_4h = fetch_twelve_data(ticker_input, "4h", api_key)

    if err_1d: api_error_log.append(f"1-Day Data Error: {err_1d}")
    if err_4h: api_error_log.append(f"4-Hour Data Error: {err_4h}")
    if err_exec: api_error_log.append(f"Execution Stream Data Error: {err_exec}")
    
    if df_1d is not None and df_4h_data is not None and df_execution_raw is not None and len(df_1d) > 0:
        live_price_val = float(df_execution_raw['Close'].iloc[-1])
        
        df_exec_inds = df_execution_raw.iloc[:-1].copy() if len(df_execution_raw) > 1 else df_execution_raw.copy()
        df_1d_inds = df_1d.iloc[:-1].copy() if len(df_1d) > 1 else df_1d.copy()
        df_4h_inds = df_4h_data.iloc[:-1].copy() if len(df_4h_data) > 1 else df_4h_data.copy()

        rsi_1d, ema_1d, _, _ = compute_tradingview_style_indicators(df_1d_inds)
        sma_50 = compute_50_sma(df_1d_inds)
        rsi_exec, _, _, _ = compute_tradingview_style_indicators(df_exec_inds)
        _, _, macd_4h, hist_4h = compute_tradingview_style_indicators(df_4h_inds)
        adx_4h = compute_adx(df_4h_inds)
        
        market_data = {
            "price": live_price_val, "rsi_1d": rsi_1d, "ema_1d": ema_1d, "sma_50": sma_50,
            "macd_4h": macd_4h, "hist_4h": hist_4h, "adx_4h": adx_4h, "rsi_1h": rsi_exec
        }

current_price = market_data["price"] if market_data else 115.76

price_input = st.number_input("Current Price ($)", value=round(current_price, 2), step=0.01)

calculated_support = find_structural_support(df_4h_data, is_leveraged)
default_stop = calculated_support if calculated_support else round(price_input * (0.90 if is_leveraged else 0.95), 2)
stop_level = st.number_input("Stop Level (Structural Support)", value=default_stop, step=0.01)

rsi_1h_val = market_data["rsi_1h"] if market_data else 42.5
rsi_1d_val = market_data["rsi_1d"] if market_data else 45.0
hist_4h_val = market_data["hist_4h"] if market_data else -0.29
ema_1d_val = market_data["ema_1d"] if market_data else price_input
sma_50_val = market_data["sma_50"] if market_data else price_input
adx_4h_val = market_data["adx_4h"] if market_data else 18.0

# Decision Logic
decision = "WAIT"
border_color = "#ffe600"
is_alert = False

if not api_key:
    summary_markdown = "⚠️ **Error:** `TWELVE_DATA_API_KEY` not found in Streamlit Secrets."
elif price_input < stop_level:
    decision = "EXIT"
    is_alert = True
    summary_markdown = f"""
**CRITICAL EXIT TRIGGER**
* **Trigger Event:** Current price (${price_input}) breached your 4H structural support floor (**${stop_level}**).
* **Action:** Leveraged volatility decay overrides oversold indicators. Execute exit immediately.
    """
    border_color = "#ff2d55"
else:
    decision = "WAIT"
    needed_price = round(ema_1d_val, 2)
    
    # Check if all conditions are met for a RE-ENTER
    if (price_input > needed_price) and (rsi_1d_val >= 50.0) and (hist_4h_val > min_macd_hist) and (adx_4h_val > min_adx) and (rsi_1h_val > min_rsi_execution):
        decision = "RE-ENTER"
        border_color = "#30d158"

    macro_status = "BULLISH TREND" if price_input > sma_50_val else "COUNTER-TREND BOUNCE"
    
    summary_markdown = f"""
{profile_label}<br>
*Macro Baseline context: Price is currently in a {macro_status} relative to the 50 SMA (${round(sma_50_val, 2)}).*<br><br>
**Exact Criteria Required for a RE-ENTER / BUY Signal:**
* **1-Day Swing Setup:** Price must close above **${needed_price}** (20 EMA) **AND** RSI >= 50.0 (currently `{round(rsi_1d_val, 1)}`).
* **4-Hour Momentum & Trend:** MACD Hist > **{min_macd_hist}** (currently `{round(hist_4h_val, 2)}`) **AND** ADX > **{min_adx}** (currently `{round(adx_4h_val, 1)}`).
* **1-Hour Execution Trigger:** Intraday RSI > **{min_rsi_execution}** (currently `{round(rsi_1h_val, 1)}`).
* **Capital Protection:** Active stop level anchored to 4H support at **${stop_level}**.
    """

if is_alert:
    st.markdown(f'<div class="alert-box">⚠ RISK OVERRIDE: Price breached support threshold (${stop_level}).</div>', unsafe_allow_html=True)

st.markdown("---")
if decision == "EXIT":
    st.markdown(f'<div class="decision-exit">{decision}</div>', unsafe_allow_html=True)
elif decision == "RE-ENTER":
    st.markdown(f'<div class="decision-reenter">{decision}</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="decision-wait">{decision}</div>', unsafe_allow_html=True)
st.markdown("---")

# Render Multi-Timeframe Breakdown
st.markdown("### Multi-Timeframe Technical Breakdown & Targets")
if market_data:
    col_a, col_b, col_c = st.columns(3)
    
    d1_bullish = market_data['price'] > ema_1d_val and rsi_1d_val >= 50.0
    d1_trend = "Bullish" if d1_bullish else "Bearish"
    d1_color = "#30d158" if d1_bullish else "#ff2d55"
    
    h4_bullish = hist_4h_val > min_macd_hist and adx_4h_val > min_adx
    h4_trend = "Bullish Conviction" if h4_bullish else ("Weak / Choppy" if adx_4h_val <= min_adx else "Bearish")
    h4_color = "#30d158" if h4_bullish else ("#ffe600" if adx_4h_val <= min_adx else "#ff2d55")
    
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
                <div style="font-size:0.7rem; color:#a0a0b0; letter-spacing:1px;">1-DAY (MACRO SWING)</div>
                <div style="font-size:1.0rem; font-weight:bold; color:{d1_color}; margin-top:6px;">{d1_trend}</div>
                <div style="font-size:0.75rem; color:#ffffff; margin-top:6px;">RSI: {round(rsi_1d_val, 1)} | 20 EMA: ${round(ema_1d_val, 1)}</div>
                <div style="font-size:0.75rem; color:#8e8e93; margin-top:2px;">50 SMA Baseline: ${round(sma_50_val, 1)}</div>
                <div style="font-size:0.65rem; color:#8e8e93; margin-top:6px; border-top:1px solid #333342; padding-top:4px;">Target: Price > 20 EMA & RSI >= 50</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col_b:
        st.markdown(f"""
            <div class="metric-container">
                <div style="font-size:0.7rem; color:#a0a0b0; letter-spacing:1px;">4-HR (MOMENTUM & TREND)</div>
                <div style="font-size:1.0rem; font-weight:bold; color:{h4_color}; margin-top:6px;">{h4_trend}</div>
                <div style="font-size:0.75rem; color:#ffffff; margin-top:6px;">MACD Hist: {round(hist_4h_val, 2)} | ADX: {round(adx_4h_val, 1)}</div>
                <div style="font-size:0.75rem; color:#8e8e93; margin-top:2px;">(ADX < 20 indicates high decay chop)</div>
                <div style="font-size:0.65rem; color:#8e8e93; margin-top:6px; border-top:1px solid #333342; padding-top:4px;">Target: Hist > {min_macd_hist} & ADX > {min_adx}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col_c:
        st.markdown(f"""
            <div class="metric-container">
                <div style="font-size:0.7rem; color:#a0a0b0; letter-spacing:1px;">1-HR EXECUTION STREAM</div>
                <div style="font-size:1.0rem; font-weight:bold; color:{h1_color}; margin-top:6px;">{h1_trend}</div>
                <div style="font-size:0.75rem; color:#ffffff; margin-top:6px;">RSI: {round(rsi_1h_val, 1)}</div>
                <div style="font-size:0.75rem; color:#8e8e93; margin-top:2px;">&nbsp;</div>
                <div style="font-size:0.65rem; color:#8e8e93; margin-top:6px; border-top:1px solid #333342; padding-top:4px;">Target: RSI > {min_rsi_execution}</div>
            </div>
        """, unsafe_allow_html=True)
else:
    st.error("Unable to fetch data streams. API Error Logs:")
    for err in api_error_log:
        st.code(err)

st.markdown(f"""
    <div style="margin-top: 20px; font-size: 0.85rem; color: #aeaeb2; line-height: 1.6; padding: 15px; background: #16161c; border-radius: 12px; border-left: 4px solid {border_color};">
        {summary_markdown}
    </div>
""", unsafe_allow_html=True)
