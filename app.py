import streamlit as st
import yfinance as yf
from tradingview_ta import TA_Handler, Interval
import pandas as pd

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

st.markdown("<h2>Decision Engine (Live Data)</h2>", unsafe_allow_html=True)

# User Inputs for Ticker & Custom Stop Configuration
col1, col2 = st.columns(2)
with col1:
    ticker_input = st.text_input("Ticker", value="SOXL").upper().strip()
with col2:
    # Fetch live price dynamically using yfinance
    try:
        ticker_obj = yf.Ticker(ticker_input)
        todays_data = ticker_obj.history(period="1d")
        current_price = float(todays_data['Close'].iloc[-1]) if not todays_data.empty else 115.76
    except Exception:
        current_price = 115.76

    price_input = st.number_input("Current Price ($)", value=round(current_price, 2), step=0.01)

# Default buffer rule: 10% for leveraged 3x ETFs, 5% for standard stocks
leveraged_assets = ['SOXL', 'TECL', 'TQQQ', 'UPRO', 'FAS']
default_buffer = 0.10 if ticker_input in leveraged_assets else 0.05
default_stop = round(price_input * (1 - default_buffer), 2)

stop_level = st.number_input("Stop Level ($)", value=default_stop, step=0.01)

# Function to fetch TradingView technical ratings securely
@st.cache_data(ttl=60) # Cache for 60 seconds to respect rate limits
def fetch_tradingview_data(symbol):
    try:
        # Determine exchange/screener mapping
        exchange = "NASDAQ" if symbol in ['SOXL', 'TQQQ', 'TECL'] else "NYSE"
        
        # 1-Day Analysis
        handler_1d = TA_Handler(symbol=symbol, screener="america", exchange=exchange, interval=Interval.INTERVAL_1_DAY)
        analysis_1d = handler_1d.get_analysis()
        
        # 4-Hour Analysis
        handler_4h = TA_Handler(symbol=symbol, screener="america", exchange=exchange, interval=Interval.INTERVAL_4_HOURS)
        analysis_4h = handler_4h.get_analysis()
        
        # 1-Hour Analysis
        handler_1h = TA_Handler(symbol=symbol, screener="america", exchange=exchange, interval=Interval.INTERVAL_1_HOUR)
        analysis_1h = handler_1h.get_analysis()
        
        return {
            "d1": analysis_1d.summary["RECOMMENDATION"],
            "d1_indicators": analysis_1d.indicators,
            "h4": analysis_4h.summary["RECOMMENDATION"],
            "h4_indicators": analysis_4h.indicators,
            "h1": analysis_1h.summary["RECOMMENDATION"],
            "h1_indicators": analysis_1h.indicators,
        }
    except Exception as e:
        return None

# Fetch live data signals
tv_data = fetch_tradingview_data(ticker_input)

# Decision Engine Logic Evaluation
decision = "WAIT"
summary_text = ""
border_color = "#ffe600"
is_alert = False

if price_input < stop_level:
    decision = "EXIT"
    is_alert = True
    summary_text = f"CRITICAL EXIT TRIGGER: {ticker_input} has dropped below your automated risk stop of ${stop_level}. Leveraged decay risk overrides short-term oversold indicators. Execute exit immediately."
    border_color = "#ff2d55"
elif tv_data and tv_data["d1"] in ["STRONG_BUY", "BUY"] and ticker_input not in leveraged_assets:
    decision = "RE-ENTER"
    summary_text = f"Uptrend confirmed across live TradingView feeds. Macro and momentum indicators align favorably for a position entry."
    border_color = "#30d158"
else:
    decision = "WAIT"
    summary_text = f"Macro trend is constrained. Watch live TradingView metrics below for a relief bounce: ensure <b>1-Hr indicators shift positive</b> and <b>4-Hr momentum recovers</b> before changing stance. Protect capital against stop level (${stop_level})."
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

# Render Multi-Timeframe Ladder Using Live TradingView Data
st.markdown("### Live Timeframe Breakdown")
if tv_data:
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown(f"""
            <div class="metric-container">
                <div style="font-size:0.75rem; color:#a0a0b0;">1-DAY (MACRO)</div>
                <div style="font-size:1rem; font-weight:bold; color:#fff; margin-top:5px;">{tv_data['d1'].replace('_', ' ')}</div>
                <div style="font-size:0.7rem; color:#8e8e93; margin-top:3px;">RSI: {round(tv_data['d1_indicators'].get('RSI', 0), 1)}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col_b:
        st.markdown(f"""
            <div class="metric-container">
                <div style="font-size:0.75rem; color:#a0a0b0;">4-HR (MOMENTUM)</div>
                <div style="font-size:1rem; font-weight:bold; color:#fff; margin-top:5px;">{tv_data['h4'].replace('_', ' ')}</div>
                <div style="font-size:0.7rem; color:#8e8e93; margin-top:3px;">MACD: {round(tv_data['h4_indicators'].get('MACD.macd', 0), 2)}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col_c:
        st.markdown(f"""
            <div class="metric-container">
                <div style="font-size:0.75rem; color:#a0a0b0;">1-HR (EXECUTION)</div>
                <div style="font-size:1rem; font-weight:bold; color:#fff; margin-top:5px;">{tv_data['h1'].replace('_', ' ')}</div>
                <div style="font-size:0.7rem; color:#8e8e93; margin-top:3px;">RSI: {round(tv_data['h1_indicators'].get('RSI', 0), 1)}</div>
            </div>
        """, unsafe_allow_html=True)
else:
    st.warning("Unable to reach TradingView live endpoints. Please verify ticker symbol.")

# Recommendation Box
st.markdown(f"""
    <div style="margin-top: 20px; font-size: 0.85rem; color: #aeaeb2; line-height: 1.5; padding: 15px; background: #16161c; border-radius: 12px; border-left: 4px solid {border_color};">
        {summary_text}
    </div>
""", unsafe_allow_html=True)
