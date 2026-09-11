import streamlit as st
import requests
import pandas as pd
import numpy as np

# Page Configuration for Mobile View
st.set_page_config(
    page_title="Trade Decision Engine",
    page_icon="📈",
    layout="wide" 
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
    .divergence-box { background: rgba(255, 230, 0, 0.12); border: 1px solid #ffe600; color: #ffe600; padding: 10px; border-radius: 10px; font-size: 0.8rem; text-align: center; margin-bottom: 12px; }
    .check-row { display:flex; justify-content:space-between; padding:7px 4px; border-bottom:1px solid #22222d; font-size:0.85rem; }
    .check-pass { color:#30d158; }
    .check-fail { color:#8e8e93; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2>Decision Engine</h2>", unsafe_allow_html=True)

# --- SIDEBAR ENGINE TOGGLE ---
st.sidebar.markdown("### ⚙️ Engine Configuration")
engine_choice = st.sidebar.radio(
    "Indicator Math Engine",
    ["Fast (Vectorized Pandas)", "Exact (TradingView Loops)"],
    index=1,
    help="Toggle between high-speed vectorized calculation and 100% exact TradingView loop parity. Defaults to Exact so displayed values track TradingView's Pine-Script math as closely as possible."
)

st.sidebar.markdown("### 🕯️ 4H Signal Basis")
bar_basis_choice = st.sidebar.radio(
    "Which 4H bar drives the decision?",
    ["Confirmed (closed bar only)", "Live (current forming bar)"],
    index=0,
    help="Confirmed uses only fully-closed 4H candles — slower but avoids acting on a value that can still change before the candle closes. Live matches what TradingView's Technicals tab shows in real time, including the still-forming candle, but that number can repaint several times before the bar closes."
)

# Load API Key automatically from Streamlit Secrets
api_key = st.secrets.get("TWELVE_DATA_API_KEY", "")

# Ticker Input
ticker_input = st.text_input("Ticker", value="SOXL").upper().strip()

# Asset Class Profile Definition
# NOTE: this is a maintained list, not automatic detection — extend it if you trade
# other leveraged products. Bull and inverse (bear) leveraged funds are tracked
# separately because the sector/market regime filter needs to point the opposite
# direction for an inverse fund (e.g. SOXS wants the sector FALLING, not rising).
bull_leveraged_assets = ['SOXL', 'TECL', 'TQQQ', 'UPRO', 'FAS']
inverse_leveraged_assets = ['SOXS', 'TECS', 'SQQQ', 'SPXS', 'FAZ']
leveraged_assets = bull_leveraged_assets + inverse_leveraged_assets
is_leveraged = ticker_input in leveraged_assets
is_inverse = ticker_input in inverse_leveraged_assets

# Sector/Market regime proxy per leveraged ticker (underlying, unleveraged benchmark).
# Bull and inverse funds on the same sector map to the same proxy symbol — direction
# of the regime check (below) is what differs, not the benchmark itself.
leveraged_proxy_map = {
    'SOXL': 'SOXX', 'SOXS': 'SOXX',  # semiconductor sector
    'TECL': 'XLK', 'TECS': 'XLK',    # technology sector
    'TQQQ': 'QQQ', 'SQQQ': 'QQQ',    # nasdaq 100
    'UPRO': 'SPY', 'SPXS': 'SPY',    # S&P 500
    'FAS': 'XLF', 'FAZ': 'XLF',      # financials sector
}
regime_proxy_symbol = leveraged_proxy_map.get(ticker_input, 'SPY')

# Adaptive Thresholds Based on Asset Class
if is_leveraged:
    min_macd_hist = 0.05
    min_rsi_execution = 55.0
    min_adx = 20.0
    min_volume_zscore = 0.0   # require at/above this ticker's own recent average volume
    direction_label = "Inverse (Bear)" if is_inverse else "Bull"
    profile_label = f"⚡ **Asset Profile:** 3x Leveraged ETF — {direction_label} (Stricter Filters Active)"
else:
    min_macd_hist = 0.00
    min_rsi_execution = 50.0
    min_adx = 15.0
    min_volume_zscore = -0.5  # allow somewhat below-average volume for standard equities
    profile_label = "📊 **Asset Profile:** Standard Equity / Single Stock Swing Profile"


# --- ENGINE 1: FAST VECTORIZED METHODS ---
def compute_vectorized_indicators(df, window=14):
    if df is None or len(df) < 50:
        return 50.0, 0.0, 0.0, 0.0
    close = df['Close']
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100)

    ema20 = close.ewm(span=20, adjust=False).mean()
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal

    return rsi.iloc[-1], ema20.iloc[-1], macd.iloc[-1], histogram.iloc[-1]


# --- ENGINE 2: EXACT TRADINGVIEW LOOP METHODS ---
def calc_rma(series, length):
    s = series.astype(float).copy()
    res = pd.Series(index=s.index, dtype=float)
    valid_idx = s.dropna().index
    if len(valid_idx) < length:
        return res
    alpha = 1 / length
    sma = s.loc[valid_idx[:length]].mean()
    res.loc[valid_idx[length - 1]] = sma
    prev_val = sma
    for idx in valid_idx[length:]:
        val = s.loc[idx]
        prev_val = (alpha * val) + (1 - alpha) * prev_val
        res.loc[idx] = prev_val
    return res

def calc_exact_ema(s, span):
    res = pd.Series(index=s.index, dtype=float)
    valid = s.dropna()
    if len(valid) < span:
        return res
    alpha = 2 / (span + 1)
    sma = valid.iloc[:span].mean()
    res.loc[valid.index[span - 1]] = sma
    prev = sma
    for idx in valid.index[span:]:
        prev = (alpha * s.loc[idx]) + ((1 - alpha) * prev)
        res.loc[idx] = prev
    return res

def compute_loop_indicators(df, window=14):
    if df is None or len(df) < 50:
        return 50.0, 0.0, 0.0, 0.0
    close = df['Close']
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = calc_rma(gain, window)
    avg_loss = calc_rma(loss, window)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100)

    ema20 = calc_exact_ema(close, 20)
    exp1 = calc_exact_ema(close, 12)
    exp2 = calc_exact_ema(close, 26)
    macd_line = exp1 - exp2
    signal_line = calc_exact_ema(macd_line, 9)
    histogram = macd_line - signal_line

    return rsi.iloc[-1], ema20.iloc[-1], macd_line.iloc[-1], histogram.iloc[-1]


# --- UNIFIED WRAPPER ROUTER ---
def compute_tradingview_style_indicators(df, window=14):
    if engine_choice == "Exact (TradingView Loops)":
        return compute_loop_indicators(df, window)
    else:
        return compute_vectorized_indicators(df, window)


def _true_range(df):
    return np.maximum(
        df['High'] - df['Low'],
        np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1)))
    )

def compute_adx(df, window=14):
    if df is None or len(df) < window * 2:
        return 0.0
    df = df.copy()
    high_diff = df['High'].diff()
    low_diff = -df['Low'].diff()

    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)

    tr = _true_range(df)

    tr_rma = calc_rma(pd.Series(tr, index=df.index), window)
    plus_dm_rma = calc_rma(pd.Series(plus_dm, index=df.index), window)
    minus_dm_rma = calc_rma(pd.Series(minus_dm, index=df.index), window)

    tr_rma = tr_rma.replace(0, np.nan)
    plus_di = 100 * (plus_dm_rma / tr_rma)
    minus_di = 100 * (minus_dm_rma / tr_rma)

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (abs(plus_di - minus_di) / di_sum)
    dx = dx.fillna(0)

    adx = calc_rma(dx, window)
    return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0.0

def compute_atr_series(df, window=14):
    """Average True Range as a full series (Wilder RMA of True Range), so we can
    compare the latest ATR reading against its own recent baseline to detect
    an elevated-volatility regime."""
    if df is None or len(df) < window * 2:
        return None
    tr = _true_range(df)
    atr = calc_rma(pd.Series(tr, index=df.index), window)
    return atr

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
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=800&apikey={key}"
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
df_4h_confirmed = None
api_error_log = []

if not api_key:
    api_error_log.append("`TWELVE_DATA_API_KEY` is missing from your Streamlit Secrets.")
else:
    df_execution_raw, err_exec = fetch_twelve_data(ticker_input, "1h", api_key)
    df_1d, err_1d = fetch_twelve_data(ticker_input, "1day", api_key)
    df_4h_data, err_4h = fetch_twelve_data(ticker_input, "4h", api_key)
    df_proxy_1d, err_proxy = fetch_twelve_data(regime_proxy_symbol, "1day", api_key)

    if err_1d: api_error_log.append(f"1-Day Data Error: {err_1d}")
    if err_4h: api_error_log.append(f"4-Hour Data Error: {err_4h}")
    if err_exec: api_error_log.append(f"Execution Stream Data Error: {err_exec}")
    if err_proxy: api_error_log.append(f"Regime Proxy ({regime_proxy_symbol}) Data Error: {err_proxy}")

    if df_1d is not None and df_4h_data is not None and df_execution_raw is not None and len(df_1d) > 0:
        live_price_val = float(df_execution_raw['Close'].iloc[-1])

        df_exec_inds = df_execution_raw.iloc[:-1].copy() if len(df_execution_raw) > 1 else df_execution_raw.copy()
        df_1d_inds = df_1d.iloc[:-1].copy() if len(df_1d) > 1 else df_1d.copy()

        # CONFIRMED basis: drop the last (still-forming) 4H bar so the value can't
        # change again before the candle closes. This is what the decision engine acts on.
        df_4h_confirmed = df_4h_data.iloc[:-1].copy() if len(df_4h_data) > 1 else df_4h_data.copy()
        # LIVE basis: include the still-forming bar, same as TradingView's real-time
        # Technicals tab. Shown for comparison only, not used to gate the decision
        # unless the user explicitly switches "4H Signal Basis" to Live.
        df_4h_live = df_4h_data.copy()

        rsi_1d, ema_1d, _, _ = compute_tradingview_style_indicators(df_1d_inds)
        sma_50 = compute_50_sma(df_1d_inds)
        adx_1d = compute_adx(df_1d_inds)

        rsi_exec, _, _, _ = compute_tradingview_style_indicators(df_exec_inds)

        _, _, macd_4h_confirmed, hist_4h_confirmed = compute_tradingview_style_indicators(df_4h_confirmed)
        adx_4h_confirmed = compute_adx(df_4h_confirmed)

        _, _, macd_4h_live, hist_4h_live = compute_tradingview_style_indicators(df_4h_live)
        adx_4h_live = compute_adx(df_4h_live)

        # --- ATR: volatility level on the confirmed 4H series, and its own recent baseline ---
        atr_4h_series = compute_atr_series(df_4h_confirmed)
        if atr_4h_series is not None and not atr_4h_series.dropna().empty:
            atr_4h_val = atr_4h_series.dropna().iloc[-1]
            baseline_pool = atr_4h_series.dropna().tail(50)
            atr_4h_baseline = baseline_pool.mean() if len(baseline_pool) > 0 else atr_4h_val
        else:
            atr_4h_val, atr_4h_baseline = 0.0, 0.0
        vol_elevated = (atr_4h_baseline > 0) and (atr_4h_val > 1.3 * atr_4h_baseline)

        # --- Volume: z-score of the current confirmed 4H bar vs its own recent 20-bar
        # distribution. Self-normalizes per ticker (a "big" volume bar means something
        # different for NVDA than for a thin small-cap) instead of one flat multiplier
        # applied to every stock. Fails open (doesn't block) when there isn't enough
        # confirmed history yet — e.g. a recently-listed/spun-off ticker.
        vol_series = df_4h_confirmed['Volume']
        if len(vol_series) >= 21:
            vol_baseline = vol_series.iloc[-21:-1]
            vol_current = vol_series.iloc[-1]
            vol_mean, vol_std = vol_baseline.mean(), vol_baseline.std()
            if vol_std and vol_std > 0:
                rel_volume_zscore = (vol_current - vol_mean) / vol_std
                volume_sample_ok = True
            else:
                rel_volume_zscore, volume_sample_ok = 0.0, False
        else:
            rel_volume_zscore, volume_sample_ok = 0.0, False

        # --- Market/Sector regime: is the underlying benchmark moving the way this
        # specific position needs it to? Bull leveraged funds want the sector ABOVE its
        # 20 EMA; inverse/bear leveraged funds want it BELOW (SOXS wants semis falling).
        if df_proxy_1d is not None and len(df_proxy_1d) > 0:
            df_proxy_inds = df_proxy_1d.iloc[:-1].copy() if len(df_proxy_1d) > 1 else df_proxy_1d.copy()
            if len(df_proxy_inds) >= 50:
                _, regime_ema, _, _ = compute_tradingview_style_indicators(df_proxy_inds)
            else:
                regime_ema = 0.0
            regime_price = float(df_proxy_1d['Close'].iloc[-1])
        else:
            regime_ema, regime_price = 0.0, 0.0

        market_data = {
            "price": live_price_val, "rsi_1d": rsi_1d, "ema_1d": ema_1d, "sma_50": sma_50,
            "adx_1d": adx_1d, "rsi_1h": rsi_exec,
            "macd_4h_confirmed": macd_4h_confirmed, "hist_4h_confirmed": hist_4h_confirmed, "adx_4h_confirmed": adx_4h_confirmed,
            "macd_4h_live": macd_4h_live, "hist_4h_live": hist_4h_live, "adx_4h_live": adx_4h_live,
            "atr_4h": atr_4h_val, "atr_4h_baseline": atr_4h_baseline, "vol_elevated": vol_elevated,
            "rel_volume_zscore": rel_volume_zscore, "volume_sample_ok": volume_sample_ok,
            "regime_symbol": regime_proxy_symbol, "regime_price": regime_price, "regime_ema20": regime_ema,
        }

current_price = market_data["price"] if market_data else 115.76
price_input = st.number_input("Current Price ($)", value=round(current_price, 2), step=0.01)

atr_4h_val_f = market_data["atr_4h"] if market_data else 0.0
vol_elevated_f = market_data["vol_elevated"] if market_data else False

calculated_support = find_structural_support(df_4h_data, is_leveraged)
raw_default_stop = calculated_support if calculated_support else round(price_input * (0.90 if is_leveraged else 0.95), 2)
atr_cushion = round(0.5 * atr_4h_val_f, 2) if vol_elevated_f else 0.0
default_stop = round(raw_default_stop - atr_cushion, 2)
stop_level = st.number_input("Stop Level (Structural Support)", value=default_stop, step=0.01)
if vol_elevated_f:
    st.caption(f"🌪️ 4H volatility is elevated (ATR {round(atr_4h_val_f,2)} vs its own ~50-bar baseline). A cushion of ${atr_cushion} was subtracted from the raw structural low so a normal-sized wick doesn't trigger a premature exit. Override the number above any time.")
else:
    st.caption("4H volatility is in a normal range — stop is the raw structural support level, no ATR cushion applied.")

rsi_1h_val = market_data["rsi_1h"] if market_data else 42.5
rsi_1d_val = market_data["rsi_1d"] if market_data else 45.0
adx_1d_val = market_data["adx_1d"] if market_data else 22.0
ema_1d_val = market_data["ema_1d"] if market_data else price_input
sma_50_val = market_data["sma_50"] if market_data else price_input
needed_price = round(ema_1d_val, 2)

hist_4h_confirmed_val = market_data["hist_4h_confirmed"] if market_data else -0.29
macd_4h_confirmed_val = market_data["macd_4h_confirmed"] if market_data else -0.29
adx_4h_confirmed_val = market_data["adx_4h_confirmed"] if market_data else 18.0

hist_4h_live_val = market_data["hist_4h_live"] if market_data else hist_4h_confirmed_val
macd_4h_live_val = market_data["macd_4h_live"] if market_data else macd_4h_confirmed_val
adx_4h_live_val = market_data["adx_4h_live"] if market_data else adx_4h_confirmed_val

rel_volume_zscore_val = market_data["rel_volume_zscore"] if market_data else 0.0
volume_sample_ok_val = market_data["volume_sample_ok"] if market_data else False
regime_symbol_val = market_data["regime_symbol"] if market_data else regime_proxy_symbol
regime_price_val = market_data["regime_price"] if market_data else 0.0
regime_ema_val = market_data["regime_ema20"] if market_data else 0.0

# Fail-open on both supplementary filters: neither is a safety trigger like the
# stop-loss, so missing/insufficient data shouldn't silently block the whole engine.
if not volume_sample_ok_val:
    volume_ok = True   # not enough confirmed 4H history yet (e.g. a recent listing) — skip, don't block
else:
    volume_ok = rel_volume_zscore_val >= min_volume_zscore

if not (market_data and regime_ema_val):
    regime_ok = True
elif is_inverse:
    regime_ok = regime_price_val < regime_ema_val   # inverse fund wants the sector FALLING
else:
    regime_ok = regime_price_val > regime_ema_val    # bull fund wants the sector RISING

# Which 4H reading actually drives the decision, per the sidebar toggle
if bar_basis_choice == "Live (current forming bar)":
    hist_4h_val, adx_4h_val = hist_4h_live_val, adx_4h_live_val
    basis_label = "LIVE (forming candle — matches TradingView real-time)"
else:
    hist_4h_val, adx_4h_val = hist_4h_confirmed_val, adx_4h_confirmed_val
    basis_label = "CONFIRMED (last closed 4H candle only)"

# Flag when live and confirmed disagree on bullish/bearish sign — this is the
# scenario that produces "the app says bullish, TradingView says bearish" confusion.
signals_diverge = (hist_4h_confirmed_val > min_macd_hist) != (hist_4h_live_val > min_macd_hist)

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

    reenter_conditions = (
        (price_input > needed_price) and (rsi_1d_val >= 50.0)
        and (hist_4h_val > min_macd_hist) and (adx_4h_val > min_adx)
        and volume_ok
        and (rsi_1h_val > min_rsi_execution)
        and regime_ok
    )
    if reenter_conditions:
        decision = "RE-ENTER"
        border_color = "#30d158"

    macro_status = "BULLISH TREND" if price_input > sma_50_val else "COUNTER-TREND BOUNCE"

    summary_markdown = f"""
{profile_label}<br>
*Macro Baseline context (informational only — not a gate; entries intentionally use the faster 20 EMA so leveraged upside isn't missed waiting on the 50 SMA): price is in a {macro_status} relative to the 50 SMA (${round(sma_50_val, 2)}).*<br><br>
**Exact Criteria Required for a RE-ENTER / BUY Signal:**
* **1-Day Swing Setup:** Price must close above **${needed_price}** (20 EMA) **AND** RSI >= 50.0 (currently `{round(rsi_1d_val, 1)}`).
* **4-Hour Momentum & Trend ({basis_label}):** MACD Hist > **{min_macd_hist}** (currently `{round(hist_4h_val, 2)}`) **AND** ADX > **{min_adx}** (currently `{round(adx_4h_val, 1)}`).
* **4-Hour Volume:** z-score vs its own 20-bar pattern >= **{min_volume_zscore}** (currently `{round(rel_volume_zscore_val, 2)}`{' — insufficient history, check skipped' if not volume_sample_ok_val else ''}).
* **1-Hour Execution Trigger:** Intraday RSI > **{min_rsi_execution}** (currently `{round(rsi_1h_val, 1)}`).
* **Sector/Market Regime:** {regime_symbol_val} must be {'below' if is_inverse else 'above'} its own 20 EMA (currently {'✅ aligned' if regime_ok else '❌ against you'}).
* **Capital Protection:** Active stop level anchored to 4H support at **${stop_level}**{' (ATR-widened)' if vol_elevated_f else ''}.
    """

if is_alert:
    st.markdown(f'<div class="alert-box">⚠ RISK OVERRIDE: Price breached support threshold (${stop_level}).</div>', unsafe_allow_html=True)

if signals_diverge and not is_alert:
    st.markdown(
        f'<div class="divergence-box">⚠ The live (forming) 4H MACD Hist ({round(hist_4h_live_val, 2)}) and the last confirmed/closed 4H MACD Hist ({round(hist_4h_confirmed_val, 2)}) disagree on direction. '
        f'This is normal mid-candle and is why the app and TradingView\'s real-time technicals can flash different colors until the current 4H candle closes. '
        f'The decision below uses the <b>{basis_label}</b> reading.</div>',
        unsafe_allow_html=True
    )

st.markdown("---")
if decision == "EXIT":
    st.markdown(f'<div class="decision-exit">{decision}</div>', unsafe_allow_html=True)
elif decision == "RE-ENTER":
    st.markdown(f'<div class="decision-reenter">{decision}</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="decision-wait">{decision}</div>', unsafe_allow_html=True)
st.markdown("---")

# --- SIMPLE, PLAIN-LANGUAGE CHECKLIST (always visible, no raw numbers to parse) ---
if market_data and not is_alert:
    st.markdown("##### Entry Checklist")
    regime_direction_note = "falling" if is_inverse else "rising"
    volume_state = "skip" if not volume_sample_ok_val else ("pass" if volume_ok else "fail")
    checklist_items = [
        ("1-Day trend is up (price above 20 EMA & RSI ≥ 50)", "pass" if (price_input > needed_price) and (rsi_1d_val >= 50.0) else "fail"),
        ("4-Hour momentum improving", "pass" if hist_4h_val > min_macd_hist else "fail"),
        ("4-Hour trend has real strength (not choppy)", "pass" if adx_4h_val > min_adx else "fail"),
        (f"4-Hour volume backs up the move{' (not enough history — skipped)' if not volume_sample_ok_val else ''}", volume_state),
        ("1-Hour momentum confirms the entry timing", "pass" if rsi_1h_val > min_rsi_execution else "fail"),
        (f"{regime_symbol_val} (sector/market) is {regime_direction_note}, as this position needs", "pass" if regime_ok else "fail"),
    ]
    rows_html = ""
    icon_map = {"pass": ("✅", "check-pass"), "fail": ("❌", "check-fail"), "skip": ("➖", "check-fail")}
    for label, state in checklist_items:
        icon, css_class = icon_map[state]
        rows_html += f'<div class="check-row"><span>{label}</span><span class="{css_class}">{icon}</span></div>'
    st.markdown(f'<div class="metric-container" style="text-align:left;">{rows_html}</div>', unsafe_allow_html=True)
    st.markdown("")

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
                <div style="font-size:0.75rem; color:#ffffff; margin-top:6px;">RSI: {round(rsi_1d_val, 1)} | ADX: {round(adx_1d_val, 1)}</div>
                <div style="font-size:0.75rem; color:#8e8e93; margin-top:2px;">20 EMA: ${round(ema_1d_val, 1)} | 50 SMA: ${round(sma_50_val, 1)} (context only)</div>
                <div style="font-size:0.65rem; color:#8e8e93; margin-top:6px; border-top:1px solid #333342; padding-top:4px;">Target: Price > 20 EMA & RSI >= 50</div>
            </div>
        """, unsafe_allow_html=True)

    with col_b:
        confirmed_active = (bar_basis_choice != "Live (current forming bar)")
        live_active = not confirmed_active
        confirmed_style = "color:#30d158; font-weight:800;" if confirmed_active else "color:#5a5a66; font-weight:400;"
        live_style = "color:#30d158; font-weight:800;" if live_active else "color:#5a5a66; font-weight:400;"
        confirmed_tag = " ◀ ACTIVE" if confirmed_active else ""
        live_tag = " ◀ ACTIVE" if live_active else ""
        confirmed_bg = "background:rgba(48,209,88,0.10); border-radius:6px;" if confirmed_active else ""
        live_bg = "background:rgba(48,209,88,0.10); border-radius:6px;" if live_active else ""

        st.markdown(f"""
            <div class="metric-container">
                <div style="font-size:0.7rem; color:#a0a0b0; letter-spacing:1px;">4-HR (MOMENTUM & TREND)</div>
                <div style="font-size:1.0rem; font-weight:bold; color:{h4_color}; margin-top:6px;">{h4_trend}</div>
                <div style="font-size:0.75rem; color:#ffffff; margin-top:6px;">MACD Hist: {round(hist_4h_val, 2)} | ADX: {round(adx_4h_val, 1)} | Vol z: {round(rel_volume_zscore_val, 2) if volume_sample_ok_val else 'n/a'}</div>
                <div style="font-size:0.65rem; color:#8e8e93; margin-top:6px; border-top:1px solid #333342; padding-top:4px;">Decision uses: {basis_label}</div>
            </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.markdown(f"""
            <div class="metric-container">
                <div style="font-size:0.7rem; color:#a0a0b0; letter-spacing:1px;">1-HR EXECUTION STREAM</div>
                <div style="font-size:1.0rem; font-weight:bold; color:{h1_color}; margin-top:6px;">{h1_trend}</div>
                <div style="font-size:0.75rem; color:#ffffff; margin-top:6px;">RSI: {round(rsi_1h_val, 1)}</div>
                <div style="font-size:0.75rem; color:#8e8e93; margin-top:2px;">Regime ({regime_symbol_val}): {'OK ✅' if regime_ok else 'Against you ❌'}</div>
                <div style="font-size:0.65rem; color:#8e8e93; margin-top:6px; border-top:1px solid #333342; padding-top:4px;">Target: RSI > {min_rsi_execution}</div>
            </div>
        """, unsafe_allow_html=True)

    with st.expander("🔍 Show detailed indicator values (Confirmed vs Live, ATR, regime raw numbers)"):
        st.markdown(f"""
- **4H Confirmed (closed bar):** Hist `{round(hist_4h_confirmed_val, 2)}` / MACD `{round(macd_4h_confirmed_val, 2)}` / ADX `{round(adx_4h_confirmed_val, 1)}`
- **4H Live (forming bar):** Hist `{round(hist_4h_live_val, 2)}` / MACD `{round(macd_4h_live_val, 2)}` / ADX `{round(adx_4h_live_val, 1)}`
- **4H ATR:** `{round(atr_4h_val_f, 2)}` vs its own ~50-bar baseline `{round(market_data['atr_4h_baseline'], 2) if market_data else 0.0}` &rarr; volatility {'ELEVATED' if vol_elevated_f else 'normal'}
- **4H Volume z-score:** `{round(rel_volume_zscore_val, 2) if volume_sample_ok_val else 'n/a — insufficient history'}` vs its own 20-bar mean/std
- **Regime Proxy ({regime_symbol_val}):** price `{round(regime_price_val, 2)}` vs 20 EMA `{round(regime_ema_val, 2)}` — needs {'BELOW (inverse fund)' if is_inverse else 'ABOVE (bull fund)'}
        """)
else:
    st.error("Unable to fetch data streams. API Error Logs:")
    for err in api_error_log:
        st.code(err)

st.markdown(f"""
    <div style="margin-top: 20px; font-size: 0.85rem; color: #aeaeb2; line-height: 1.6; padding: 15px; background: #16161c; border-radius: 12px; border-left: 4px solid {border_color};">
        {summary_markdown}
    </div>
""", unsafe_allow_html=True)

# --- PLAIN-LANGUAGE INDICATOR CHEAT SHEET ---
st.markdown("""
    <div style="margin-top: 14px; font-size: 0.78rem; color: #aeaeb2; line-height: 1.7; padding: 14px 15px; background: #16161c; border-radius: 12px; border-left: 4px solid #333342;">
        <div style="font-size:0.7rem; color:#a0a0b0; letter-spacing:1px; margin-bottom:6px;">QUICK INDICATOR CHEAT SHEET</div>
        <b style="color:#ffffff;">RSI</b> — is price up or down more than usual lately? High = recent gains dominating (strong/overbought). Low = recent losses dominating (weak/oversold).<br>
        <b style="color:#ffffff;">MACD line</b> — is the short-term average price above or below the long-term average? Positive = bullish trend. Negative = bearish trend.<br>
        <b style="color:#ffffff;">Hist</b> (MACD Histogram) — is momentum speeding up or slowing down versus its own recent pace? Positive = accelerating/improving. Negative = decelerating/fading. <i>Not</i> the same as trend direction — a negative MACD line with a positive Hist means "still bearish overall, but getting less bearish."<br>
        <b style="color:#ffffff;">ADX</b> — how strong or powerful is the current trend, regardless of direction? High = real conviction move. Low = weak/choppy, no clear trend to trust yet. Doesn't use volume.<br>
        <b style="color:#ffffff;">Volume (relative)</b> — how much participation is behind the current move vs. normal? Confirms whether a price move is backed by real flow or just drift.<br>
        <b style="color:#ffffff;">ATR</b> — how big are typical price swings right now? Used here to widen your stop slightly when volatility is elevated, so normal noise doesn't fake you out.
    </div>
""", unsafe_allow_html=True)
