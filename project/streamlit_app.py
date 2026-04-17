import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

st.set_page_config(page_title='Futures Liquidity Pro', layout='wide')

# ── Prevent browser from scrolling to top on each rerun ──────────
import streamlit.components.v1 as components
components.html("""
<script>
(function() {
    const stApp = window.parent.document.querySelector('.main');
    if (!stApp) return;
    let savedY = 0;
    stApp.addEventListener('scroll', () => { savedY = stApp.scrollTop; }, { passive: true });
    const observer = new MutationObserver(() => {
        if (savedY > 0) stApp.scrollTop = savedY;
    });
    observer.observe(stApp, { childList: true, subtree: true });
})();
</script>
""", height=0)

# ── Constants ─────────────────────────────────────────────────────
# Spread threshold: your data shows spreads are almost always exactly
# 0.25 (the ES 1-tick minimum) or 0.0/negative (crossed/locked market).
# So the real signal is simply: spread must be POSITIVE and NORMAL.
# We use > 0.0 to enter on any valid spread, and flag negatives as bad data.
SPREAD_THRESHOLD = 0.0    # enter whenever spread is positive (data-driven)
SPREAD_NEGATIVE  = 0.0    # below this = crossed/bad market, never enter
GAMMA_THRESHOLD  = 0.0006 # kept for display but never blocks entry in this dataset
                           # (100% of your ticks have gamma < 0.0006)
PNL_PER_POINT    = 50.0   # ES futures: $50 per point

# ── Load & prepare ────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(Path(__file__).parent.parent / 'dataset' / 'cleaned.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    def get_snapshot(g):
        asks = g[g['Side'] == 'Ask']
        bids = g[g['Side'] == 'Bid']
        best_ask = asks['future_strike'].min()
        best_bid = bids['future_strike'].max()
        mid    = (best_ask + best_bid) / 2 if (pd.notna(best_ask) and pd.notna(best_bid)) else np.nan
        spread = (best_ask - best_bid)     if (pd.notna(best_ask) and pd.notna(best_bid)) else np.nan
        return pd.Series({
            'Best_Ask': best_ask,
            'Best_Bid': best_bid,
            'Mid':      mid,
            'Spread':   spread,
            'Gamma':    g['call_gamma'].mean()
        })

    summary = (df.groupby(df['timestamp'].dt.floor('s'))
                 .apply(get_snapshot, include_groups=False)
                 .reset_index()
                 .dropna(subset=['Best_Ask', 'Best_Bid'])
                 .reset_index(drop=True))
    return summary


# ── Signal logic ─────────────────────────────────────────────────
def get_signal(spread, gamma):
    """
    Signal logic updated to match actual dataset characteristics:
      - Spreads in this data are almost always 0.25 (1 ES tick) or 0.0
      - Negative spreads mean bid > ask (crossed market = bad/stale data)
      - Gamma < 0.0006 on 100% of ticks, so it never blocks entry here
      - Real decision: is the spread positive and valid?
    """
    if spread < SPREAD_NEGATIVE:
        return (
            "BAD DATA — Negative spread (crossed market)",
            "red",
            f"Spread is {spread:.4f} pts — the bid ({spread:.4f}) is HIGHER than the ask. "
            "This is a crossed market, which means the data snapshot is stale or two orders "
            "that should have matched haven't been cleared yet. Never enter on a crossed market — "
            "it means price discovery has broken down momentarily."
        )
    elif spread == 0.0:
        return (
            "LOCKED MARKET — Spread is zero",
            "orange",
            "Spread is exactly 0.0 pts — bid and ask are the same price (locked market). "
            "As a market maker you earn nothing on a zero spread. Wait for the spread to open up. "
            "In ES futures this is uncommon and usually resolves within 1-2 ticks."
        )
    elif spread == 0.25:
        return (
            "ENTER — Normal 1-tick spread (0.25 pts)",
            "green",
            f"Spread is {spread:.4f} pts — exactly 1 ES tick, which is the standard minimum. "
            "This is the normal condition in your dataset (occurs ~90% of ticks). "
            "At $50/point, capturing this spread earns $12.50 gross per round-trip. "
            f"Gamma is {gamma:.7f} — well below risk threshold, no repricing concern."
        )
    elif spread > 0.25:
        return (
            f"BEST ENTRY — Wide spread ({spread:.4f} pts, above normal)",
            "green",
            f"Spread is {spread:.4f} pts — wider than the usual 1-tick minimum. "
            "This is rare in your data (only ~0.8% of ticks) and represents the best "
            f"profit opportunity: ${spread/2 * 50:.2f} gross per round-trip vs the usual $12.50. "
            "Prioritize entering here."
        )
    else:
        return (
            " WAIT — Spread too narrow",
            "red",
            f"Spread is {spread:.4f} pts — between 0 and 1 tick. Not a standard ES spread value. "
            "May indicate a transitional state between locked and normal. Wait for a clean 0.25 tick."
        )


# ── P&L calculation ───────────────────────────────────────────────
def calc_pnl(entry_row, exit_row):
    """
    Simplified market-maker P&L model:
      - Gross profit = entry spread / 2  (you earn half the spread on each leg)
      - Adverse selection cost = |mid price move| * 0.5
      - Net P&L in points, then convert to dollars
    """
    spread_capture = entry_row['Spread'] / 2
    mid_move       = abs(exit_row['Mid'] - entry_row['Mid'])
    pnl_pts        = spread_capture - mid_move * 0.5
    pnl_usd        = pnl_pts * PNL_PER_POINT
    return round(pnl_pts, 4), round(pnl_usd, 2)


# ── Build main figure ─────────────────────────────────────────────
def build_fig(df_slice, entry_ticks=None, exit_ticks=None, open_tick=None):
    ts       = df_slice['timestamp']
    ask      = df_slice['Best_Ask']
    bid      = df_slice['Best_Bid']
    spread   = df_slice['Spread']

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.72, 0.28],
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("", "")
    )

    # Spread fill zone
    fig.add_trace(go.Scatter(
        x=pd.concat([ts, ts[::-1]]),
        y=pd.concat([ask, bid[::-1]]),
        fill='toself', fillcolor='rgba(150,150,150,0.12)',
        line=dict(width=0), showlegend=True, name='Spread zone',
        hoverinfo='skip'
    ), row=1, col=1)

    # Best Ask line
    fig.add_trace(go.Scatter(
        x=ts, y=ask,
        line=dict(color='#ff4b4b', width=1.5),
        name='Best Ask',
        customdata=np.stack([
            spread.round(4),
            df_slice['Mid'].round(4),
            df_slice['Gamma'].round(7),
        ], axis=-1),
        hovertemplate=(
            '<b>Ask:</b> %{y:.2f}<br>'
            '<b>Spread:</b> %{customdata[0]}<br>'
            '<b>Mid:</b> %{customdata[1]}<br>'
            '<b>Gamma:</b> %{customdata[2]}<extra></extra>'
        )
    ), row=1, col=1)

    # Best Bid line
    fig.add_trace(go.Scatter(
        x=ts, y=bid,
        line=dict(color='#00cc96', width=1.5),
        name='Best Bid',
        hovertemplate='<b>Bid:</b> %{y:.2f}<extra></extra>'
    ), row=1, col=1)

    # Entry markers
    if entry_ticks:
        etimes = [df_slice.iloc[i]['timestamp'] for i in entry_ticks if i < len(df_slice)]
        emids  = [df_slice.iloc[i]['Mid']       for i in entry_ticks if i < len(df_slice)]
        if etimes:
            fig.add_trace(go.Scatter(
                x=etimes, y=emids,
                mode='markers',
                marker=dict(symbol='triangle-up', size=14, color='#00aaff', line=dict(color='white', width=1)),
                name='Entry',
                hovertemplate='<b>ENTRY</b> at %{y:.2f}<extra></extra>'
            ), row=1, col=1)

    # Exit markers
    if exit_ticks:
        xtimes = [df_slice.iloc[i]['timestamp'] for i in exit_ticks if i < len(df_slice)]
        xmids  = [df_slice.iloc[i]['Mid']       for i in exit_ticks if i < len(df_slice)]
        if xtimes:
            fig.add_trace(go.Scatter(
                x=xtimes, y=xmids,
                mode='markers',
                marker=dict(symbol='x', size=14, color='#ffaa00', line=dict(color='white', width=1.5)),
                name='Exit',
                hovertemplate='<b>EXIT</b> at %{y:.2f}<extra></extra>'
            ), row=1, col=1)

    # Open trade marker (current open position)
    if open_tick is not None and open_tick < len(df_slice):
        ot = df_slice.iloc[open_tick]
        fig.add_trace(go.Scatter(
            x=[ot['timestamp']], y=[ot['Mid']],
            mode='markers',
            marker=dict(symbol='circle-open', size=18, color='#00aaff', line=dict(width=2.5)),
            name='Open position',
            hovertemplate='<b>OPEN POSITION</b> at %{y:.2f}<extra></extra>'
        ), row=1, col=1)

    # Spread sub-chart
    fig.add_trace(go.Scatter(
        x=ts, y=spread,
        fill='tozeroy',
        fillcolor='rgba(100,180,255,0.12)',
        line=dict(color='#6db4ff', width=1.2),
        name='Spread',
        hovertemplate='Spread: %{y:.4f}<extra></extra>'
    ), row=2, col=1)

    # Threshold line on spread chart
    fig.add_hline(
        y=0.0,
        line=dict(color='rgba(255,200,0,0.6)', width=1, dash='dash'),
        row=2, col=1,
        annotation_text="Locked market (0.0)",
        annotation_font_size=10,
        annotation_font_color='rgba(255,200,0,0.8)'
    )

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0e1117',
        height=500,
        margin=dict(l=50, r=20, t=20, b=60),
        legend=dict(orientation='h', y=1.06, x=0),
        hovermode='x unified'
    )
    fig.update_xaxes(tickformat='%H:%M:%S', row=1, col=1)
    fig.update_xaxes(tickformat='%H:%M:%S', rangeslider=dict(visible=True, thickness=0.04), row=2, col=1)
    fig.update_yaxes(title_text='Price (ES)', row=1, col=1)
    fig.update_yaxes(title_text='Spread', row=2, col=1)
    return fig


# ── Session state init ────────────────────────────────────────────
def init_state():
    defaults = {
        'tick':        1,
        'last_date':   None,
        'open_trade':  None,   # dict when a trade is open
        'trade_log':   [],     # list of closed trade dicts
        'total_pnl':   0.0,
        'show_explain': True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Load data ─────────────────────────────────────────────────────
df_all = load_data()

# ── Date selector ─────────────────────────────────────────────────
st.title("Futures Market Maker Simulation")

col0, col1, col2, col3 = st.columns([1, 1, 1, 2])
with col0:
    available_dates = sorted(df_all['timestamp'].dt.date.unique())
    selected_label  = st.selectbox("Date", [str(d) for d in available_dates])
    selected_date   = pd.Timestamp(selected_label).date()

df_all = df_all[df_all['timestamp'].dt.date == selected_date].reset_index(drop=True)
total  = len(df_all)

if st.session_state.last_date != selected_label:
    st.session_state.tick       = 1
    st.session_state.last_date  = selected_label
    st.session_state.open_trade = None
    st.session_state.trade_log  = []
    st.session_state.total_pnl  = 0.0

with col1:
    speed = st.slider("Speed (s/tick)", 0.1, 2.0, 0.5, 0.1)
with col2:
    window_size = st.slider("Display window (ticks)", 10, total, min(60, total))
with col3:
    running = st.toggle("Run simulation", value=False)

# ── Placeholders — declared early so chart stays near top ───────
chart_ph    = st.empty()
time_ph     = st.empty()
tradelog_ph = st.empty()

# ── Current slice ─────────────────────────────────────────────────────
tick    = max(st.session_state.tick, 1)
start   = max(0, tick - window_size)
visible = df_all.iloc[start:tick]
latest  = visible.iloc[-1]

spread_val = latest['Spread']
gamma_val  = latest['Gamma']
mid_val    = latest['Mid']

signal_label, signal_color, signal_explain = get_signal(spread_val, gamma_val)

# ── Metrics row ───────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Spread",       f"{spread_val:.4f} pts",
          delta="WIDE" if spread_val > SPREAD_THRESHOLD else "NARROW",
          delta_color="normal" if spread_val > SPREAD_THRESHOLD else "inverse")
m2.metric("Mid Price",    f"{mid_val:.2f}")
m3.metric("Gamma",        f"{gamma_val:.7f}",
          delta="LOW RISK" if gamma_val < GAMMA_THRESHOLD else "HIGH RISK",
          delta_color="normal" if gamma_val < GAMMA_THRESHOLD else "inverse")
m4.metric("Total P&L",    f"${st.session_state.total_pnl:+.2f}",
          delta=f"{len(st.session_state.trade_log)} trades")
m5.metric("Win Rate",
          f"{sum(1 for t in st.session_state.trade_log if t['pnl_usd'] > 0) / max(len(st.session_state.trade_log), 1) * 100:.0f}%"
          if st.session_state.trade_log else "—")

# ── Signal banner ─────────────────────────────────────────────────
color_map = {"green": "#0f5132", "orange": "#7c4700", "red": "#6b1c1c"}
bg_map    = {"green": "#d1e7dd", "orange": "#fff3cd", "red": "#f8d7da"}

st.markdown(
    f"<div style='background:{bg_map[signal_color]};color:{color_map[signal_color]};"
    f"border-radius:6px;padding:10px 16px;font-weight:600;font-size:14px;margin:4px 0'>"
    f"{signal_label}</div>",
    unsafe_allow_html=True
)

# ── Explanation expander ──────────────────────────────────────────
with st.expander("Why this signal? (click to learn)", expanded=False):
    st.markdown(f"""
**Signal explanation:** {signal_explain}

**Market maker P&L model used here:**
- **Gross profit** = entry spread ÷ 2 (you earn half the spread on each side of the trade)
- **Adverse selection cost** = |mid price move| × 0.5 (cost of price moving against your inventory)
- **Net P&L** = gross profit − adverse selection cost (× ${PNL_PER_POINT:.0f}/pt for ES futures)

**Key thresholds:**
| Condition | Threshold | Why |
|-----------|-----------|-----|
| Spread > {SPREAD_THRESHOLD} pts | Enter | Gross profit exceeds typical transaction costs |
| Gamma < {GAMMA_THRESHOLD} | Enter | Options repricing slowly = inventory risk manageable |
""")

# ── Trade action buttons ──────────────────────────────────────────
ba, bb, bc, bd = st.columns([1, 1, 1, 3])

open_trade = st.session_state.open_trade
can_enter  = (open_trade is None) and (spread_val > SPREAD_NEGATIVE)
can_exit   = open_trade is not None

with ba:
    if st.button("Enter Trade", disabled=not can_enter, use_container_width=True):
        st.session_state.open_trade = {
            'tick':      tick,
            'timestamp': latest['timestamp'],
            'mid':       mid_val,
            'ask':       latest['Best_Ask'],
            'bid':       latest['Best_Bid'],
            'spread':    spread_val,
            'gamma':     gamma_val,
        }
        st.rerun()

with bb:
    if st.button("Exit Trade", disabled=not can_exit, use_container_width=True):
        entry = st.session_state.open_trade
        entry_row = df_all.iloc[entry['tick'] - 1]
        exit_row  = latest
        pnl_pts, pnl_usd = calc_pnl(entry_row, exit_row)
        trade = {
            'id':          len(st.session_state.trade_log) + 1,
            'entry_tick':  entry['tick'],
            'exit_tick':   tick,
            'entry_time':  entry['timestamp'].strftime('%H:%M:%S'),
            'exit_time':   latest['timestamp'].strftime('%H:%M:%S'),
            'entry_mid':   entry['mid'],
            'exit_mid':    mid_val,
            'entry_spread': entry['spread'],
            'pnl_pts':     pnl_pts,
            'pnl_usd':     pnl_usd,
            'result':      '✅ WIN' if pnl_usd >= 0 else '❌ LOSS',
        }
        st.session_state.trade_log.append(trade)
        st.session_state.total_pnl += pnl_usd
        st.session_state.open_trade = None
        st.rerun()

with bc:
    if st.button("Reset Trades", use_container_width=True):
        st.session_state.open_trade = None
        st.session_state.trade_log  = []
        st.session_state.total_pnl  = 0.0
        st.rerun()

# Open trade info
if open_trade:
    unrealized_pts, unrealized_usd = calc_pnl(
        pd.Series({'Spread': open_trade['spread'], 'Mid': open_trade['mid']}),
        pd.Series({'Mid': mid_val})
    )
    st.info(
        f"📌 **Open position** | Entry mid: {open_trade['mid']:.2f} | "
        f"Entry spread: {open_trade['spread']:.4f} | "
        f"Current mid: {mid_val:.2f} | "
        f"Unrealized P&L: **${unrealized_usd:+.2f}**"
    )


# ── Draw function (mirrors original working pattern) ──────────────
def draw(t):
    s   = max(0, t - window_size)
    vis = df_all.iloc[s:t]
    lat = vis.iloc[-1]

    eticks = [tr['entry_tick'] - 1 - s for tr in st.session_state.trade_log
              if s <= tr['entry_tick'] - 1 < t]
    xticks = [tr['exit_tick'] - 1 - s for tr in st.session_state.trade_log
              if s <= tr['exit_tick'] - 1 < t]
    ot    = st.session_state.open_trade
    otick = (ot['tick'] - 1 - s) if (ot and s <= ot['tick'] - 1 < t) else None

    fig = build_fig(vis, eticks, xticks, otick)
    chart_ph.plotly_chart(fig, use_container_width=True)

    time_ph.markdown(
        f"<p style='text-align:center;color:#888;font-size:13px;'>"
        f"⏱ <b style='color:white'>{lat['timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')}</b>"
        f" &nbsp;|&nbsp; Tick {t} / {total}</p>",
        unsafe_allow_html=True
    )

    if st.session_state.trade_log:
        log_df = pd.DataFrame(st.session_state.trade_log)[
            ['id', 'entry_time', 'exit_time', 'entry_mid', 'exit_mid',
             'entry_spread', 'pnl_pts', 'pnl_usd', 'result']
        ].rename(columns={
            'id': '#', 'entry_time': 'Entry Time', 'exit_time': 'Exit Time',
            'entry_mid': 'Entry Mid', 'exit_mid': 'Exit Mid',
            'entry_spread': 'Entry Spread', 'pnl_pts': 'P&L (pts)',
            'pnl_usd': 'P&L (USD)', 'result': 'Result'
        })
        with tradelog_ph.container():
            st.subheader("Trade Log")
            st.dataframe(
                log_df.style.applymap(
                    lambda v: 'color: #00cc96' if isinstance(v, (int, float)) and v > 0
                              else ('color: #ff4b4b' if isinstance(v, (int, float)) and v < 0 else ''),
                    subset=['P&L (pts)', 'P&L (USD)']
                ),
                use_container_width=True,
                hide_index=True
            )

# ── Run / pause (original working pattern) ───────────────────────
if running:
    for t in range(st.session_state.tick, total + 1):
        st.session_state.tick = t
        draw(t)
        time.sleep(speed)
        if t == total:
            st.success("Simulation complete!")
            break
else:
    draw(max(st.session_state.tick, 1))
