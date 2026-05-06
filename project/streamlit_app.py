import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import time

st.set_page_config(page_title='Futures Liquidity Pro', layout='wide')

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
SPREAD_NEGATIVE  = 0.0
GAMMA_THRESHOLD  = 0.0006
PNL_PER_POINT    = 50.0
PRED_STEPS       = 5
FILL_LOOKAHEAD = 15

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

# ── Prediction ────────────────────────────────────────────────────
def predict_next(series, steps=PRED_STEPS, window=5):
    if len(series) < 2:
        return np.full(steps, series.iloc[-1])
    recent = series.iloc[-window:].values
    x = np.arange(len(recent))
    slope, intercept = np.polyfit(x, recent, 1)
    return intercept + slope * np.arange(len(recent), len(recent) + steps)

# ── Signal logic ──────────────────────────────────────────────────
def get_signal(spread, gamma):
    if spread < SPREAD_NEGATIVE:
        return ("BAD DATA — Negative spread (crossed market)", "red",
                f"Spread is {spread:.4f} pts — crossed market, data stale. Never enter.")
    elif spread == 0.0:
        return ("LOCKED MARKET — Spread is zero", "orange",
                "Bid and ask are the same price. Wait for spread to open up.")
    elif spread == 0.25:
        return ("ENTER — Normal 1-tick spread (0.25 pts)", "green",
                f"Standard 1 ES tick spread. Earns $12.50 gross per round-trip. Gamma {gamma:.7f} — low risk.")
    elif spread > 0.25:
        return (f"BEST ENTRY — Wide spread ({spread:.4f} pts)", "green",
                f"Wider than normal — best profit opportunity: ${spread/2*50:.2f} gross per round-trip.")
    else:
        return ("WAIT — Spread too narrow", "red",
                f"Spread {spread:.4f} pts — between 0 and 1 tick. Wait for clean 0.25 tick.")

# ── Market Making P&L ─────────────────────────────────────────────
def is_clean_market(row):
    return (
        pd.notna(row["Best_Ask"])
        and pd.notna(row["Best_Bid"])
        and row["Best_Ask"] > row["Best_Bid"]
        and row["Spread"] > 0
    )


def simulate_quote_fill(quote_row, future_rows):
    """
    Market maker posts BOTH:
    - buy limit at bid
    - sell limit at ask

    Profit happens only if both sides fill.
    """

    bid_price = quote_row["Best_Bid"]
    ask_price = quote_row["Best_Ask"]
    spread = quote_row["Spread"]

    if spread <= 0:
        return None

    buy_filled = False
    sell_filled = False
    buy_time = None
    sell_time = None

    for _, row in future_rows.iterrows():
        # Skip bad market states
        if not is_clean_market(row):
            continue

        # Buy order fills if market ask trades down to our bid
        if not buy_filled and row["Best_Ask"] <= bid_price:
            buy_filled = True
            buy_time = row["timestamp"]

        # Sell order fills if market bid trades up to our ask
        if not sell_filled and row["Best_Bid"] >= ask_price:
            sell_filled = True
            sell_time = row["timestamp"]

        if buy_filled and sell_filled:
            pnl_pts = ask_price - bid_price
            pnl_usd = pnl_pts * PNL_PER_POINT

            return {
                "filled": True,
                "buy_time": buy_time,
                "sell_time": sell_time,
                "bid_price": bid_price,
                "ask_price": ask_price,
                "spread": spread,
                "pnl_pts": round(pnl_pts, 4),
                "pnl_usd": round(pnl_usd, 2),
                "result": "✅ SPREAD CAPTURED"
            }

    return {
        "filled": False,
        "buy_time": buy_time,
        "sell_time": sell_time,
        "bid_price": bid_price,
        "ask_price": ask_price,
        "spread": spread,
        "pnl_pts": 0.0,
        "pnl_usd": 0.0,
       "result": " PARTIAL / NOT BOTH FILLED"
    }
# # ── P&L ───────────────────────────────────────────────────────────
# def calc_pnl(entry_row, exit_row):
#     spread_capture = entry_row['Spread'] / 2
#     mid_move       = abs(exit_row['Mid'] - entry_row['Mid'])
#     pnl_pts        = spread_capture - mid_move * 0.5
#     return round(pnl_pts, 4), round(pnl_pts * PNL_PER_POINT, 2)

# ── Build figure ──────────────────────────────────────────────────
def build_fig(df_slice, entry_ticks=None, exit_ticks=None, open_tick=None):
    ts  = df_slice['timestamp']
    ask = df_slice['Best_Ask']
    bid = df_slice['Best_Bid']

    # Prediction
    pred_ts  = pd.date_range(ts.iloc[-1], periods=PRED_STEPS + 1, freq='s')[1:]
    pred_ask = predict_next(ask)
    pred_bid = predict_next(bid)

    fig = go.Figure()

    # Spread fill
    fig.add_trace(go.Scatter(
        x=pd.concat([ts, ts[::-1]]),
        y=pd.concat([ask, bid[::-1]]),
        fill='toself', fillcolor='rgba(150,150,150,0.12)',
        line=dict(width=0), showlegend=True, name='Spread zone', hoverinfo='skip'
    ))

    # Best Ask
    fig.add_trace(go.Scatter(
        x=ts, y=ask,
        line=dict(color='#ff4b4b', width=1.5), name='Best Ask',
        customdata=np.stack([
            df_slice['Spread'].round(4),
            df_slice['Mid'].round(4),
            df_slice['Gamma'].round(7),
            np.where((df_slice['Spread'] > 0.25) & (df_slice['Gamma'] < GAMMA_THRESHOLD), '🟢 HIGH', '🔴 LOW')
        ], axis=-1),
        hovertemplate=(
            '<b>Ask:</b> %{y:.2f}<br>'
            '<b>Spread:</b> %{customdata[0]}<br>'
            '<b>Mid:</b> %{customdata[1]}<br>'
            '<b>Gamma:</b> %{customdata[2]}<br>'
            '<b>Profitability:</b> %{customdata[3]}<extra></extra>'
        )
    ))

    # Best Bid
    fig.add_trace(go.Scatter(
        x=ts, y=bid,
        line=dict(color='#00cc96', width=1.5), name='Best Bid',
        hovertemplate='<b>Bid:</b> %{y:.2f}<extra></extra>'
    ))

    # Predicted Ask (dashed)
    fig.add_trace(go.Scatter(
        x=[ts.iloc[-1]] + list(pred_ts),
        y=[ask.iloc[-1]] + list(pred_ask),
        line=dict(color='#ff4b4b', width=1, dash='dash'),
        name='Predicted Ask', opacity=0.6
    ))

    # Predicted Bid (dashed)
    fig.add_trace(go.Scatter(
        x=[ts.iloc[-1]] + list(pred_ts),
        y=[bid.iloc[-1]] + list(pred_bid),
        line=dict(color='#00cc96', width=1, dash='dash'),
        name='Predicted Bid', opacity=0.6
    ))

    # "Now" vertical line
    fig.add_vline(x=ts.iloc[-1], line=dict(color='white', width=1, dash='dot'))

    # Entry markers
    if entry_ticks:
        etimes = [df_slice.iloc[i]['timestamp'] for i in entry_ticks if i < len(df_slice)]
        emids  = [df_slice.iloc[i]['Mid']       for i in entry_ticks if i < len(df_slice)]
        if etimes:
            fig.add_trace(go.Scatter(
                x=etimes, y=emids, mode='markers',
                marker=dict(symbol='triangle-up', size=14, color='#00aaff', line=dict(color='white', width=1)),
                name='Entry', hovertemplate='<b>ENTRY</b> at %{y:.2f}<extra></extra>'
            ))

    # Exit markers
    if exit_ticks:
        xtimes = [df_slice.iloc[i]['timestamp'] for i in exit_ticks if i < len(df_slice)]
        xmids  = [df_slice.iloc[i]['Mid']       for i in exit_ticks if i < len(df_slice)]
        if xtimes:
            fig.add_trace(go.Scatter(
                x=xtimes, y=xmids, mode='markers',
                marker=dict(symbol='x', size=14, color='#ffaa00', line=dict(color='white', width=1.5)),
                name='Exit', hovertemplate='<b>EXIT</b> at %{y:.2f}<extra></extra>'
            ))

    # Open position marker
    if open_tick is not None and open_tick < len(df_slice):
        ot = df_slice.iloc[open_tick]
        fig.add_trace(go.Scatter(
            x=[ot['timestamp']], y=[ot['Mid']], mode='markers',
            marker=dict(symbol='circle-open', size=18, color='#00aaff', line=dict(width=2.5)),
            name='Open position', hovertemplate='<b>OPEN</b> at %{y:.2f}<extra></extra>'
        ))

    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0e1117',
        height=500, margin=dict(l=50, r=20, t=20, b=60),
        legend=dict(orientation='h', y=1.06, x=0), hovermode='x unified'
    )
    fig.update_xaxes(tickformat='%H:%M:%S', rangeslider=dict(visible=True, thickness=0.04))
    fig.update_yaxes(title_text='Price (ES)')
    return fig

# ── Session state ─────────────────────────────────────────────────
for k, v in {'tick': 1, 'last_date': None, 'open_trade': None,
             'trade_log': [], 'total_pnl': 0.0}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Load & filter by date ─────────────────────────────────────────
df_all = load_data()
st.title("Futures Market Maker Simulation")

col0, col1, col2, col3 = st.columns([1, 1, 1, 2])
with col0:
    available_dates = sorted(df_all['timestamp'].dt.date.unique())
    selected_label  = st.selectbox("Date", [str(d) for d in available_dates])
    selected_date   = pd.Timestamp(selected_label).date()

df_all = df_all[df_all['timestamp'].dt.date == selected_date].reset_index(drop=True)
total  = len(df_all)

if st.session_state.last_date != selected_label:
    st.session_state.update({'tick': 1, 'last_date': selected_label,
                              'open_trade': None, 'trade_log': [], 'total_pnl': 0.0})

with col1:
    speed = st.slider("Speed (s/tick)", 0.1, 2.0, 0.5, 0.1)
with col2:
    window_size = st.slider("Display window (ticks)", 10, total, min(60, total))
with col3:
    running = st.toggle("Run simulation", value=False)

# ── Placeholders ──────────────────────────────────────────────────
chart_ph    = st.empty()
time_ph     = st.empty()
tradelog_ph = st.empty()

# ── Current state ─────────────────────────────────────────────────
tick    = max(st.session_state.tick, 1)
start   = max(0, tick - window_size)
visible = df_all.iloc[start:tick]
latest  = visible.iloc[-1]

spread_val = latest['Spread']
gamma_val  = latest['Gamma']
mid_val    = latest['Mid']
signal_label, signal_color, signal_explain = get_signal(spread_val, gamma_val)

# ── Metrics ───────────────────────────────────────────────────────
m2, m3, m4, m5 = st.columns(4)
m2.metric("Mid Price", f"{mid_val:.2f}")
m3.metric("Gamma", f"{gamma_val:.7f}",
          delta="LOW RISK" if gamma_val < GAMMA_THRESHOLD else "HIGH RISK",
          delta_color="normal" if gamma_val < GAMMA_THRESHOLD else "inverse")
m4.metric("Total P&L", f"${st.session_state.total_pnl:+.2f}",
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
    f"{signal_label}</div>", unsafe_allow_html=True
)

with st.expander("Why this signal?", expanded=False):
    st.markdown(f"""
**Signal explanation:** {signal_explain}

**P&L model:**
- Gross profit = entry spread ÷ 2
- Adverse selection cost = |mid price move| × 0.5
- Net P&L = gross profit − adverse selection cost (× ${PNL_PER_POINT:.0f}/pt)

| Condition | Threshold |
|-----------|-----------|
| Spread > 0 pts | Quote both sides |
| Spread = 0 | Locked market — avoid |
| Spread < 0 | Crossed/bad market — reject |
| Both bid and ask fill | Capture spread |
| Gamma < {GAMMA_THRESHOLD} | Low risk |
""")

# # ── Trade button ──────────────────────────────────────────────────
# open_trade = st.session_state.open_trade
# ba, bb, _ = st.columns([1, 1, 4])

# with ba:
#     if open_trade is None:
#         # No open position → quote both sides simultaneously
#         btn_label    = "🟢 Quote Market (Buy + Sell)"
#         btn_disabled = spread_val <= SPREAD_NEGATIVE
#     else:
#         # Open position → close both sides
#         btn_label    = "🔴 Close Quote"
#         btn_disabled = False

#     if st.button(btn_label, disabled=btn_disabled, use_container_width=True):
#         if open_trade is None:
#             # Open: post bid AND ask at the same time
#             st.session_state.open_trade = {
#                 'tick': tick, 'timestamp': latest['timestamp'],
#                 'mid': mid_val,
#                 'ask': latest['Best_Ask'], 'bid': latest['Best_Bid'],
#                 'spread': spread_val, 'gamma': gamma_val,
#             }
#         else:
#             # Close: collect spread from both sides
#             entry = st.session_state.open_trade
#             pnl_pts, pnl_usd = calc_pnl(
#                 pd.Series({'Spread': entry['spread'], 'Mid': entry['mid']}),
#                 pd.Series({'Mid': mid_val})
#             )
#             st.session_state.trade_log.append({
#                 'id':           len(st.session_state.trade_log) + 1,
#                 'entry_tick':   entry['tick'], 'exit_tick': tick,
#                 'entry_time':   entry['timestamp'].strftime('%H:%M:%S'),
#                 'exit_time':    latest['timestamp'].strftime('%H:%M:%S'),
#                 'entry_mid':    entry['mid'], 'exit_mid': mid_val,
#                 'entry_spread': entry['spread'],
#                 'pnl_pts':      pnl_pts, 'pnl_usd': pnl_usd,
#                 'result':       '✅ WIN' if pnl_usd >= 0 else '❌ LOSS',
#             })
#             st.session_state.total_pnl  += pnl_usd
#             st.session_state.open_trade  = None
#         st.rerun()

# with bb:
#     if st.button("Reset", use_container_width=True):
#         st.session_state.update({'open_trade': None, 'trade_log': [], 'total_pnl': 0.0})
#         st.rerun()

# if open_trade:
#     _, unr_usd = calc_pnl(
#         pd.Series({'Spread': open_trade['spread'], 'Mid': open_trade['mid']}),
#         pd.Series({'Mid': mid_val})
#     )
#     st.info(
#         f"📌 **Quoting both sides** | "
#         f"Bid: {open_trade['bid']:.2f} / Ask: {open_trade['ask']:.2f} | "
#         f"Spread: {open_trade['spread']:.4f} pts | "
#         f"Unrealized P&L: **${unr_usd:+.2f}**"
#     )
# ── Market Making Button ──────────────────────────────────────────
ba, bb, _ = st.columns([1.4, 1, 3.6])

clean_market = is_clean_market(latest)
safe_gamma = gamma_val < GAMMA_THRESHOLD
can_quote = clean_market and safe_gamma

with ba:
    if st.button(
        "🟢 Quote Both Sides",
        disabled=not can_quote,
        use_container_width=True
    ):
        lookahead = df_all.iloc[tick:min(tick + FILL_LOOKAHEAD, total)]

        fill_result = simulate_quote_fill(latest, lookahead)

        if fill_result is None:
            st.warning("Bad market state. Quote rejected.")
        else:
            st.session_state.trade_log.append({
                "id": len(st.session_state.trade_log) + 1,
                "quote_tick": tick,
                "quote_time": latest["timestamp"].strftime("%H:%M:%S"),
                "bid_price": fill_result["bid_price"],
                "ask_price": fill_result["ask_price"],
                "spread": fill_result["spread"],
                "buy_time": fill_result["buy_time"].strftime("%H:%M:%S") if fill_result["buy_time"] is not None else "—",
                "sell_time": fill_result["sell_time"].strftime("%H:%M:%S") if fill_result["sell_time"] is not None else "—",
                "pnl_pts": fill_result["pnl_pts"],
                "pnl_usd": fill_result["pnl_usd"],
                "result": fill_result["result"],
            })

            st.session_state.total_pnl += fill_result["pnl_usd"]

        st.rerun()

with bb:
    if st.button("Reset", use_container_width=True):
        st.session_state.update({
            "open_trade": None,
            "trade_log": [],
            "total_pnl": 0.0
        })
        st.rerun()
# ── Draw ──────────────────────────────────────────────────────────
def draw(t):
    s   = max(0, t - window_size)
    vis = df_all.iloc[s:t]
    lat = vis.iloc[-1]
    quote_ticks = [
        tr['quote_tick'] - 1 - s
        for tr in st.session_state.trade_log
        if s <= tr['quote_tick'] - 1 < t
    ]

    chart_ph.plotly_chart(build_fig(vis, quote_ticks), use_container_width=True)

    time_ph.markdown(
        f"<p style='text-align:center;color:#888;font-size:13px;'>"
        f"⏱ <b style='color:white'>{lat['timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')}</b>"
        f" &nbsp;|&nbsp; Tick {t} / {total}</p>", unsafe_allow_html=True
    )
    if st.session_state.trade_log:
        log_df = pd.DataFrame(st.session_state.trade_log)[
            [
                "id", "quote_time", "bid_price", "ask_price",
                "spread", "buy_time", "sell_time",
                "pnl_pts", "pnl_usd", "result"
            ]
        ].rename(columns={
            "id": "#",
            "quote_time": "Quote Time",
            "bid_price": "Quoted Bid",
            "ask_price": "Quoted Ask",
            "spread": "Spread",
            "buy_time": "Buy Fill",
            "sell_time": "Sell Fill",
            "pnl_pts": "P&L (pts)",
            "pnl_usd": "P&L (USD)",
            "result": "Result"
        })

        with tradelog_ph.container():
            st.subheader("Market Making Log")
            st.dataframe(
                log_df.style.map(
                    lambda v: "color:#00cc96" if isinstance(v, (int, float)) and v > 0
                    else ("color:#ff4b4b" if isinstance(v, (int, float)) and v < 0 else ""),
                    subset=["P&L (pts)", "P&L (USD)"]
                ),
                use_container_width=True,
                hide_index=True
            )

    # if st.session_state.trade_log:
    #     log_df = pd.DataFrame(st.session_state.trade_log)[
    #         ['id','entry_time','exit_time','entry_mid','exit_mid','entry_spread','pnl_pts','pnl_usd','result']
    #     ].rename(columns={'id':'#','entry_time':'Entry','exit_time':'Exit',
    #                       'entry_mid':'Entry Mid','exit_mid':'Exit Mid',
    #                       'entry_spread':'Spread','pnl_pts':'P&L (pts)','pnl_usd':'P&L (USD)','result':'Result'})
    #     with tradelog_ph.container():
    #         st.subheader("Trade Log")
    #         st.dataframe(log_df.style.map(
    #             lambda v: 'color:#00cc96' if isinstance(v,(int,float)) and v>0
    #                  else ('color:#ff4b4b' if isinstance(v,(int,float)) and v<0 else ''),
    #             subset=['P&L (pts)','P&L (USD)']), use_container_width=True, hide_index=True)

# ── Run / pause ───────────────────────────────────────────────────
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