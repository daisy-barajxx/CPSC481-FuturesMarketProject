import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import time

st.set_page_config(page_title='Futures Liquidity Pro', layout='wide')

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
def predict_next(series, steps=5, window=5):
    if len(series) < 2:
        return np.full(steps, series.iloc[-1])
    recent = series.iloc[-window:].values
    x = np.arange(len(recent))
    slope, intercept = np.polyfit(x, recent, 1)
    return intercept + slope * np.arange(len(recent), len(recent) + steps)

# ── Build Plotly figure ───────────────────────────────────────────
def build_fig(df_all, tick):
    visible = df_all.iloc[:tick]
    ts = visible['timestamp']

    pred_steps = 5
    pred_ts  = pd.date_range(ts.iloc[-1], periods=pred_steps + 1, freq='s')[1:]
    pred_ask = predict_next(visible['Best_Ask'], steps=pred_steps)
    pred_bid = predict_next(visible['Best_Bid'], steps=pred_steps)

    fig = go.Figure()

    # Spread fill
    fig.add_trace(go.Scatter(
        x=pd.concat([ts, ts[::-1]]),
        y=pd.concat([visible['Best_Ask'], visible['Best_Bid'][::-1]]),
        fill='toself', fillcolor='rgba(150,150,150,0.15)',
        line=dict(width=0), showlegend=True, name='Spread zone',
        hoverinfo='skip'
    ))

    # Best Ask
    fig.add_trace(go.Scatter(
        x=ts, y=visible['Best_Ask'],
        line=dict(color='#ff4b4b', width=1.5),
        name='Best Ask',
        customdata=np.stack([
            visible['Spread'].round(4),
            visible['Mid'].round(4),
            visible['Gamma'].round(7),
            np.where((visible['Spread'] > 0.25) & (visible['Gamma'] < 0.0006), '🟢 HIGH', '🔴 LOW')
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
        x=ts, y=visible['Best_Bid'],
        line=dict(color='#00cc96', width=1.5),
        name='Best Bid',
        hovertemplate='<b>Bid:</b> %{y:.2f}<extra></extra>'
    ))

    # Predicted Ask (dashed)
    fig.add_trace(go.Scatter(
        x=[ts.iloc[-1]] + list(pred_ts),
        y=[visible['Best_Ask'].iloc[-1]] + list(pred_ask),
        line=dict(color='#ff4b4b', width=1, dash='dash'),
        name='Predicted Ask', opacity=0.6
    ))

    # Predicted Bid (dashed)
    fig.add_trace(go.Scatter(
        x=[ts.iloc[-1]] + list(pred_ts),
        y=[visible['Best_Bid'].iloc[-1]] + list(pred_bid),
        line=dict(color='#00cc96', width=1, dash='dash'),
        name='Predicted Bid', opacity=0.6
    ))

    # "Now" vertical line
    fig.add_vline(x=ts.iloc[-1], line=dict(color='white', width=1, dash='dot'))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0e1117',
        height=420,
        margin=dict(l=50, r=20, t=20, b=60),
        legend=dict(orientation='h', y=1.08, x=0),
        xaxis=dict(
            title='Time',
            rangeslider=dict(visible=True, thickness=0.06),
            tickformat='%H:%M:%S',
        ),
        yaxis=dict(title='Price (ES Points)'),
        hovermode='x unified'
    )
    return fig

# ── UI ────────────────────────────────────────────────────────────
st.title("🛡️ Futures Market — Live Simulation")

df_all = load_data()
total  = len(df_all)

col0, col1, col2, col3 = st.columns([1, 1, 1, 2])
with col0:
    available_dates = sorted(df_all['timestamp'].dt.date.unique())
    date_labels = [str(d) for d in available_dates]
    selected_label = st.selectbox("📅 Date", date_labels)
    selected_date  = pd.Timestamp(selected_label).date()

# Filter to selected date
df_all = df_all[df_all['timestamp'].dt.date == selected_date].reset_index(drop=True)
total  = len(df_all)

# Reset tick when date changes
if 'last_date' not in st.session_state or st.session_state.last_date != selected_label:
    st.session_state.tick = 1
    st.session_state.last_date = selected_label

with col1:
    speed = st.slider("Speed (s per tick)", 0.1, 2.0, 0.5, 0.1)
with col2:
    window_size = st.slider("Display window (ticks)", 10, total, min(60, total))
with col3:
    running = st.toggle("▶ Run simulation", value=False)

m1, m2, m3, m4 = st.columns(4)
metric_spread = m1.empty()
metric_mid    = m2.empty()
metric_gamma  = m3.empty()
metric_rating = m4.empty()

chart_placeholder = st.empty()
time_placeholder  = st.empty()

if 'tick' not in st.session_state:
    st.session_state.tick = 1
if 'last_date' not in st.session_state:
    st.session_state.last_date = None

# ── Draw ──────────────────────────────────────────────────────────
def draw(tick):
    start   = max(0, tick - window_size)
    visible = df_all.iloc[start:tick]
    latest  = visible.iloc[-1]

    spread_val = latest['Spread']
    gamma_val  = latest['Gamma']
    rating     = "🟢 HIGH" if spread_val > 0.25 and gamma_val < 0.0006 else "🔴 LOW"

    metric_spread.metric("Spread",      f"{spread_val:.2f} pts")
    metric_mid.metric("Mid Price",      f"{latest['Mid']:.2f}")
    metric_gamma.metric("Gamma",        f"{gamma_val:.7f}")
    metric_rating.metric("Profitability", rating)

    fig = build_fig(df_all.iloc[start:tick], tick - start)
    chart_placeholder.plotly_chart(fig, use_container_width=True)

    ts_str = latest['timestamp'].strftime('%Y-%m-%d  %H:%M:%S UTC')
    time_placeholder.markdown(
        f"<p style='text-align:center;color:#888;font-size:13px;'>"
        f"⏱ <b style='color:white'>{ts_str}</b> &nbsp;|&nbsp; Tick {tick} / {total}</p>",
        unsafe_allow_html=True
    )

# ── Run / pause ───────────────────────────────────────────────────
if running:
    for tick in range(st.session_state.tick, total + 1):
        st.session_state.tick = tick
        draw(tick)
        time.sleep(speed)
        if tick == total:
            st.success("Simulation complete!")
            break
else:
    draw(max(st.session_state.tick, 1))