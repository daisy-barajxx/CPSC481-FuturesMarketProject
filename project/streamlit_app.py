import streamlit as st
import pandas as pd
import numpy as np
import ast
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title='Futures Liquidity Pro', layout='wide')

@st.cache_data
def get_clean_data():
    # Load data
    # DATA_FILENAME = Path(__file__).parent/'futures_0414.csv'
    # df = pd.read_csv(DATA_FILENAME)
    
    # For this example, I'll assume 'df' is loaded from your CSV
    df = pd.read_csv('data/futures_0414.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. Parse MBO Sizes
    def parse_mbo(val):
        try:
            l = ast.literal_eval(val)
            return sum(l) if isinstance(l, list) else 0
        except: return 0
    df['total_size'] = df['MBO'].apply(parse_mbo)

    # 2. Extract Best Bid/Ask per second
    def get_snapshot(g):
        asks = g[g['Side'] == 'Ask']
        bids = g[g['Side'] == 'Bid']
        
        best_ask = asks['future_strike'].min()
        best_bid = bids['future_strike'].max()
        
        return pd.Series({
            'Best_Ask': best_ask,
            'Best_Bid': best_bid,
            'Spread': best_ask - best_bid if (best_ask and best_bid) else 0.25,
            'Mid_Price': (best_ask + best_bid) / 2 if (best_ask and best_bid) else g['current_es_price'].mean()/100,
            'Gamma': g['call_gamma'].mean()
        })

    # Group by second to make the chart readable
    summary = df.groupby(df['timestamp'].dt.floor('S')).apply(get_snapshot).reset_index()
    return summary

# --- UI ---
st.title("🛡️ Bid-Ask Spread Analysis")

try:
    df_plot = get_clean_data()

    # --- TOP METRICS (The Spread Calculator) ---
    avg_spread = df_plot['Spread'].mean()
    max_gamma = df_plot['Gamma'].max()
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Average Spread", f"{avg_spread:.2f} pts", help="Your potential profit per round-trip trade")
    c2.metric("Market Volatility (Gamma)", f"{max_gamma:.5f}")
    c3.metric("Profitability Rating", "HIGH" if avg_spread > 0.25 and max_gamma < 0.0006 else "LOW")

    # --- THE CHART ---
    st.subheader("Micro-Level Order Book (Bid vs Ask)")
    
    # Using Matplotlib for the "Fill Between" effect
    fig, ax1 = plt.subplots(figsize=(10, 4))
    plt.style.use('dark_background')

    # Plot Bid and Ask
    ax1.plot(df_plot['timestamp'], df_plot['Best_Ask'], color='#ff4b4b', label='Best Ask (Sellers)', linewidth=1)
    ax1.plot(df_plot['timestamp'], df_plot['Best_Bid'], color='#00cc96', label='Best Bid (Buyers)', linewidth=1)
    
    # SHADE THE SPREAD (Your Profit Zone)
    ax1.fill_between(df_plot['timestamp'], df_plot['Best_Bid'], df_plot['Best_Ask'], color='gray', alpha=0.3, label='The Spread')

    ax1.set_ylabel('Price (ES Points)')
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.1)

    # Overlay Gamma on a second axis
    ax2 = ax1.twinx()
    ax2.plot(df_plot['timestamp'], df_plot['Gamma'], color='#0088ff', alpha=0.5, linestyle='--', label='Gamma Risk')
    ax2.set_ylabel('Gamma')
    
    st.pyplot(fig)

    st.success(f"Analysis Complete: Over the last 59 seconds, the average spread was {avg_spread:.4f}. Providing liquidity is most profitable when the gray shaded area is wide and the blue dashed line (Gamma) is flat.")

except Exception as e:
    st.error(f"Waiting for data or error in processing: {e}")
    st.info("Make sure 'futures_0414.csv' is in the same folder as this script!")
    