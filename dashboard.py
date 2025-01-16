import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
from alpaca_trade_api.rest import REST
import os
from dotenv import load_dotenv

# Load environment variables and configure page
st.set_page_config(page_title="Trading Performance Dashboard", layout="wide")

# Initialize session state for refresh tracking
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.min

def load_api_keys():
    """Load API keys from environment variables or Streamlit secrets"""
    if os.path.exists('.env'):
        load_dotenv()
        return {
            'ALPACA_API_KEY': os.getenv('ALPACA_API_KEY'),
            'ALPACA_SECRET_KEY': os.getenv('ALPACA_SECRET_KEY'),
            'ALPACA_BASE_URL': os.getenv('ALPACA_BASE_URL')
        }
    else:
        return st.secrets['alpaca']

def should_refresh():
    """Check if data should be refreshed based on market hours and last refresh"""
    now = datetime.now(pytz.timezone('US/Eastern'))
    last_refresh = st.session_state.last_refresh
    
    # Check if it's a weekday
    if now.weekday() > 4:  # Saturday = 5, Sunday = 6
        return False
    
    # Check market hours (9:30 AM - 4:00 PM ET)
    market_open = now.replace(hour=9, minute=30)
    market_close = now.replace(hour=16, minute=0)
    is_market_hours = market_open <= now <= market_close
    
    # Refresh if it's been an hour and we're in market hours
    return (now - last_refresh) >= timedelta(hours=1) and is_market_hours

def create_metrics_cards(trade_stats):
    """Create metric cards for key statistics"""
    cols = st.columns(4)
    
    metrics = [
        ("Win Rate", f"{trade_stats['Win Rate %']}%"),
        ("Total Profit", f"${trade_stats['Total Profit']:,.2f}"),
        ("Avg Profit/Trade", f"${trade_stats['Average Profit per Trade']:,.2f}"),
        ("Total Trades", trade_stats['Total Trades'])
    ]
    
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)

def plot_daily_performance(daily_summary):
    """Create daily performance chart"""
    fig = go.Figure()
    
    # Add profit line
    fig.add_trace(go.Scatter(
        x=daily_summary.index,
        y=daily_summary['Total Profit'],
        name='Total Profit',
        line=dict(color='green', width=2)
    ))
    
    # Add win rate bars
    fig.add_trace(go.Bar(
        x=daily_summary.index,
        y=daily_summary['Win Rate %'],
        name='Win Rate %',
        yaxis='y2',
        marker_color='blue',
        opacity=0.3
    ))
    
    fig.update_layout(
        title='Daily Performance',
        yaxis=dict(title='Profit ($)', side='left'),
        yaxis2=dict(title='Win Rate (%)', side='right', overlaying='y', range=[0, 100]),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_stock_performance(stock_summary):
    """Create stock performance visualization"""
    fig = px.scatter(
        stock_summary,
        x='Win_Rate_Pct',
        y='Average_Profit',
        size='Total_Trades',
        color='Total_Profit',
        hover_data=['Symbol', 'Total_Trades', 'Total_Profit'],
        title='Stock Performance Analysis'
    )
    
    fig.update_layout(
        xaxis_title='Win Rate (%)',
        yaxis_title='Average Profit per Trade ($)',
        coloraxis_colorbar_title='Total Profit ($)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("Trading Performance Dashboard")
    
    # Load API keys
    api_keys = load_api_keys()
    api = REST(api_keys['ALPACA_API_KEY'], 
              api_keys['ALPACA_SECRET_KEY'], 
              api_keys['ALPACA_BASE_URL'])
    
    # Check if we should refresh data
    if should_refresh() or 'data' not in st.session_state:
        with st.spinner('Refreshing trading data...'):
            # Your existing analysis code here
            executed_trades = get_executed_trades(api)
            df_trades = pd.DataFrame(executed_trades)
            matched_trades = match_buy_and_sell_trades(df_trades)
            trade_stats = calculate_trade_stats(matched_trades, api)
            stock_summary = calculate_stock_summary(matched_trades)
            daily_summary, daily_performers = calculate_daily_summary(matched_trades)
            current_holdings = get_current_holdings(api)
            
            # Store in session state
            st.session_state.data = {
                'trade_stats': trade_stats,
                'stock_summary': stock_summary,
                'daily_summary': daily_summary,
                'daily_performers': daily_performers,
                'current_holdings': current_holdings
            }
            st.session_state.last_refresh = datetime.now(pytz.timezone('US/Eastern'))
    
    # Display last refresh time
    st.caption(f"Last refreshed: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S ET')}")
    
    # Create dashboard layout
    create_metrics_cards(st.session_state.data['trade_stats'])
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Daily Performance", "Stock Analysis", "Current Holdings"])
    
    with tab1:
        plot_daily_performance(st.session_state.data['daily_summary'])
        
    with tab2:
        plot_stock_performance(st.session_state.data['stock_summary'])
        
    with tab3:
        st.dataframe(st.session_state.data['current_holdings'])
    
    # Schedule next refresh
    if should_refresh():
        st.experimental_rerun()

if __name__ == "__main__":
    main()
