import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
from alpaca_trade_api.rest import REST
import os
import time
import statistics
from dotenv import load_dotenv

# Load environment variables and configure page
st.set_page_config(page_title="Trading Performance Dashboard", layout="wide")

# Initialize session state for refresh tracking
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now(pytz.timezone('US/Eastern'))

def convert_to_est(timestamp):
    """Convert timestamp to EST timezone"""
    if pd.isna(timestamp):
        return None
    
    # Convert to pandas timestamp if string
    if isinstance(timestamp, str):
        timestamp = pd.Timestamp(timestamp)
    
    # Make timezone aware if naive
    if timestamp.tzinfo is None:
        timestamp = pytz.utc.localize(timestamp)
    
    # Convert to EST
    est = pytz.timezone('US/Eastern')
    return timestamp.astimezone(est)

def load_api_keys():
    """Load API keys for all three strategies from environment variables or Streamlit secrets"""
    if os.path.exists('.env'):
        load_dotenv()
        return {
            'strat_1': {
                'ALPACA_API_KEY': os.getenv('ALPACA_API_KEY'),
                'ALPACA_SECRET_KEY': os.getenv('ALPACA_SECRET_KEY'),
                'ALPACA_BASE_URL': os.getenv('ALPACA_BASE_URL')
            },
            'strat_2': {
                'ALPACA_API_KEY': os.getenv('ALPACA_API_KEY_STRAT_2'),
                'ALPACA_SECRET_KEY': os.getenv('ALPACA_SECRET_KEY_STRAT_2'),
                'ALPACA_BASE_URL': os.getenv('ALPACA_BASE_URL')
            },
            'strat_3': {
                'ALPACA_API_KEY': os.getenv('ALPACA_API_KEY_STRAT_3'),
                'ALPACA_SECRET_KEY': os.getenv('ALPACA_SECRET_KEY_STRAT_3'),
                'ALPACA_BASE_URL': os.getenv('ALPACA_BASE_URL')
            }
        }
    else:
        return {
            'strat_1': st.secrets['alpaca_strat_1'],
            'strat_2': st.secrets['alpaca_strat_2'],
            'strat_3': st.secrets['alpaca_strat_3']
        }

def should_refresh():
    """Check if data should be refreshed based on market hours and last refresh"""
    now = datetime.now(pytz.timezone('US/Eastern'))
    last_refresh = st.session_state.last_refresh
    
    # Convert last_refresh to timezone-aware if it isn't already
    if last_refresh.tzinfo is None:
        last_refresh = pytz.timezone('US/Eastern').localize(last_refresh)
    
    # Check if it's a weekday
    if now.weekday() > 4:  # Saturday = 5, Sunday = 6
        return False
    
    # Check market hours (9:30 AM - 4:00 PM ET)
    market_open = now.replace(hour=9, minute=30)
    market_close = now.replace(hour=16, minute=0)
    is_market_hours = market_open <= now <= market_close
    
    # Refresh if it's been an hour and we're in market hours
    return (now - last_refresh) >= timedelta(hours=1) and is_market_hours

def create_metrics_cards(trade_stats, trailing_stats):
    """Create metric cards for overall and trailing performance"""
    # Overall performance metrics
    st.subheader("Overall Performance")
    cols = st.columns(5)
    metrics = [
        ("Portfolio Value", f"${trade_stats['Portfolio Value']:,.2f}"),
        ("Portfolio Return", f"{trade_stats['Portfolio Return']}%"),
        ("Win Rate", f"{trade_stats['Win Rate %']}%"),
        ("Realized Profit", f"${trade_stats['Total Profit']:,.2f}"),
        ("Total Trades", trade_stats['Total Trades'])
    ]
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)
    
    # Trailing 7-day performance metrics
    st.subheader("Trailing 7-Day Performance")
    cols = st.columns(5)
    trailing_metrics = [
        ("Return", f"{trailing_stats['Return']}%"),
        ("Win Rate", f"{trailing_stats['Win Rate']}%"),
        ("Realized Profit", f"${trailing_stats['Realized Profit']:,.2f}"),
        ("Unrealized Profit", f"${trailing_stats['Unrealized Profit']:,.2f}"),
        ("Total Trades", trailing_stats['Total Trades'])
    ]
    
    for col, (label, value) in zip(cols, trailing_metrics):
        col.metric(label, value)


def plot_daily_performance(daily_summary):
    """Create daily performance chart, excluding days with no trades"""
    # Filter out days with no trades
    daily_summary = daily_summary[daily_summary['Number of Trades'] > 0]
    
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
    """Create stock performance visualization, excluding stocks with no trades"""
    # Filter out stocks with no trades
    stock_summary = stock_summary[stock_summary['Total_Trades'] > 0]
    
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

# Fetch all trades executed (this retrieves orders; we will filter for filled trades)
def get_executed_trades(api):
    CHUNK_SIZE = 500
    all_orders = []
    start_time = pd.Timestamp.utcnow()
    check_for_more_orders = True

    while check_for_more_orders:
        api_orders = api.list_orders(
            status='closed',
            until=start_time.isoformat(),
            direction='desc',
            limit=CHUNK_SIZE,
            nested=False,
        )

        all_orders.extend(api_orders)

        if len(api_orders) == CHUNK_SIZE:
            last_order_time = all_orders[-3].submitted_at
            start_time = pd.Timestamp(last_order_time).tz_convert('UTC') if pd.Timestamp(last_order_time).tzinfo else pd.Timestamp(last_order_time, tz='UTC')
        else:
            check_for_more_orders = False

    trades = []
    for order in all_orders:
        if order.filled_at and order.status == 'filled':
            # Convert timestamps to EST
            filled_at_est = convert_to_est(order.filled_at)
            submitted_at_est = convert_to_est(order.submitted_at)
            
            trades.append({
                'Order ID': order.id,
                'Symbol': order.symbol,
                'Side': order.side,
                'Quantity': order.qty,
                'Filled Quantity': order.filled_qty,
                'Filled Average Price': order.filled_avg_price,
                'Status': order.status,
                'Filled At': filled_at_est,
                'Submitted At': submitted_at_est,
                'Type': order.order_type,
                'Time in Force': order.time_in_force
            })

    return trades


def match_buy_and_sell_trades(df_trades):
    # Separate buy and sell trades
    buys = df_trades[df_trades['Side'] == 'buy'].copy()
    sells = df_trades[df_trades['Side'] == 'sell'].copy()

    # Convert 'Filled At' to datetime in EST if not already
    buys['Filled At'] = buys['Filled At'].apply(convert_to_est)
    sells['Filled At'] = sells['Filled At'].apply(convert_to_est)

    # Sort both DataFrames by 'Filled At'
    buys = buys.sort_values(by='Filled At')
    sells = sells.sort_values(by='Filled At')

    matches = []
    for _, sell in sells.iterrows():
        potential_matches = buys[
            (buys['Symbol'] == sell['Symbol']) &
            (buys['Filled Quantity'] == sell['Filled Quantity']) &
            (buys['Filled At'] < sell['Filled At'])
        ]

        if not potential_matches.empty:
            matched_buy = potential_matches.iloc[0]
            
            matches.append({
                'Buy Order ID': matched_buy['Order ID'],
                'Buy Filled At': matched_buy['Filled At'],
                'Sell Order ID': sell['Order ID'],
                'Sell Filled At': sell['Filled At'],
                'Symbol': sell['Symbol'],
                'Quantity': float(sell['Filled Quantity']),
                'Buy Price': matched_buy['Filled Average Price'],
                'Sell Price': sell['Filled Average Price'],
                'Profit': (float(sell['Filled Average Price']) - float(matched_buy['Filled Average Price'])) * float(sell['Filled Quantity']),
                'Trade Duration (mins)': (sell['Filled At'] - matched_buy['Filled At']).total_seconds() / 60
            })

            buys = buys.drop(matched_buy.name)

    matched_trades = pd.DataFrame(matches)
    return matched_trades


# Calculate statistics
def get_current_holdings(api):
    positions = api.list_positions()
    holdings = []
    for position in positions:
        holdings.append({
            'Symbol': position.symbol,
            'Quantity': float(position.qty),
            'Current Price': float(position.current_price),
            'Market Value': float(position.market_value),
            'Cost Basis': float(position.cost_basis),
            'Unrealized P&L': float(position.unrealized_pl),
            'Unrealized P&L %': float(position.unrealized_plpc) * 100
        })
    return pd.DataFrame(holdings)


def calculate_trade_stats(matched_trades, api, initial_balance=100000):
    matched_trades[['Sell Price', 'Buy Price']] = matched_trades[['Sell Price', 'Buy Price']].astype(float)
    profitable_trades = len(matched_trades[matched_trades['Profit'] > 0])
    total_profit = matched_trades['Profit'].sum()

    # Get actual account balance and portfolio value from Alpaca
    account = api.get_account()
    current_balance = float(account.portfolio_value)  # Total portfolio value including positions
    cash_balance = float(account.cash)  # Cash only

    stats = {
        'Portfolio Value': round(current_balance, 2),
        'Portfolio Return': round((current_balance - initial_balance) / initial_balance * 100, 2),
        'Cash Balance': round(cash_balance, 2),
        'Win Rate %': round((profitable_trades / len(matched_trades)) * 100, 2),
        'Return on Initial Capital %': round((total_profit / initial_balance) * 100, 2),
        'Total Trades': len(matched_trades),
        'Total Profit': round(matched_trades['Profit'].sum(), 2),
        'Average Profit per Trade': round(matched_trades['Profit'].mean(), 2),
        'Average Profit % per Trade': round((matched_trades['Profit'] /
                                             (matched_trades['Buy Price'] * matched_trades['Quantity']) * 100).mean(),
                                            2),
        'Max Profit': round(matched_trades['Profit'].max(), 2),
        'Max Profit %': round((matched_trades['Profit'] /
                               (matched_trades['Buy Price'] * matched_trades['Quantity']) * 100).max(), 2),
        'Min Profit': round(matched_trades['Profit'].min(), 2),
        'Min Profit %': round((matched_trades['Profit'] /
                               (matched_trades['Buy Price'] * matched_trades['Quantity']) * 100).min(), 2),
        'Average Trade Duration (mins)': round(matched_trades['Trade Duration (mins)'].mean(), 2)
    }
    return stats

def calculate_stock_summary(matched_trades):
    stock_summary = matched_trades.groupby('Symbol').agg(
        Win_Rate_Pct=('Profit', lambda x: round(len(x[x > 0]) / len(x) * 100, 2)),
        Total_Trades=('Quantity', 'count'),
        Total_Quantity=('Quantity', 'sum'),
        Total_Profit=('Profit', lambda x: round(x.sum(), 2)),
        Average_Profit=('Profit', lambda x: round(x.mean(), 2)),
        Average_Profit_Pct=('Profit', lambda x: round((x.mean() /
            (matched_trades.loc[x.index, 'Buy Price'] *
             matched_trades.loc[x.index, 'Quantity']).mean() * 100), 2)),
        Max_Profit=('Profit', lambda x: round(x.max(), 2)),
        Max_Profit_Pct=('Profit', lambda x: round((x.max() /
            (matched_trades.loc[x.index, 'Buy Price'] *
             matched_trades.loc[x.index, 'Quantity']).max() * 100), 2)),
        Min_Profit=('Profit', lambda x: round(x.min(), 2)),
        Min_Profit_Pct=('Profit', lambda x: round((x.min() /
            (matched_trades.loc[x.index, 'Buy Price'] *
             matched_trades.loc[x.index, 'Quantity']).min() * 100), 2)),
        Average_Trade_Duration=('Trade Duration (mins)', lambda x: round(x.mean(), 2))
    ).reset_index()
    return stock_summary

def calculate_daily_summary(matched_trades, days=9):
    # Ensure timestamps are in EST
    matched_trades['Sell Filled At'] = matched_trades['Sell Filled At'].apply(convert_to_est)
    
    # Get only the trades from last 5 trading days
    today = pd.Timestamp.now(tz=pytz.timezone('US/Eastern')).normalize()
    cutoff_date = today - pd.Timedelta(days=days)
    recent_trades = matched_trades[matched_trades['Sell Filled At'] >= cutoff_date].copy()

    recent_trades['Profit %'] = (((recent_trades['Sell Price'] - recent_trades['Buy Price']) / recent_trades['Buy Price']) * 100) - 1

    daily_summary = recent_trades.groupby(recent_trades['Sell Filled At'].dt.date).agg({
        'Profit': ['sum', 'count', lambda x: (x > 0).mean() * 100, 'mean'],
        'Profit %': 'mean',
        'Symbol': 'nunique',
        'Trade Duration (mins)': 'mean'
    }).round(2)

    daily_summary.columns = ['Total Profit', 'Number of Trades', 'Win Rate %', 'Avg Profit/Trade',
                           'Avg Profit %', 'Unique Symbols', 'Avg Duration (mins)']

    daily_performers = {}
    for date in recent_trades['Sell Filled At'].dt.date.unique():
        day_trades = recent_trades[recent_trades['Sell Filled At'].dt.date == date]
        symbol_profits = day_trades.groupby('Symbol')['Profit'].sum()

        if not symbol_profits.empty:
            best_symbol = symbol_profits.idxmax()
            worst_symbol = symbol_profits.idxmin()
            daily_performers[date] = {
                'Best': (best_symbol, symbol_profits[best_symbol]),
                'Worst': (worst_symbol, symbol_profits[worst_symbol])
            }

    return daily_summary, daily_performers

def combine_daily_summaries(daily_summaries):
    """Combine daily summaries from multiple strategies"""
    # Convert all summaries to dataframes if they aren't already
    dfs = [summary if isinstance(summary, pd.DataFrame) else pd.DataFrame(summary) 
           for summary in daily_summaries]
    
    # Combine all dataframes
    combined = pd.concat(dfs)
    
    # Group by date and aggregate
    return combined.groupby(combined.index).agg({
        'Total Profit': 'sum',
        'Number of Trades': 'sum',
        'Win Rate %': lambda x: round(x.mean(), 2),  # Round win rate to 2 decimal places
        'Avg Profit/Trade': 'mean',
        'Avg Profit %': 'mean',
        'Unique Symbols': 'sum',
        'Avg Duration (mins)': 'mean'
    })

def combine_stock_summaries(stock_summaries):
    """Combine stock summaries from multiple strategies"""
    # Convert all summaries to dataframes if they aren't already
    dfs = [summary if isinstance(summary, pd.DataFrame) else pd.DataFrame(summary) 
           for summary in stock_summaries]
    
    # Combine all dataframes
    combined = pd.concat(dfs)
    
    # Group by symbol and aggregate
    return combined.groupby('Symbol').agg({
        'Win_Rate_Pct': 'mean',
        'Total_Trades': 'sum',
        'Total_Quantity': 'sum',
        'Total_Profit': 'sum',
        'Average_Profit': 'mean',
        'Average_Profit_Pct': 'mean',
        'Max_Profit': 'max',
        'Max_Profit_Pct': 'max',
        'Min_Profit': 'min',
        'Min_Profit_Pct': 'min',
        'Average_Trade_Duration': 'mean'
    }).reset_index()

def main():
    st.title("Trading Performance Dashboard")
    
    # Load API keys for all strategies
    api_keys = load_api_keys()
    apis = {
        'Strategy 1': REST(api_keys['strat_1']['ALPACA_API_KEY'], 
                         api_keys['strat_1']['ALPACA_SECRET_KEY'], 
                         api_keys['strat_1']['ALPACA_BASE_URL']),
        'Strategy 2': REST(api_keys['strat_2']['ALPACA_API_KEY'], 
                         api_keys['strat_2']['ALPACA_SECRET_KEY'], 
                         api_keys['strat_2']['ALPACA_BASE_URL']),
        'Strategy 3': REST(api_keys['strat_3']['ALPACA_API_KEY'], 
                         api_keys['strat_3']['ALPACA_SECRET_KEY'], 
                         api_keys['strat_3']['ALPACA_BASE_URL'])
    }
    
    # Add strategy selector
    selected_strategy = st.sidebar.radio(
        "Select Strategy",
        ["All Strategies", "Strategy 1", "Strategy 2", "Strategy 3"]
    )
    
    # Check if we should refresh data
    if should_refresh() or 'data' not in st.session_state:
        with st.spinner('Refreshing trading data...'):
            all_data = {}
            
            for strategy_name, api in apis.items():
                executed_trades = get_executed_trades(api)
                df_trades = pd.DataFrame(executed_trades)
                matched_trades = match_buy_and_sell_trades(df_trades)
                trade_stats = calculate_trade_stats(matched_trades, api)
                stock_summary = calculate_stock_summary(matched_trades)
                daily_summary, daily_performers = calculate_daily_summary(matched_trades)
                current_holdings = get_current_holdings(api)
                
                all_data[strategy_name] = {
                    'trade_stats': trade_stats,
                    'stock_summary': stock_summary,
                    'daily_summary': daily_summary,
                    'daily_performers': daily_performers,
                    'current_holdings': current_holdings,
                    'matched_trades': matched_trades
                }
            
            # Store in session state
            st.session_state.data = all_data
            st.session_state.last_refresh = datetime.now(pytz.timezone('US/Eastern'))
    
    # Display last refresh time
    st.caption(f"Last refreshed: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S ET')}")
    
   # Calculate and display metrics
    if selected_strategy == "All Strategies":
        combined_stats = combine_strategy_stats([data['trade_stats'] 
                                             for data in st.session_state.data.values()])
        combined_trailing = combine_trailing_stats([
            calculate_trailing_stats(
                data['matched_trades'],
                apis[strategy].list_positions()
            ) for strategy, data in st.session_state.data.items()
        ])
        create_metrics_cards(combined_stats, combined_trailing)
    else:
        trailing_stats = calculate_trailing_stats(
            st.session_state.data[selected_strategy]['matched_trades'],
            apis[selected_strategy].list_positions()
        )
        create_metrics_cards(
            st.session_state.data[selected_strategy]['trade_stats'],
            trailing_stats
        )
    
    # Update the tabs to include All Trades
    tab1, tab2, tab3, tab4 = st.tabs(["Daily Performance", "Stock Analysis", "Current Holdings", "All Trades"])
    
    with tab1:
        if selected_strategy == "All Strategies":
            # Combine daily summaries
            combined_daily = combine_daily_summaries([data['daily_summary'] 
                                                   for data in st.session_state.data.values()])
            plot_daily_performance(combined_daily)
        else:
            plot_daily_performance(st.session_state.data[selected_strategy]['daily_summary'])
        
    with tab2:
        if selected_strategy == "All Strategies":
            combined_stock = combine_stock_summaries([data['stock_summary'] 
                                                   for data in st.session_state.data.values()])
            plot_stock_performance(combined_stock)
        else:
            plot_stock_performance(st.session_state.data[selected_strategy]['stock_summary'])
        
    with tab3:
        if selected_strategy == "All Strategies":
            for strategy in apis.keys():
                st.subheader(f"{strategy} Holdings")
                st.dataframe(st.session_state.data[strategy]['current_holdings'])
        else:
            st.dataframe(st.session_state.data[selected_strategy]['current_holdings'])
    
    with tab4:
        st.subheader("All Trades")
        
        if selected_strategy == "All Strategies":
            # Combine matched trades from all strategies
            all_trades = []
            for strategy_name, data in st.session_state.data.items():
                strategy_trades = data['matched_trades'].copy()
                strategy_trades['Strategy'] = strategy_name
                all_trades.append(strategy_trades)
            
            trades_df = pd.concat(all_trades, ignore_index=True)
        else:
            trades_df = st.session_state.data[selected_strategy]['matched_trades'].copy()
            trades_df['Strategy'] = selected_strategy
        
        # Add filters
        col1, col2, col3 = st.columns(3)
        with col1:
            # Filter by symbol
            symbols = sorted(trades_df['Symbol'].unique())
            selected_symbol = st.selectbox('Filter by Symbol', ['All'] + list(symbols))
        
        with col2:
            # Filter by date range
            min_date = pd.to_datetime(trades_df['Buy Filled At']).min().date()
            max_date = pd.to_datetime(trades_df['Sell Filled At']).max().date()
            date_range = st.date_input(
                'Select Date Range',
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        
        with col3:
            # Filter by profit/loss
            pnl_filter = st.selectbox('Filter by P&L', ['All', 'Profitable', 'Loss'])
        
        # Apply filters
        filtered_df = trades_df.copy()
        
        # Symbol filter
        if selected_symbol != 'All':
            filtered_df = filtered_df[filtered_df['Symbol'] == selected_symbol]
        
        # Date filter
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (pd.to_datetime(filtered_df['Buy Filled At']).dt.date >= start_date) &
                (pd.to_datetime(filtered_df['Sell Filled At']).dt.date <= end_date)
            ]
        
        # P&L filter
        if pnl_filter == 'Profitable':
            filtered_df = filtered_df[filtered_df['Profit'] > 0]
        elif pnl_filter == 'Loss':
            filtered_df = filtered_df[filtered_df['Profit'] <= 0]
        
        # Format the dataframe for display
        display_df = filtered_df.copy()
        
        # Convert timestamps to EST format
        display_df['Buy Filled At'] = display_df['Buy Filled At'].apply(
            lambda x: x.strftime('%Y-%m-%d %H:%M:%S %Z') if pd.notnull(x) else ''
        )
        display_df['Sell Filled At'] = display_df['Sell Filled At'].apply(
            lambda x: x.strftime('%Y-%m-%d %H:%M:%S %Z') if pd.notnull(x) else ''
        )
        # Format numeric columns
        display_df['Buy Price'] = display_df['Buy Price'].round(2)
        display_df['Sell Price'] = display_df['Sell Price'].round(2)
        display_df['Profit'] = display_df['Profit'].round(2)
        display_df['Trade Duration (mins)'] = display_df['Trade Duration (mins)'].round(2)
        
        # Reorder columns for better readability
        columns_order = [
            'Strategy',
            'Symbol',
            'Buy Filled At',
            'Sell Filled At',
            'Buy Price',
            'Sell Price',
            'Quantity',
            'Profit',
            'Trade Duration (mins)'
        ]
        
        display_df = display_df[columns_order]
        
        # Show the filtered trades
        st.dataframe(
            display_df,
            hide_index=True,
            column_config={
                'Profit': st.column_config.NumberColumn(
                    'Profit',
                    format="$%.2f",
                    help="Profit/Loss from the trade"
                ),
                'Buy Price': st.column_config.NumberColumn(
                    'Buy Price',
                    format="$%.2f"
                ),
                'Sell Price': st.column_config.NumberColumn(
                    'Sell Price',
                    format="$%.2f"
                )
            }
        )
        
        # Show summary statistics for filtered trades
        st.subheader("Filtered Trades Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Number of Trades", len(filtered_df))
        with col2:
            st.metric("Total Profit", f"${filtered_df['Profit'].sum():.2f}")
        with col3:
            st.metric("Average Profit", f"${filtered_df['Profit'].mean():.2f}")
        with col4:
            win_rate = (filtered_df['Profit'] > 0).mean() * 100
            st.metric("Win Rate", f"{win_rate:.2f}%")

    # Force refresh if needed
    if should_refresh():
        time.sleep(3600)
        st.experimental_rerun()

# Helper function to combine stats from multiple strategies
def combine_strategy_stats(stats_list):
    """Combine statistics from multiple strategies with default values"""
    # Set default values if keys don't exist
    default_stats = {
        'Portfolio Value': 0,
        'Cash Balance': 0,
        'Total Trades': 0,
        'Total Profit': 0,
        'Average Profit per Trade': 0,
        'Win Rate %': 0
    }
    
    # Create a list of stats with defaults filled in
    processed_stats = []
    for s in stats_list:
        # Create a new dict with defaults
        stat_with_defaults = default_stats.copy()
        # Update with actual values
        stat_with_defaults.update(s)
        processed_stats.append(stat_with_defaults)
    
    total_initial_capital = sum(100000 for s in processed_stats)
    combined = {
        'Portfolio Value': sum(s['Portfolio Value'] for s in processed_stats),
        'Portfolio Return': round((sum(s['Portfolio Value'] for s in processed_stats) - total_initial_capital) / total_initial_capital * 100, 2),
        'Cash Balance': sum(s['Cash Balance'] for s in processed_stats),
        'Total Trades': sum(s['Total Trades'] for s in processed_stats),
        'Total Profit': sum(s['Total Profit'] for s in processed_stats),
        'Average Profit per Trade': statistics.mean(s['Average Profit per Trade'] for s in processed_stats),
        'Win Rate %': round(statistics.mean(s['Win Rate %'] for s in processed_stats), 2)
    }
    return combined

def combine_trailing_stats(stats_list):
    """Combine trailing statistics from multiple strategies"""
    combined = {
        'Return': round(sum(s['Return'] for s in stats_list), 2),
        'Win Rate': round(statistics.mean(s['Win Rate'] for s in stats_list), 2),
        'Realized Profit': round(sum(s['Realized Profit'] for s in stats_list), 2),
        'Unrealized Profit': round(sum(s['Unrealized Profit'] for s in stats_list), 2)
    }
    return combined


def calculate_trailing_stats(matched_trades, current_holdings, initial_balance=100000):
    seven_days_ago = pd.Timestamp.now(tz=pytz.timezone('US/Eastern')) - pd.Timedelta(days=7)
    recent_trades = matched_trades[matched_trades['Sell Filled At'] >= seven_days_ago]
    
    if recent_trades.empty:
        return {
            'Return': 0,
            'Win Rate': 0,
            'Realized Profit': 0,
            'Unrealized Profit': sum(float(pos.unrealized_pl) for pos in current_holdings)
        }
    
    realized_profit = recent_trades['Profit'].sum()
    win_rate = (len(recent_trades[recent_trades['Profit'] > 0]) / len(recent_trades)) * 100
    unrealized_profit = sum(float(pos.unrealized_pl) for pos in current_holdings)
    
    return {
        'Return': round((realized_profit + unrealized_profit) / initial_balance * 100, 2),
        'Win Rate': round(win_rate, 2),
        'Realized Profit': round(realized_profit, 2),
        'Unrealized Profit': round(unrealized_profit, 2)
    }


if __name__ == "__main__":
    main()
