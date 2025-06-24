#!/usr/bin/env python3
"""
Streamlit Dashboard for Option Chain Data
Real-time visualization of data stored in PostgreSQL
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from database import engine
from sqlalchemy import text
from datetime import datetime, timedelta
import time
import numpy as np
from streamlit_autorefresh import st_autorefresh
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Option Chain Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .data-table {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chart-container {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .stAlert {
        border-radius: 0.5rem;
    }
    .stButton > button {
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .stSelectbox > div > div {
        border-radius: 0.5rem;
    }
    /* Make data table cells wider and increase font size for large numbers */
    .stDataFrame tbody td {
        min-width: 90px;
        max-width: 200px;
        padding: 10px 8px !important;
        font-size: 1.1rem !important;
        white-space: nowrap;
    }
    .stDataFrame thead th {
        min-width: 90px;
        max-width: 200px;
        padding: 10px 8px !important;
        font-size: 1.1rem !important;
        white-space: nowrap;
    }
    /* Top-right controls margin for sidebar */
    .top-right-controls {
        margin-top: 1.5rem;
        margin-right: 2.5rem;
        display: flex;
        flex-direction: row;
        gap: 1.5rem;
        align-items: center;
        justify-content: flex-end;
    }
    @media (max-width: 900px) {
        .top-right-controls {
            margin-right: 1rem;
            margin-top: 1rem;
            flex-direction: column;
            gap: 0.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_available_symbols():
    """Get list of available symbols from database"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT schema_name 
                FROM information_schema.schemata 
                WHERE schema_name LIKE 'option_chain%'
                ORDER BY schema_name
            """))
            schemas = [row[0] for row in result]
            symbols = [schema.replace('option_chain_', '').upper() for schema in schemas if schema != 'option_chain']
            return symbols
    except Exception as e:
        st.error(f"Error getting symbols: {str(e)}")
        return []

@st.cache_data(ttl=60)
def get_available_expiries(symbol):
    """Get list of available expiries for a symbol"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'option_chain_{symbol.lower()}'
                ORDER BY table_name
            """))
            tables = [row[0] for row in result]
            expiries = []
            for table in tables:
                expiry_part = table.replace(f"{symbol.lower()}_", "")
                expiry_date = expiry_part.replace("_", " ")
                expiries.append(expiry_date)
            return expiries
    except Exception as e:
        st.error(f"Error getting expiries: {str(e)}")
        return []

@st.cache_data(ttl=60)
def get_available_dates(symbol, expiry_date=None):
    """Get list of available trading dates from database"""
    try:
        if expiry_date:
            table_name = f"{symbol}_{expiry_date.replace(' ', '_').replace('-', '_')}"
            query = f"""
            SELECT DISTINCT DATE(fetch_time) as trading_date
            FROM option_chain_{symbol.lower()}.{table_name}
            ORDER BY trading_date DESC
            """
        else:
            # Get data from all tables for the symbol
            query = f"""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'option_chain_{symbol.lower()}'
            ORDER BY table_name;
            """
            
            with engine.connect() as conn:
                result = conn.execute(text(query))
                tables = [row[0] for row in result]
            
            if not tables:
                return []
            
            # Union all tables to get dates
            union_queries = []
            for table in tables:
                union_queries.append(f"SELECT DISTINCT DATE(fetch_time) as trading_date FROM option_chain_{symbol.lower()}.{table}")
            
            query = " UNION ".join(union_queries) + " ORDER BY trading_date DESC"
        
        with engine.connect() as conn:
            result = conn.execute(text(query))
            dates = [row[0] for row in result]
            return dates
    except Exception as e:
        st.error(f"Error getting dates: {str(e)}")
        return []

@st.cache_data(ttl=30)  # Cache for 30 seconds for real-time updates
def get_data_by_timeframe(symbol, expiry_date=None, selected_date=None, timeframe_minutes=1):
    """Get data filtered by date and time interval"""
    try:
        if expiry_date:
            table_name = f"{symbol}_{expiry_date.replace(' ', '_').replace('-', '_')}"
            query = f"""
            SELECT * FROM option_chain_{symbol.lower()}.{table_name}
            WHERE DATE(fetch_time) = '{selected_date}'
            ORDER BY fetch_time ASC, timestamp ASC
            """
        else:
            # Get data from all tables for the symbol
            query = f"""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'option_chain_{symbol.lower()}'
            ORDER BY table_name;
            """
            
            with engine.connect() as conn:
                result = conn.execute(text(query))
                tables = [row[0] for row in result]
            
            if not tables:
                return pd.DataFrame()
            
            # Union all tables
            union_queries = []
            for table in tables:
                union_queries.append(f"SELECT * FROM option_chain_{symbol.lower()}.{table}")
            
            query = " UNION ALL ".join(union_queries) + f" WHERE DATE(fetch_time) = '{selected_date}' ORDER BY fetch_time ASC, timestamp ASC"
        
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        if not df.empty:
            # Convert fetch_time to datetime if it's not already
            df['fetch_time'] = pd.to_datetime(df['fetch_time'])
            
            # Apply time interval filtering for display purposes only
            if timeframe_minutes > 1:
                # Create a copy for display intervals, but keep original fetch_time for current price
                df_display = df.copy()
                df_display['time_rounded'] = df_display['fetch_time'].dt.ceil(f'{timeframe_minutes}min')
                
                # For each interval and strike, get the last available data point
                df_display = df_display.groupby(['time_rounded', 'Strike Price']).last().reset_index()
                
                # Use rounded time for display, but keep original fetch_time for current price calculation
                df_display['display_time'] = df_display['time_rounded']
                df_display = df_display.drop(columns=['time_rounded'])
                
                # Return the display dataframe, but ensure current price comes from original data
                return df_display.sort_values(['display_time', 'Strike Price']).reset_index(drop=True)
        
        return df.sort_values(['fetch_time', 'Strike Price']).reset_index(drop=True)
    except Exception as e:
        st.error(f"Error getting data: {str(e)}")
        return pd.DataFrame()

def get_dashboard_stats():
    """Get overall dashboard statistics"""
    try:
        with engine.connect() as conn:
            # Total records
            result = conn.execute(text("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT table_schema) as total_symbols,
                    COUNT(DISTINCT table_name) as total_tables
                FROM information_schema.tables 
                WHERE table_schema LIKE 'option_chain%'
            """))
            stats = result.fetchone()
            
            # Dynamically get all option_chain tables
            result = conn.execute(text("""
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_schema LIKE 'option_chain%'
            """))
            tables = [(row[0], row[1]) for row in result]
            
            union_queries = [
                f'SELECT fetch_time FROM "{schema}"."{table}"' for schema, table in tables
            ]
            latest_time = None
            if union_queries:
                union_sql = " UNION ALL ".join(union_queries)
                full_query = f'SELECT MAX(fetch_time) as latest_time FROM ({union_sql}) as all_times'
                result = conn.execute(text(full_query))
                latest_time = result.fetchone()[0]
            
            return {
                'total_records': stats[0] if stats[0] else 0,
                'total_symbols': stats[1] if stats[1] else 0,
                'total_tables': stats[2] if stats[2] else 0,
                'latest_time': latest_time if latest_time else None
            }
    except Exception as e:
        return {
            'total_records': 0,
            'total_symbols': 0,
            'total_tables': 0,
            'latest_time': None
        }

def create_spot_price_chart(df, selected_date):
    """Create spot price trend chart for the selected date only"""
    if df.empty:
        return go.Figure()
    
    # Filter data for the selected date only
    df['fetch_time'] = pd.to_datetime(df['fetch_time'])
    df_filtered = df[df['fetch_time'].dt.date == pd.to_datetime(selected_date).date()]
    
    if df_filtered.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    # Group by fetch_time and get latest spot price for the selected date
    spot_data = df_filtered.groupby('fetch_time')['Spot Price'].last().reset_index()
    
    # Calculate y-axis range with margin
    min_spot = spot_data['Spot Price'].min()
    max_spot = spot_data['Spot Price'].max()
    margin = (max_spot - min_spot) * 0.1 if max_spot > min_spot else 10
    y_min = min_spot - margin
    y_max = max_spot + margin
    
    fig.add_trace(go.Scatter(
        x=spot_data['fetch_time'],
        y=spot_data['Spot Price'],
        mode='lines+markers',
        name='Spot Price',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8, color='#1f77b4'),
        fill='tonexty',
        fillcolor='rgba(31, 119, 180, 0.1)'
    ))
    
    fig.update_layout(
        title=f'📈 Spot Price Trend - {selected_date}',
        xaxis_title='Time',
        yaxis_title='Spot Price',
        height=400,
        showlegend=True,
        template='plotly_white',
        hovermode='x unified',
        yaxis=dict(range=[y_min, y_max])
    )
    
    return fig

def create_oi_chart(df, selected_date):
    """Create OI comparison chart for the selected date only"""
    if df.empty:
        return go.Figure()
    
    # Filter data for the selected date only
    df['fetch_time'] = pd.to_datetime(df['fetch_time'])
    df_filtered = df[df['fetch_time'].dt.date == pd.to_datetime(selected_date).date()]
    
    if df_filtered.empty:
        return go.Figure()
    
    # Get latest data for each strike for the selected date
    latest_data = df_filtered.groupby('Strike Price').last().reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=latest_data['Strike Price'],
        y=latest_data['CE OI'],
        name='CE OI',
        marker_color='#2ca02c',
        opacity=0.8
    ))
    
    fig.add_trace(go.Bar(
        x=latest_data['Strike Price'],
        y=latest_data['PE OI'],
        name='PE OI',
        marker_color='#d62728',
        opacity=0.8
    ))
    
    fig.update_layout(
        title=f'📊 Open Interest by Strike Price - {selected_date}',
        xaxis_title='Strike Price',
        yaxis_title='Open Interest',
        height=400,
        barmode='group',
        template='plotly_white',
        showlegend=True
    )
    
    return fig

def create_volume_chart(df, selected_date):
    """Create volume comparison chart for the selected date only"""
    if df.empty:
        return go.Figure()
    
    # Filter data for the selected date only
    df['fetch_time'] = pd.to_datetime(df['fetch_time'])
    df_filtered = df[df['fetch_time'].dt.date == pd.to_datetime(selected_date).date()]
    
    if df_filtered.empty:
        return go.Figure()
    
    # Get latest data for each strike for the selected date
    latest_data = df_filtered.groupby('Strike Price').last().reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=latest_data['Strike Price'],
        y=latest_data['CE Volume'],
        name='CE Volume',
        marker_color='#17a2b8',
        opacity=0.8
    ))
    
    fig.add_trace(go.Bar(
        x=latest_data['Strike Price'],
        y=latest_data['PE Volume'],
        name='PE Volume',
        marker_color='#fd7e14',
        opacity=0.8
    ))
    
    fig.update_layout(
        title=f'📈 Volume by Strike Price - {selected_date}',
        xaxis_title='Strike Price',
        yaxis_title='Volume',
        height=400,
        barmode='group',
        template='plotly_white',
        showlegend=True
    )
    
    return fig

def create_iv_chart(df, selected_date):
    """Create Implied Volatility chart for the selected date only"""
    if df.empty:
        return go.Figure()
    
    # Filter data for the selected date only
    df['fetch_time'] = pd.to_datetime(df['fetch_time'])
    df_filtered = df[df['fetch_time'].dt.date == pd.to_datetime(selected_date).date()]
    
    if df_filtered.empty:
        return go.Figure()
    
    # Get latest data for each strike for the selected date
    latest_data = df_filtered.groupby('Strike Price').last().reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=latest_data['Strike Price'],
        y=latest_data['CE IV'],
        mode='lines+markers',
        name='CE IV',
        line=dict(color='#2ca02c', width=3),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=latest_data['Strike Price'],
        y=latest_data['PE IV'],
        mode='lines+markers',
        name='PE IV',
        line=dict(color='#d62728', width=3),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=f'📊 Implied Volatility by Strike Price - {selected_date}',
        xaxis_title='Strike Price',
        yaxis_title='Implied Volatility (%)',
        height=400,
        template='plotly_white',
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def create_greeks_chart(df, greek_type='Delta', selected_date=None):
    """Create Greeks comparison chart for the selected date only"""
    if df.empty:
        return go.Figure()
    
    # Filter data for the selected date only
    df['fetch_time'] = pd.to_datetime(df['fetch_time'])
    df_filtered = df[df['fetch_time'].dt.date == pd.to_datetime(selected_date).date()]
    
    if df_filtered.empty:
        return go.Figure()
    
    # Get latest data for each strike for the selected date
    latest_data = df_filtered.groupby('Strike Price').last().reset_index()
    
    fig = go.Figure()
    
    ce_greek_col = f'CE {greek_type}'
    pe_greek_col = f'PE {greek_type}'
    
    if ce_greek_col in latest_data.columns and pe_greek_col in latest_data.columns:
        fig.add_trace(go.Scatter(
            x=latest_data['Strike Price'],
            y=latest_data[ce_greek_col],
            mode='lines+markers',
            name=f'CE {greek_type}',
            line=dict(color='#2ca02c', width=3),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=latest_data['Strike Price'],
            y=latest_data[pe_greek_col],
            mode='lines+markers',
            name=f'PE {greek_type}',
            line=dict(color='#d62728', width=3),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title=f'📊 {greek_type} by Strike Price - {selected_date}',
        xaxis_title='Strike Price',
        yaxis_title=f'{greek_type}',
        height=400,
        template='plotly_white',
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def create_pcr_chart(df, selected_date):
    """Create Put-Call Ratio chart for the selected date only"""
    if df.empty:
        return go.Figure()
    
    # Filter data for the selected date only
    df['fetch_time'] = pd.to_datetime(df['fetch_time'])
    df_filtered = df[df['fetch_time'].dt.date == pd.to_datetime(selected_date).date()]
    
    if df_filtered.empty:
        return go.Figure()
    
    # Group by fetch_time and calculate PCR for the selected date
    pcr_data = df_filtered.groupby('fetch_time').agg({
        'PE OI': 'sum',
        'CE OI': 'sum'
    }).reset_index()
    
    pcr_data['PCR'] = pcr_data['PE OI'] / pcr_data['CE OI']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=pcr_data['fetch_time'],
        y=pcr_data['PCR'],
        mode='lines+markers',
        name='Put-Call Ratio',
        line=dict(color='#9467bd', width=3),
        marker=dict(size=8)
    ))
    
    # Add horizontal line at PCR = 1
    fig.add_hline(y=1, line_dash="dash", line_color="red", 
                  annotation_text="PCR = 1 (Neutral)")
    
    fig.update_layout(
        title=f'📊 Put-Call Ratio Over Time - {selected_date}',
        xaxis_title='Time',
        yaxis_title='Put-Call Ratio',
        height=400,
        template='plotly_white',
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def create_heatmap(df, selected_date):
    """Create heatmap of OI data for the selected date only"""
    if df.empty:
        return go.Figure()
    
    # Filter data for the selected date only
    df['fetch_time'] = pd.to_datetime(df['fetch_time'])
    df_filtered = df[df['fetch_time'].dt.date == pd.to_datetime(selected_date).date()]
    
    if df_filtered.empty:
        return go.Figure()
    
    # Get latest data for each strike for the selected date
    latest_data = df_filtered.groupby('Strike Price').last().reset_index()
    
    # Create heatmap data
    strikes = latest_data['Strike Price'].tolist()
    ce_oi = latest_data['CE OI'].tolist()
    pe_oi = latest_data['PE OI'].tolist()
    
    fig = go.Figure(data=go.Heatmap(
        z=[ce_oi, pe_oi],
        x=strikes,
        y=['CE OI', 'PE OI'],
        colorscale='Viridis',
        showscale=True
    ))
    
    fig.update_layout(
        title=f'🔥 Open Interest Heatmap - {selected_date}',
        xaxis_title='Strike Price',
        yaxis_title='Option Type',
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_stacked_bar_chart(pivot_df, spot_price_df=None, title="", orientation="v", show_latest_price=False):
    """Create a stacked bar chart from the pivot table data by time interval, supporting both vertical and horizontal orientation."""
    if pivot_df.empty:
        return go.Figure()

    fig = go.Figure()
    # The first two columns are 'Instrument' and 'Strike Price', the rest are time intervals
    time_cols = pivot_df.columns[2:]

    if orientation == "v":
        for time_col in time_cols:
            bar_text = pivot_df[time_col].apply(lambda x: f'{x:,.0f}' if x > 0 else '')
            fig.add_trace(go.Bar(
                x=pivot_df['Strike Price'],
                y=pivot_df[time_col],
                name=time_col,
                text=bar_text,
                textposition='inside',
                orientation='v',
            ))
        fig.update_layout(
            barmode='stack',
            title_text=f'<b>{title} Distribution Over Time</b>',
            xaxis_title='Strike Price',
            yaxis_title=f'Total {title}',
            legend_title='Time Interval',
            height=500,
            template='plotly_white',
        )
    else:
        for time_col in time_cols:
            bar_text = pivot_df[time_col].apply(lambda x: f'{x:,.0f}' if x > 0 else '')
            fig.add_trace(go.Bar(
                y=pivot_df['Strike Price'],
                x=pivot_df[time_col],
                name=time_col,
                text=bar_text,
                textposition='inside',
                orientation='h',
            ))
        fig.update_layout(
            barmode='stack',
            title_text=f'<b>{title} Distribution Over Time</b>',
            yaxis_title='Strike Price',
            xaxis_title=f'Total {title}',
            legend_title='Time Interval',
            height=500,
            template='plotly_white',
        )
    # Only set textangle for Bar traces to avoid ValueError on Scatter
    for trace in fig.data:
        if isinstance(trace, go.Bar):
            trace.textangle = 0
    return fig

def create_pivot_table(df, value_col='CE OI'):
    """Create a pivot table of a selected metric over time for each strike."""
    if df.empty or value_col not in df.columns:
        st.warning("Not enough data to create a pivot table for the selected timeframe.")
        return pd.DataFrame()

    try:
        df['fetch_time'] = pd.to_datetime(df['fetch_time'])
        df['time_str'] = df['fetch_time'].dt.strftime('%H:%M')
        
        pivot_df = df.pivot_table(
            index='Strike Price', 
            columns='time_str', 
            values=value_col,
            aggfunc='last'
        )
        
        pivot_df = pivot_df.reset_index()

        if 'Symbol' in df.columns:
            symbol = df['Symbol'].iloc[0]
            pivot_df.insert(0, 'Instrument', symbol)
        
        pivot_df = pivot_df.fillna(0)
        return pivot_df
    except Exception as e:
        st.error(f"Error creating pivot table: {e}")
        return pd.DataFrame()

def calculate_analytics(df, selected_date, original_df=None):
    """Calculate various analytics from the most recent data of the selected date"""
    if df.empty:
        return {}
    
    # Use original_df if provided (for timeframe filtering), otherwise use df
    data_source = original_df if original_df is not None else df
    
    # Filter data for the selected date only
    data_source['fetch_time'] = pd.to_datetime(data_source['fetch_time'])
    df_filtered = data_source[data_source['fetch_time'].dt.date == pd.to_datetime(selected_date).date()]
    
    if df_filtered.empty:
        return {}
    
    analytics = {}
    
    # Get the most recent data for each strike (latest fetch_time) - ALWAYS from original data
    latest_fetch_time = df_filtered['fetch_time'].max()
    latest_data = df_filtered[df_filtered['fetch_time'] == latest_fetch_time]
    
    # Basic stats from most recent data
    analytics['total_strikes'] = len(latest_data)
    analytics['avg_spot_price'] = latest_data['Spot Price'].mean()
    analytics['current_spot'] = latest_data['Spot Price'].iloc[0] if len(latest_data) > 0 else 0
    
    # OI Analysis from most recent data
    analytics['total_ce_oi'] = latest_data['CE OI'].sum()
    analytics['total_pe_oi'] = latest_data['PE OI'].sum()
    analytics['pcr'] = analytics['total_pe_oi'] / analytics['total_ce_oi'] if analytics['total_ce_oi'] > 0 else 0
    
    # Volume Analysis from most recent data
    analytics['total_ce_volume'] = latest_data['CE Volume'].sum()
    analytics['total_pe_volume'] = latest_data['PE Volume'].sum()
    
    # IV Analysis from most recent data
    analytics['avg_ce_iv'] = latest_data['CE IV'].mean()
    analytics['avg_pe_iv'] = latest_data['PE IV'].mean()
    
    # Find ATM strike from most recent data
    if 'ATM Strike' in latest_data.columns:
        atm_strike = latest_data['ATM Strike'].iloc[0]
        atm_data = latest_data[latest_data['Strike Price'] == atm_strike]
        if not atm_data.empty:
            analytics['atm_ce_oi'] = atm_data['CE OI'].iloc[0]
            analytics['atm_pe_oi'] = atm_data['PE OI'].iloc[0]
            analytics['atm_ce_iv'] = atm_data['CE IV'].iloc[0]
            analytics['atm_pe_iv'] = atm_data['PE IV'].iloc[0]
    
    # Add timestamp of the most recent data
    analytics['latest_update'] = latest_fetch_time
    
    return analytics

def create_grouped_bar_with_price(df, metric_ce, metric_pe, spot_price_col='Spot Price', title=""):
    """Create a grouped bar chart for CE and PE for each strike price, with spot price overlay and opening price reference line."""
    if df.empty or metric_ce not in df.columns or metric_pe not in df.columns or spot_price_col not in df.columns:
        return go.Figure()

    # Get latest data for each strike
    latest_data = df.groupby('Strike Price').last().reset_index()
    strike_prices = latest_data['Strike Price']
    ce_values = latest_data[metric_ce]
    pe_values = latest_data[metric_pe]
    spot_prices = latest_data[spot_price_col]

    fig = go.Figure()
    # Add CE bars
    fig.add_trace(go.Bar(
        x=strike_prices,
        y=ce_values,
        name='Call (CE)',
        marker_color='#1f77b4',
        offsetgroup=0,
        text=[f'{v:,.0f}' if v > 0 else '' for v in ce_values],
        textposition='outside',
    ))
    # Add PE bars
    fig.add_trace(go.Bar(
        x=strike_prices,
        y=pe_values,
        name='Put (PE)',
        marker_color='#ff7f0e',
        offsetgroup=1,
        text=[f'{v:,.0f}' if v > 0 else '' for v in pe_values],
        textposition='outside',
    ))
    # Overlay spot price as a line (right y-axis)
    fig.add_trace(go.Scatter(
        x=strike_prices,
        y=spot_prices,
        mode='lines+markers',
        name='Spot Price',
        yaxis='y2',
        line=dict(color='black', width=3, dash='dot'),
        marker=dict(size=7, color='black'),
    ))
    # Add horizontal reference line at opening price
    opening_price = spot_prices.iloc[0] if not spot_prices.empty else None
    if opening_price is not None:
        fig.add_hline(y=opening_price, line_dash="dash", line_color="green", annotation_text="Opening Price", annotation_position="top left", yref="y2")
    # Layout
    fig.update_layout(
        barmode='group',
        bargap=0.35,
        title_text=f'<b>{title} (CE/PE Grouped) with Spot Price Overlay</b>',
        xaxis_title='Strike Price',
        yaxis_title=f'{metric_ce} / {metric_pe}',
        yaxis=dict(
            title=f'{metric_ce} / {metric_pe}',
            showgrid=True,
            zeroline=True,
            side='left',
            anchor='x',
            matches=None  # Ensure not linked
        ),
        yaxis2=dict(
            overlaying='y',
            side='right',
            showgrid=False,
            title='Spot Price',
            matches=None,  # Ensure not linked
            anchor=None
        ),
        legend_title='Legend',
        height=500,
        template='plotly_white',
    )
    return fig

def main():
    # Auto-refresh every 1 minute (60 seconds)
    st_autorefresh(interval=60 * 1000, key="dashboard_autorefresh")

    # Get available symbols
    symbols = get_available_symbols()
    if not symbols:
        st.warning("No data available. Please start the data collection first.")
        return

    # Unified top row: nav bar (left) and top-right controls (right)
    stats = get_dashboard_stats()
    top_nav_col, top_right_col = st.columns([7, 2])
    with top_nav_col:
        nav1, nav2, nav3, nav4 = st.columns([2, 2, 2, 2])
        # Symbol selection persistence
        if 'selected_symbol' not in st.session_state:
            st.session_state['selected_symbol'] = symbols[0] if symbols else None
        with nav1:
            selected_symbol = st.selectbox(
                "Select Symbol:",
                symbols,
                index=symbols.index(st.session_state['selected_symbol']) if st.session_state['selected_symbol'] in symbols else 0,
                key='topnav_symbol'
            )
            st.session_state['selected_symbol'] = selected_symbol
        # Expiry selection persistence
        with nav2:
            if selected_symbol:
                expiries = get_available_expiries(selected_symbol)
                if 'selected_expiry' not in st.session_state or st.session_state['selected_expiry'] not in (["All Expiries"] + expiries):
                    st.session_state['selected_expiry'] = "All Expiries"
                selected_expiry = st.selectbox(
                    "Select Expiry:",
                    ["All Expiries"] + expiries,
                    index=(["All Expiries"] + expiries).index(st.session_state['selected_expiry']) if st.session_state['selected_expiry'] in (["All Expiries"] + expiries) else 0,
                    key='topnav_expiry'
                )
                st.session_state['selected_expiry'] = selected_expiry
            else:
                selected_expiry = None
        # Date selection persistence
        with nav3:
            if selected_symbol:
                available_dates = get_available_dates(selected_symbol, None if selected_expiry == "All Expiries" else selected_expiry)
                if available_dates:
                    if 'selected_date' not in st.session_state or st.session_state['selected_date'] not in available_dates:
                        st.session_state['selected_date'] = available_dates[0]
                    selected_date = st.selectbox(
                        "Select Trading Date:",
                        available_dates,
                        index=available_dates.index(st.session_state['selected_date']) if st.session_state['selected_date'] in available_dates else 0,
                        key='topnav_date'
                    )
                    st.session_state['selected_date'] = selected_date
                else:
                    st.warning("No trading dates available")
                    return
            else:
                selected_date = None
        # Time interval selection persistence
        with nav4:
            timeframe_options = {
                "1 Minute": 1,
                "5 Minutes": 5,
                "15 Minutes": 15,
                "30 Minutes": 30,
                "1 Hour": 60
            }
            if 'selected_timeframe' not in st.session_state or st.session_state['selected_timeframe'] not in timeframe_options:
                st.session_state['selected_timeframe'] = "1 Minute"
            selected_timeframe = st.selectbox(
                "Select Time Interval:",
                list(timeframe_options.keys()),
                index=list(timeframe_options.keys()).index(st.session_state['selected_timeframe']) if st.session_state['selected_timeframe'] in timeframe_options else 0,
                key='topnav_timeframe'
            )
            st.session_state['selected_timeframe'] = selected_timeframe
            timeframe_minutes = timeframe_options[selected_timeframe]
    with top_right_col:
        col_refresh, col_update = st.columns([1, 2])
        with col_refresh:
            if st.button('Refresh Data', key='refresh_data_btn'):
                st.cache_data.clear()
                st.rerun()
        with col_update:
            st.markdown(f"<div style='text-align:left; font-size:1.1rem; min-width:110px; margin-top:0.3rem;'>🕐 <b>Latest Update</b> {stats['latest_time'].strftime('%H:%M:%S') if stats['latest_time'] else 'No data'}</div>", unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("📋 Dashboard Controls")
    
    # Toggle for dashboard header & metrics in sidebar
    if 'show_header_metrics' not in st.session_state:
        st.session_state['show_header_metrics'] = False
    st.sidebar.markdown('---')
    st.session_state['show_header_metrics'] = st.sidebar.checkbox(
        'Show Dashboard Header & Metrics',
        value=st.session_state['show_header_metrics'],
        key='sidebar_show_header_metrics'
    )
    
    # Only show header and metrics if toggle is ON
    if st.session_state['show_header_metrics']:
        st.markdown('<h1><span style="font-size:2.5rem;">📊</span> <span class="main-header">Option Chain Dashboard</span></h1>', unsafe_allow_html=True)
        # Metrics row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📈 Total Records", f"{get_dashboard_stats()['total_records']:,}")
        with col2:
            st.metric("📊 Total Symbols", get_dashboard_stats()['total_symbols'])
        with col3:
            st.metric("📋 Total Tables", get_dashboard_stats()['total_tables'])
    
    # Section visibility controls with checkboxes (like chart options)
    section_options = [
        "Market Analytics",
        "Market Charts",
        "Strike vs. Time Analysis",
        "Raw Data Tables",
        "Detailed Data Table",
        "Data Summary"
    ]
    # Initialize session state for section visibility
    for section in section_options:
        if f'show_{section}' not in st.session_state:
            st.session_state[f'show_{section}'] = (section == "Strike vs. Time Analysis")
    st.sidebar.markdown('---')
    st.sidebar.markdown('**Show/Hide Dashboard Sections:**')
    for section in section_options:
        key = f'show_{section}'
        st.session_state[key] = st.sidebar.checkbox(
            section,
            value=st.session_state[key],
            key=f'cb_{section}'
        )
    visible_sections = [section for section in section_options if st.session_state[f'show_{section}']]
    # Move chart options below section selection
    st.sidebar.title("📊 Chart Options")
    show_spot_chart = st.sidebar.checkbox("Spot Price Trend", value=True)
    show_oi_chart = st.sidebar.checkbox("Open Interest", value=True)
    show_volume_chart = st.sidebar.checkbox("Volume Analysis", value=True)
    show_iv_chart = st.sidebar.checkbox("Implied Volatility", value=True)
    show_greeks_chart = st.sidebar.checkbox("Greeks Analysis", value=True)
    show_pcr_chart = st.sidebar.checkbox("Put-Call Ratio", value=True)
    show_heatmap = st.sidebar.checkbox("OI Heatmap", value=True)
    
    # Greeks type selection
    if show_greeks_chart:
        greek_type = st.sidebar.selectbox(
            "Select Greek:",
            ["Delta", "Gamma", "Theta", "Vega"],
            index=0
        )
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    # Get data
    if selected_symbol and 'selected_date' in locals():
        expiry_filter = None if selected_expiry == "All Expiries" else selected_expiry
        
        # Get original data (1-minute intervals) for current price calculation
        original_df = get_data_by_timeframe(selected_symbol, expiry_filter, selected_date, 1)
        
        # Get timeframe-filtered data for display
        df = get_data_by_timeframe(selected_symbol, expiry_filter, selected_date, timeframe_minutes)
        
        if not df.empty:
            # Calculate analytics using original data for current prices
            analytics = calculate_analytics(df, selected_date, original_df)
            
            # Analytics section
            if "Market Analytics" in visible_sections:
                st.subheader(f"📊 Market Analytics - {selected_date} ({selected_timeframe})")
                
                # Show latest update time
                if 'latest_update' in analytics:
                    latest_time = analytics['latest_update'].strftime('%H:%M:%S') if analytics['latest_update'] else 'N/A'
                    st.markdown(f"**🕐 Latest Update: {latest_time}**")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Spot", f"{analytics.get('current_spot', 0):.2f}")
                    st.metric("Total CE OI", f"{analytics.get('total_ce_oi', 0):,}")
                with col2:
                    st.metric("Put-Call Ratio", f"{analytics.get('pcr', 0):.2f}")
                    st.metric("Total PE OI", f"{analytics.get('total_pe_oi', 0):,}")
                with col3:
                    st.metric("Avg CE IV", f"{analytics.get('avg_ce_iv', 0):.2f}%")
                    st.metric("Total CE Volume", f"{analytics.get('total_ce_volume', 0):,}")
                with col4:
                    st.metric("Avg PE IV", f"{analytics.get('avg_pe_iv', 0):.2f}%")
                    st.metric("Total PE Volume", f"{analytics.get('total_pe_volume', 0):,}")
            
            # Charts section
            if "Market Charts" in visible_sections:
                st.subheader("📈 Market Charts")
                # First row of charts
                if show_spot_chart or show_oi_chart:
                    col1, col2 = st.columns(2)
                    with col1:
                        if show_spot_chart:
                            spot_chart = create_spot_price_chart(df, selected_date)
                            st.plotly_chart(spot_chart, use_container_width=True)
                    with col2:
                        if show_oi_chart:
                            oi_chart = create_oi_chart(df, selected_date)
                            st.plotly_chart(oi_chart, use_container_width=True)
                # Second row of charts
                if show_volume_chart or show_iv_chart:
                    col1, col2 = st.columns(2)
                    with col1:
                        if show_volume_chart:
                            volume_chart = create_volume_chart(df, selected_date)
                            st.plotly_chart(volume_chart, use_container_width=True)
                    with col2:
                        if show_iv_chart:
                            iv_chart = create_iv_chart(df, selected_date)
                            st.plotly_chart(iv_chart, use_container_width=True)
                # Third row of charts
                if show_greeks_chart or show_pcr_chart:
                    col1, col2 = st.columns(2)
                    with col1:
                        if show_greeks_chart:
                            greeks_chart = create_greeks_chart(df, greek_type, selected_date)
                            st.plotly_chart(greeks_chart, use_container_width=True)
                    with col2:
                        if show_pcr_chart:
                            pcr_chart = create_pcr_chart(df, selected_date)
                            st.plotly_chart(pcr_chart, use_container_width=True)
                # Fourth row - heatmap
                if show_heatmap:
                    heatmap_chart = create_heatmap(df, selected_date)
                    st.plotly_chart(heatmap_chart, use_container_width=True)
            
            # Pivot Table Section
            if "Strike vs. Time Analysis" in visible_sections:
                st.subheader("📈 Strike vs. Time Analysis")
                col1, col2 = st.columns([1, 3])
                with col1:
                    pivot_metric = st.selectbox(
                        "Select Metric to Analyze:",
                        ['CE OI', 'PE OI', 'CE Volume', 'PE Volume', 'CE Chg in OI', 'PE Chg in OI', 'CE IV', 'PE IV', 'CE LTP', 'PE LTP'],
                        key='pivot_metric'
                    )
                    # Show/hide price chart checkbox
                    if 'show_price_chart' not in st.session_state:
                        st.session_state['show_price_chart'] = False
                    show_price_chart = st.checkbox('Price Chart', value=st.session_state['show_price_chart'], key='show_price_chart')
                    # Persistent chart orientation
                    if 'chart_orientation' not in st.session_state:
                        st.session_state['chart_orientation'] = "Vertical"
                    chart_orientation = st.radio(
                        "Chart Orientation:",
                        ["Vertical", "Horizontal"],
                        index=["Vertical", "Horizontal"].index(st.session_state['chart_orientation']),
                        key='chart_orientation'
                    )
                    # Persistent chart type
                    if 'strike_time_chart_type' not in st.session_state:
                        st.session_state['strike_time_chart_type'] = "Stacked by Time Interval"
                    strike_time_chart_type = st.radio(
                        "Chart Type:",
                        ["Stacked by Time Interval", "Strike vs. Time"],
                        index=["Stacked by Time Interval", "Strike vs. Time"].index(st.session_state['strike_time_chart_type']),
                        key='strike_time_chart_type'
                    )
                    # Persistent bar chart variation
                    if 'bar_chart_variation' not in st.session_state:
                        st.session_state['bar_chart_variation'] = "Single Stacked Bar Chart"
                    bar_chart_variation = st.radio(
                        "Select Bar Chart Variation:",
                        ["Single Stacked Bar Chart", "Double Stacked Bar Chart (CE/PE)"],
                        index=["Single Stacked Bar Chart", "Double Stacked Bar Chart (CE/PE)"].index(st.session_state['bar_chart_variation']),
                        key='bar_chart_variation'
                    )
                # --- MOVE PRICE CHART RENDERING HERE ---
                if show_price_chart and 'fetch_time' in df.columns and 'Spot Price' in df.columns:
                    price_df = df[['fetch_time', 'Spot Price']].drop_duplicates().sort_values('fetch_time')
                    if not price_df.empty:
                        price_fig = go.Figure()
                        price_fig.add_trace(go.Scatter(
                            x=price_df['fetch_time'],
                            y=price_df['Spot Price'],
                            mode='lines+markers',
                            name='Spot Price',
                            line=dict(color='#1f77b4', width=3),
                            marker=dict(size=7, color='#1f77b4'),
                        ))
                        price_fig.update_layout(
                            title='Spot Price Trend',
                            xaxis_title='Time',
                            yaxis_title='Spot Price',
                            height=350,
                            template='plotly_white',
                        )
                        st.plotly_chart(price_fig, use_container_width=True)
                # --- END PRICE CHART RENDERING ---
                if pivot_metric:
                    if bar_chart_variation == "Double Stacked Bar Chart (CE/PE)":
                        ce_metric = pivot_metric if pivot_metric.startswith('CE ') else 'CE ' + pivot_metric[3:]
                        pe_metric = pivot_metric if pivot_metric.startswith('PE ') else 'PE ' + pivot_metric[3:]
                        ce_pivot = create_pivot_table(df, value_col=ce_metric)
                        pe_pivot = create_pivot_table(df, value_col=pe_metric)
                        if not ce_pivot.empty and not pe_pivot.empty:
                            time_cols = ce_pivot.columns[2:]
                            fig = go.Figure()
                            palette = px.colors.qualitative.Plotly
                            if strike_time_chart_type == "Stacked by Time Interval":
                                # For each time interval, plot CE and PE as neighbors, each stacked by time
                                for i, time_col in enumerate(time_cols):
                                    if chart_orientation == "Vertical":
                                        fig.add_trace(go.Bar(
                                            x=ce_pivot['Strike Price'],
                                            y=ce_pivot[time_col],
                                            name=f"CE {time_col}",
                                            text=[f'{v:,.0f}' if v > 0 else '' for v in ce_pivot[time_col]],
                                            textposition='inside',
                                            orientation='v',
                                            marker_color=palette[i % len(palette)],
                                            offsetgroup='CE',
                                            legendgroup='CE',
                                            showlegend=True if i == 0 else False,
                                            customdata=np.stack([['CE']*len(ce_pivot)], axis=-1),
                                        ))
                                        fig.add_trace(go.Bar(
                                            x=pe_pivot['Strike Price'],
                                            y=pe_pivot[time_col],
                                            name=f"PE {time_col}",
                                            text=[f'{v:,.0f}' if v > 0 else '' for v in pe_pivot[time_col]],
                                            textposition='inside',
                                            orientation='v',
                                            marker_color=palette[i % len(palette)],
                                            offsetgroup='PE',
                                            legendgroup='PE',
                                            showlegend=True if i == 0 else False,
                                            customdata=np.stack([['PE']*len(pe_pivot)], axis=-1),
                                        ))
                                    else:
                                        fig.add_trace(go.Bar(
                                            y=ce_pivot['Strike Price'],
                                            x=ce_pivot[time_col],
                                            name=f"CE {time_col}",
                                            text=[f'{v:,.0f}' if v > 0 else '' for v in ce_pivot[time_col]],
                                            textposition='inside',
                                            orientation='h',
                                            marker_color=palette[i % len(palette)],
                                            offsetgroup='CE',
                                            legendgroup='CE',
                                        ))
                                        fig.add_trace(go.Bar(
                                            y=pe_pivot['Strike Price'],
                                            x=pe_pivot[time_col],
                                            name=f"PE {time_col}",
                                            text=[f'{v:,.0f}' if v > 0 else '' for v in pe_pivot[time_col]],
                                            textposition='inside',
                                            orientation='h',
                                            marker_color=palette[i % len(palette)],
                                            offsetgroup='PE',
                                            legendgroup='PE',
                                        ))
                                fig.update_layout(
                                    barmode='stack',
                                    bargap=0.25,
                                    title_text=f'<b>Double Stacked Bar Chart (CE/PE) for {pivot_metric[3:]}</b>',
                                    xaxis_title='Strike Price' if chart_orientation == "Vertical" else f'Total {pivot_metric}',
                                    yaxis_title=f'Total {pivot_metric}' if chart_orientation == "Vertical" else 'Strike Price',
                                    legend_title='Type & Time Interval',
                                    height=500,
                                    template='plotly_white',
                                )
                                for trace in fig.data:
                                    if isinstance(trace, go.Bar):
                                        trace.textangle = 0
                                st.plotly_chart(fig, use_container_width=True)
                                # Format only numeric columns
                                numeric_cols = ce_pivot.select_dtypes(include=[np.number]).columns
                                format_dict = {col: '{:,.2f}' for col in numeric_cols}
                                ce_styled = ce_pivot.style.format(format_dict)
                                st.dataframe(ce_styled, use_container_width=True)
                                pe_styled = pe_pivot.style.format(format_dict)
                                st.dataframe(pe_styled, use_container_width=True)
                            else:  # Strike vs. Time
                                # For each time interval, show two bars (CE, PE), each stacked by strike price
                                if 'fetch_time' in df.columns and 'Strike Price' in df.columns and pivot_metric in df.columns:
                                    df['time_str'] = pd.to_datetime(df['fetch_time']).dt.strftime('%H:%M')
                                    time_strs = sorted(df['time_str'].unique())
                                    strike_prices = sorted(df['Strike Price'].unique())
                                    for idx, strike in enumerate(strike_prices):
                                        ce_y_vals = []
                                        pe_y_vals = []
                                        for t in time_strs:
                                            ce_val = df[(df['Strike Price'] == strike) & (df['time_str'] == t)][ce_metric]
                                            pe_val = df[(df['Strike Price'] == strike) & (df['time_str'] == t)][pe_metric]
                                            ce_y_vals.append(ce_val.iloc[0] if not ce_val.empty else 0)
                                            pe_y_vals.append(pe_val.iloc[0] if not pe_val.empty else 0)
                                        if chart_orientation == "Vertical":
                                            fig.add_trace(go.Bar(
                                                x=time_strs,
                                                y=ce_y_vals,
                                                name=f"CE {strike:.2f}",
                                                text=[f'{v:,.0f}' if v > 0 else '' for v in ce_y_vals],
                                                textposition='inside',
                                                orientation='v',
                                                marker_color=palette[idx % len(palette)],
                                                offsetgroup='CE',
                                                legendgroup='CE',
                                            ))
                                            fig.add_trace(go.Bar(
                                                x=time_strs,
                                                y=pe_y_vals,
                                                name=f"PE {strike:.2f}",
                                                text=[f'{v:,.0f}' if v > 0 else '' for v in pe_y_vals],
                                                textposition='inside',
                                                orientation='v',
                                                marker_color=palette[idx % len(palette)],
                                                offsetgroup='PE',
                                                legendgroup='PE',
                                            ))
                                        else:
                                            fig.add_trace(go.Bar(
                                                y=time_strs,
                                                x=ce_y_vals,
                                                name=f"CE {strike:.2f}",
                                                text=[f'{v:,.0f}' if v > 0 else '' for v in ce_y_vals],
                                                textposition='inside',
                                                orientation='h',
                                                marker_color=palette[idx % len(palette)],
                                                offsetgroup='CE',
                                                legendgroup='CE',
                                            ))
                                            fig.add_trace(go.Bar(
                                                y=time_strs,
                                                x=pe_y_vals,
                                                name=f"PE {strike:.2f}",
                                                text=[f'{v:,.0f}' if v > 0 else '' for v in pe_y_vals],
                                                textposition='inside',
                                                orientation='h',
                                                marker_color=palette[idx % len(palette)],
                                                offsetgroup='PE',
                                                legendgroup='PE',
                                            ))
                                    fig.update_layout(
                                        barmode='stack',
                                        title_text=f'<b>Double Stacked Bar Chart (CE/PE) by Time Interval (Stacked by Strike)</b>',
                                        xaxis_title='Time' if chart_orientation == "Vertical" else f'{pivot_metric}',
                                        yaxis_title=f'{pivot_metric}' if chart_orientation == "Vertical" else 'Time',
                                        legend_title='Type & Strike Price',
                                        height=500,
                                        template='plotly_white',
                                    )
                                    for trace in fig.data:
                                        if isinstance(trace, go.Bar):
                                            trace.textangle = 0
                                    st.plotly_chart(fig, use_container_width=True)
                                # Format only numeric columns
                                numeric_cols = ce_pivot.select_dtypes(include=[np.number]).columns
                                format_dict = {col: '{:,.2f}' for col in numeric_cols}
                                ce_styled = ce_pivot.style.format(format_dict)
                                st.dataframe(ce_styled, use_container_width=True)
                                pe_styled = pe_pivot.style.format(format_dict)
                                st.dataframe(pe_styled, use_container_width=True)
                    elif strike_time_chart_type == "Stacked by Time Interval":
                        pivot_table_df = create_pivot_table(df, value_col=pivot_metric)
                        if not pivot_table_df.empty:
                            st.markdown(f"### Stacked Bar Chart: {pivot_metric}")
                            stacked_chart = create_stacked_bar_chart(
                                pivot_table_df,
                                spot_price_df=None,
                                title=pivot_metric,
                                orientation='h' if chart_orientation == "Horizontal" else 'v',
                                show_latest_price=False
                            )
                            st.plotly_chart(stacked_chart, use_container_width=True)
                            # Format only numeric columns
                            numeric_cols = pivot_table_df.select_dtypes(include=[np.number]).columns
                            format_dict = {col: '{:,.2f}' for col in numeric_cols}
                            styled_df = pivot_table_df.style.format(format_dict)
                            st.dataframe(styled_df, use_container_width=True)
                    else:
                        # Alternative chart: time on X, stacked by strike
                        if 'fetch_time' in df.columns and 'Strike Price' in df.columns and pivot_metric in df.columns:
                            df['time_str'] = pd.to_datetime(df['fetch_time']).dt.strftime('%H:%M')
                            time_strs = sorted(df['time_str'].unique())
                            strike_prices = sorted(df['Strike Price'].unique())
                            fig = go.Figure()
                            for strike in strike_prices:
                                y_vals = []
                                for t in time_strs:
                                    val = df[(df['Strike Price'] == strike) & (df['time_str'] == t)][pivot_metric]
                                    y_vals.append(val.iloc[0] if not val.empty else 0)
                                if chart_orientation == "Vertical":
                                    fig.add_trace(go.Bar(
                                        x=time_strs,
                                        y=y_vals,
                                        name=f"{strike:.2f}",
                                        text=[f'{v:,.0f}' if v > 0 else '' for v in y_vals],
                                        textposition='inside',
                                        orientation='v',
                                    ))
                                else:
                                    fig.add_trace(go.Bar(
                                        y=time_strs,
                                        x=y_vals,
                                        name=f"{strike:.2f}",
                                        text=[f'{v:,.0f}' if v > 0 else '' for v in y_vals],
                                        textposition='inside',
                                        orientation='h',
                                    ))
                            if chart_orientation == "Vertical":
                                fig.update_layout(
                                    barmode='stack',
                                    title_text=f'<b>{pivot_metric} by Time (Stacked by Strike)</b>',
                                    xaxis_title='Time',
                                    yaxis_title=f'{pivot_metric}',
                                    legend_title='Strike Price',
                                    height=500,
                                    template='plotly_white',
                                )
                            else:
                                fig.update_layout(
                                    barmode='stack',
                                    title_text=f'<b>{pivot_metric} by Time (Stacked by Strike)</b>',
                                    yaxis_title='Time',
                                    xaxis_title=f'{pivot_metric}',
                                    legend_title='Strike Price',
                                    height=500,
                                    template='plotly_white',
                                )
                            for trace in fig.data:
                                if isinstance(trace, go.Bar):
                                    trace.textangle = 0
                            st.plotly_chart(fig, use_container_width=True)
                            # Create pivot_table_df for this chart type
                            pivot_table_df = create_pivot_table(df, value_col=pivot_metric)
                            # Format only numeric columns
                            numeric_cols = pivot_table_df.select_dtypes(include=[np.number]).columns
                            format_dict = {col: '{:,.2f}' for col in numeric_cols}
                            styled_df = pivot_table_df.style.format(format_dict)
                            st.dataframe(styled_df, use_container_width=True)
            
            # Data tables section
            if "Raw Data Tables" in visible_sections:
                st.subheader("📋 Raw Data Tables")
                # Latest data table
                st.markdown('<div class="data-table">', unsafe_allow_html=True)
                st.subheader("📋 Latest Data")
                # Sort by timestamp or fetch_time descending for most recent first
                sort_col = None
                for col in ['timestamp', 'fetch_time', 'Timestamp', 'Fetch Time']:
                    if col in df.columns:
                        sort_col = col
                        break
                if sort_col:
                    latest_records = df.sort_values(by=sort_col, ascending=False).head(100)
                else:
                    latest_records = df.head(100)
                # Ensure Instrument and Strike Price columns are present and first
                if 'Instrument' in latest_records.columns and 'Strike Price' in latest_records.columns:
                    cols = ['Instrument', 'Strike Price'] + [col for col in latest_records.columns if col not in ['Instrument', 'Strike Price']]
                    latest_records = latest_records[cols]
                    # Add Instrument (end) and Strike Price (end) at the end
                    latest_records['Instrument (end)'] = latest_records['Instrument']
                    latest_records['Strike Price (end)'] = latest_records['Strike Price']
                    # Reorder so (end) columns are last
                    non_end_cols = [col for col in latest_records.columns if col not in ['Instrument (end)', 'Strike Price (end)']]
                    latest_records = latest_records[non_end_cols + ['Instrument (end)', 'Strike Price (end)']]
                elif 'Strike Price' in latest_records.columns:
                    cols = ['Strike Price'] + [col for col in latest_records.columns if col != 'Strike Price']
                    latest_records = latest_records[cols]
                    # Add Strike Price (end) at the end
                    latest_records['Strike Price (end)'] = latest_records['Strike Price']
                    # Reorder so (end) column is last
                    non_end_cols = [col for col in latest_records.columns if col != 'Strike Price (end)']
                    latest_records = latest_records[non_end_cols + ['Strike Price (end)']]
                # Format Strike Price to 2 decimal places using Styler (do not convert to string)
                try:
                    styled_df = latest_records.style.format({"Strike Price": "{:.2f}" if 'Strike Price' in latest_records.columns else None})
                    if 'Instrument' in latest_records.columns and 'Strike Price' in latest_records.columns:
                        styled_df = styled_df.set_sticky(axis="columns")
                    elif 'Strike Price' in latest_records.columns:
                        styled_df = styled_df.set_sticky(axis="columns")
                    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=450)
                except Exception:
                    st.dataframe(latest_records, use_container_width=True, hide_index=True, height=450)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Detailed data table
            if "Detailed Data Table" in visible_sections:
                st.subheader("📋 Detailed Data")
                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    show_columns = st.multiselect(
                        "Select columns to display:",
                        df.columns.tolist(),
                        default=['Symbol', 'expiry_date', 'fetch_time', 'Spot Price', 'ATM Strike', 'CE OI', 'PE OI', 'CE LTP', 'PE LTP', 'CE IV', 'PE IV', 'timestamp']
                    )
                with col2:
                    search_term = st.text_input("Search in data:", "")
                # Filter data
                filtered_df = df[show_columns] if show_columns else df
                if search_term:
                    # Simple search across all string columns
                    mask = pd.DataFrame([filtered_df[col].astype(str).str.contains(search_term, case=False, na=False) 
                                       for col in filtered_df.columns]).any()
                    filtered_df = filtered_df[mask]
                st.dataframe(filtered_df, use_container_width=True)
            
            # Data summary
            if "Data Summary" in visible_sections:
                st.subheader("📈 Data Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Records", len(df))
                    st.metric("Unique Strikes", df['Strike Price'].nunique())
                with col2:
                    if 'Spot Price' in df.columns:
                        st.metric("Avg Spot Price", f"{df['Spot Price'].mean():.2f}")
                        st.metric("Min Spot Price", f"{df['Spot Price'].min():.2f}")
                with col3:
                    if 'Spot Price' in df.columns:
                        st.metric("Max Spot Price", f"{df['Spot Price'].max():.2f}")
                    if 'CE OI' in df.columns:
                        st.metric("Max CE OI", f"{df['CE OI'].max():,}")
                with col4:
                    if 'PE OI' in df.columns:
                        st.metric("Max PE OI", f"{df['PE OI'].max():,}")
                    if 'CE IV' in df.columns:
                        st.metric("Max CE IV", f"{df['CE IV'].max():.2f}%")
        
        else:
            st.warning(f"No data available for {selected_symbol} on {selected_date}")
    
    # Auto-refresh indicator
    st.sidebar.markdown("---")
    st.sidebar.markdown("🔄 Auto-refresh every 60 seconds")
    st.sidebar.markdown(f"🕒 Last dashboard refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        📊 Option Chain Dashboard | Real-time Data Visualization | Built with Streamlit & Plotly
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 
