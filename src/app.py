"""
Professional Financial Analysis Dashboard
Built with Streamlit and Plotly - Enhanced UI with Larger Charts
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import asyncio
from financial_agent import create_financial_agent
from financial_tools import (
    get_stock_data,
    calculate_moving_averages,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    get_support_resistance,
    analyze_volume_trend,
    get_comprehensive_analysis
)
from utils.config import load_env_variables, config

# Load configuration
try:
    env_name = load_env_variables()
    app_config = config[env_name]
    DEFAULT_AZURE_ENDPOINT = app_config.AZURE_OPENAI_ENDPOINT
    DEFAULT_AZURE_API_KEY = app_config.AZURE_OPENAI_API_KEY
    DEFAULT_AZURE_DEPLOYMENT = app_config.AZURE_OPENAI_DEPLOYMENT_NAME
    DEFAULT_TAVILY_API_KEY = app_config.TAVILY_API_KEY
except Exception as e:
    DEFAULT_AZURE_ENDPOINT = ""
    DEFAULT_AZURE_API_KEY = ""
    DEFAULT_AZURE_DEPLOYMENT = ""
    DEFAULT_TAVILY_API_KEY = ""

# Page configuration
st.set_page_config(
    page_title="Financial Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Professional CSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main > div {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    
    .compact-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3a8a;
        margin: 0;
        padding: 0.5rem 0;
        text-align: center;
        border-bottom: 3px solid #d4af37;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        color: white;
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid rgba(212, 175, 55, 0.3);
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(212, 175, 55, 0.3);
    }
    
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0.3rem 0;
        color: #ffffff;
    }
    
    .metric-label {
        font-size: 0.75rem;
        opacity: 0.95;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
        color: #d4af37;
    }
    
    .metric-change {
        font-size: 0.95rem;
        margin-top: 0.3rem;
    }
    
    .info-card {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        border-left: 4px solid #d4af37;
        margin-bottom: 0.8rem;
        color: #ffffff;
    }
    
    .info-card-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #d4af37;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .info-card-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    .info-card-subtitle {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-top: 0.3rem;
    }
    
    .news-card {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .news-card:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 15px rgba(59, 130, 246, 0.3);
    }
    
    .news-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #d4af37;
        margin-bottom: 0.5rem;
    }
    
    .news-content {
        font-size: 0.95rem;
        color: #e5e7eb;
        line-height: 1.6;
    }
    
    .news-source {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-top: 0.5rem;
        font-style: italic;
    }
    
    .positive {
        color: #10b981;
        font-weight: 700;
    }
    
    .negative {
        color: #ef4444;
        font-weight: 700;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 0.8rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.6rem 1.5rem;
        font-size: 0.95rem;
        font-weight: 600;
        border-radius: 6px;
        color: #ffffff;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(212, 175, 55, 0.3);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #d4af37;
        color: #1e3a8a;
    }
    
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        border: 1px solid #d4af37;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(212, 175, 55, 0.4);
        background: linear-gradient(135deg, #3b82f6 0%, #1e3a8a 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }
    
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e3a8a;
        margin: 0.5rem 0;
        padding: 0.5rem;
        background: linear-gradient(90deg, rgba(212, 175, 55, 0.1) 0%, transparent 100%);
        border-left: 4px solid #d4af37;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = "AAPL"
if 'use_config_keys' not in st.session_state:
    st.session_state.use_config_keys = True
if 'azure_endpoint' not in st.session_state:
    st.session_state.azure_endpoint = DEFAULT_AZURE_ENDPOINT
if 'azure_api_key' not in st.session_state:
    st.session_state.azure_api_key = DEFAULT_AZURE_API_KEY
if 'azure_deployment' not in st.session_state:
    st.session_state.azure_deployment = DEFAULT_AZURE_DEPLOYMENT
if 'tavily_api_key' not in st.session_state:
    st.session_state.tavily_api_key = DEFAULT_TAVILY_API_KEY
if 'ai_analysis_done' not in st.session_state:
    st.session_state.ai_analysis_done = False
if 'ai_analysis_result' not in st.session_state:
    st.session_state.ai_analysis_result = None
if 'news_loaded' not in st.session_state:
    st.session_state.news_loaded = False
if 'news_result' not in st.session_state:
    st.session_state.news_result = None


def initialize_agent():
    """Initialize the financial agent"""
    try:
        if st.session_state.agent is None:
            st.session_state.agent = create_financial_agent(
                azure_endpoint=st.session_state.get('azure_endpoint'),
                azure_api_key=st.session_state.get('azure_api_key'),
                azure_deployment=st.session_state.get('azure_deployment'),
                tavily_api_key=st.session_state.get('tavily_api_key')
            )
        return True
    except Exception as e:
        st.error(f"❌ Failed to initialize agent: {e}")
        return False


def create_metric_card(label, value, change=None, icon=""):
    """Create a styled metric card"""
    change_html = ""
    if change is not None:
        change_class = "positive" if change >= 0 else "negative"
        change_symbol = "▲" if change >= 0 else "▼"
        change_html = f'<div class="metric-change"><span class="{change_class}">{change_symbol} {abs(change):.2f}%</span></div>'
    
    return f"""
    <div class="metric-card">
        <div class="metric-label">{icon} {label}</div>
        <div class="metric-value">{value}</div>
        {change_html}
    </div>
    """


def create_info_card(title, value, subtitle=""):
    """Create an info card"""
    subtitle_html = f'<div class="info-card-subtitle">{subtitle}</div>' if subtitle else ""
    return f"""
    <div class="info-card">
        <div class="info-card-title">{title}</div>
        <div class="info-card-value">{value}</div>
        {subtitle_html}
    </div>
    """


def create_price_chart(ticker, period="3mo"):
    """Create detailed price chart with volume"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            return None
        
        # Calculate moving averages
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'{ticker} Price & Moving Averages', 'Volume'),
            vertical_spacing=0.05
        )
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#10b981',
            decreasing_line_color='#ef4444'
        ), row=1, col=1)
        
        # Moving averages
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA20'],
            name='SMA 20',
            line=dict(color='#f59e0b', width=2)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA50'],
            name='SMA 50',
            line=dict(color='#8b5cf6', width=2)
        ), row=1, col=1)
        
        # Volume bars
        colors = ['#10b981' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#ef4444' 
                  for i in range(len(df))]
        
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False
        ), row=2, col=1)
        
        fig.update_layout(
            template='plotly_dark',
            height=600,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    except Exception as e:
        return None


def create_indicators_chart(ticker):
    """Create technical indicators chart"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="3mo")
        
        if df.empty:
            return None
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('RSI (14)', 'MACD'),
            vertical_spacing=0.12,
            row_heights=[0.5, 0.5]
        )
        
        # RSI
        fig.add_trace(go.Scatter(
            x=df.index, y=rsi,
            name='RSI',
            line=dict(color='#8b5cf6', width=2),
            fill='tozeroy',
            fillcolor='rgba(139, 92, 246, 0.1)'
        ), row=1, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", line_width=1, row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#10b981", line_width=1, row=1, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="gray", opacity=0.1, row=1, col=1)
        
        # MACD
        fig.add_trace(go.Scatter(
            x=df.index, y=macd,
            name='MACD',
            line=dict(color='#3b82f6', width=2)
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index, y=signal,
            name='Signal',
            line=dict(color='#f59e0b', width=2)
        ), row=2, col=1)
        
        # MACD Histogram
        colors = ['#10b981' if val >= 0 else '#ef4444' for val in histogram]
        fig.add_trace(go.Bar(
            x=df.index,
            y=histogram,
            name='Histogram',
            marker_color=colors,
            opacity=0.5
        ), row=2, col=1)
        
        fig.update_layout(
            template='plotly_dark',
            height=500,
            showlegend=True,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_yaxes(range=[0, 100], row=1, col=1)
        
        return fig
    except Exception as e:
        return None


def create_bollinger_bands_chart(ticker):
    """Create Bollinger Bands chart"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="3mo")
        
        if df.empty:
            return None
        
        # Calculate Bollinger Bands
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['STD20'] = df['Close'].rolling(window=20).std()
        df['Upper'] = df['SMA20'] + (df['STD20'] * 2)
        df['Lower'] = df['SMA20'] - (df['STD20'] * 2)
        
        fig = go.Figure()
        
        # Upper band
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Upper'],
            name='Upper Band',
            line=dict(color='#ef4444', width=1, dash='dash'),
            fill=None
        ))
        
        # Middle band
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA20'],
            name='SMA 20',
            line=dict(color='#f59e0b', width=2)
        ))
        
        # Lower band
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Lower'],
            name='Lower Band',
            line=dict(color='#10b981', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(59, 130, 246, 0.1)'
        ))
        
        # Price
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            name='Close Price',
            line=dict(color='#3b82f6', width=2)
        ))
        
        fig.update_layout(
            title=f'{ticker} Bollinger Bands (20, 2)',
            yaxis_title='Price (USD)',
            template='plotly_dark',
            height=500,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    except Exception as e:
        return None


def create_volume_analysis_chart(ticker):
    """Create volume analysis chart"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="3mo")
        
        if df.empty:
            return None
        
        # Calculate volume moving average
        df['Volume_SMA20'] = df['Volume'].rolling(window=20).mean()
        
        fig = go.Figure()
        
        # Volume bars colored by price movement
        colors = ['#10b981' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#ef4444' 
                  for i in range(len(df))]
        
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ))
        
        # Volume moving average
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Volume_SMA20'],
            name='Volume SMA 20',
            line=dict(color='#f59e0b', width=3)
        ))
        
        fig.update_layout(
            title=f'{ticker} Volume Analysis',
            yaxis_title='Volume',
            template='plotly_dark',
            height=400,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    except Exception as e:
        return None


def create_price_distribution_chart(ticker):
    """Create price distribution histogram"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        
        if df.empty:
            return None
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=df['Close'],
            nbinsx=50,
            name='Price Distribution',
            marker_color='#3b82f6',
            opacity=0.7
        ))
        
        # Add current price line
        current_price = df['Close'].iloc[-1]
        fig.add_vline(
            x=current_price,
            line_dash="dash",
            line_color="#d4af37",
            line_width=3,
            annotation_text=f"Current: ${current_price:.2f}"
        )
        
        fig.update_layout(
            title=f'{ticker} Price Distribution (1 Year)',
            xaxis_title='Price (USD)',
            yaxis_title='Frequency',
            template='plotly_dark',
            height=400,
            showlegend=True
        )
        
        return fig
    except Exception as e:
        return None


def create_returns_heatmap(ticker):
    """Create monthly returns heatmap"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        
        if df.empty:
            return None
        
        # Calculate daily returns
        df['Returns'] = df['Close'].pct_change() * 100
        
        # Group by month and calculate average
        df['Month'] = df.index.strftime('%B')
        
        # Create pivot table
        pivot = df.pivot_table(values='Returns', index='Month', aggfunc='mean')
        
        fig = go.Figure(data=go.Heatmap(
            z=[pivot.values.flatten()],
            x=pivot.index,
            y=['Avg Return %'],
            colorscale='RdYlGn',
            text=[[f'{val:.2f}%' for val in pivot.values.flatten()]],
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title="Return %")
        ))
        
        fig.update_layout(
            title=f'{ticker} Monthly Average Returns',
            template='plotly_dark',
            height=200,
            xaxis_title='Month'
        )
        
        return fig
    except Exception as e:
        return None


def display_stock_metrics(ticker):
    """Display key stock metrics in enhanced cards"""
    try:
        result = get_stock_data.invoke({"ticker": ticker, "period": "1mo"})
        
        if "error" in result:
            st.error(result["error"])
            return
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(
                create_metric_card(
                    "Price",
                    f"${result['current_price']:.2f}",
                    result['change_percent'],
                    "💰"
                ),
                unsafe_allow_html=True
            )
        
        with col2:
            volume_m = result['volume'] / 1_000_000
            st.markdown(
                create_metric_card(
                    "Volume",
                    f"{volume_m:.1f}M",
                    icon="📊"
                ),
                unsafe_allow_html=True
            )
        
        with col3:
            if result.get('market_cap'):
                market_cap = result['market_cap'] / 1e9
                st.markdown(
                    create_metric_card(
                        "Market Cap",
                        f"${market_cap:.1f}B",
                        icon="🏢"
                    ),
                    unsafe_allow_html=True
                )
        
        with col4:
            if result.get('pe_ratio'):
                st.markdown(
                    create_metric_card(
                        "P/E Ratio",
                        f"{result['pe_ratio']:.2f}",
                        icon="📈"
                    ),
                    unsafe_allow_html=True
                )
        
        with col5:
            st.markdown(
                create_metric_card(
                    "Sector",
                    result.get('sector', 'N/A')[:12],
                    icon="🏭"
                ),
                unsafe_allow_html=True
            )
            
    except Exception as e:
        st.error(f"Error: {e}")


def load_ai_analysis(ticker):
    """Load AI analysis automatically"""
    if not st.session_state.ai_analysis_done and st.session_state.get('azure_api_key'):
        if initialize_agent():
            with st.spinner("🤖 AI is analyzing the stock..."):
                try:
                    result = st.session_state.agent.analyze_stock(ticker)
                    st.session_state.ai_analysis_result = result
                    st.session_state.ai_analysis_done = True
                except Exception as e:
                    st.error(f"❌ Error: {e}")


def load_market_news(ticker):
    """Load market news automatically"""
    if not st.session_state.news_loaded and st.session_state.get('tavily_api_key'):
        if initialize_agent():
            with st.spinner("📰 Fetching latest market news..."):
                try:
                    # Get multiple news queries
                    queries = [
                        f"{ticker} stock latest news today",
                        "stock market news today",
                        f"{ticker} analysis forecast"
                    ]
                    
                    results = []
                    for query in queries:
                        result = st.session_state.agent.get_market_news(query)
                        results.append(result)
                    
                    st.session_state.news_result = results
                    st.session_state.news_loaded = True
                except Exception as e:
                    st.error(f"❌ Error: {e}")


def main():
    """Main dashboard function"""
    
    # Sidebar
    with st.sidebar:
        st.markdown("---")
        st.title("📊 Financial Dashboard")
        st.markdown("---")
        
        # API Configuration
        with st.expander("⚙️ Configuration", expanded=False):
            use_config = st.radio(
                "Config source:",
                ["Config file", "Manual entry"],
                index=0 if st.session_state.use_config_keys else 1
            )
            
            st.session_state.use_config_keys = (use_config == "Config file")
            
            if st.session_state.use_config_keys:
                st.info("📋 Using config file")
                st.session_state.azure_endpoint = DEFAULT_AZURE_ENDPOINT
                st.session_state.azure_api_key = DEFAULT_AZURE_API_KEY
                st.session_state.azure_deployment = DEFAULT_AZURE_DEPLOYMENT
                st.session_state.tavily_api_key = DEFAULT_TAVILY_API_KEY
            else:
                azure_endpoint = st.text_input("Azure Endpoint", value=st.session_state.get('azure_endpoint', ''), type="password")
                azure_api_key = st.text_input("Azure API Key", value=st.session_state.get('azure_api_key', ''), type="password")
                azure_deployment = st.text_input("Deployment", value=st.session_state.get('azure_deployment', ''))
                tavily_api_key = st.text_input("Tavily Key", value=st.session_state.get('tavily_api_key', ''), type="password")
                
                if st.button("💾 Save", use_container_width=True):
                    st.session_state.azure_endpoint = azure_endpoint
                    st.session_state.azure_api_key = azure_api_key
                    st.session_state.azure_deployment = azure_deployment
                    st.session_state.tavily_api_key = tavily_api_key
                    st.session_state.agent = None
                    st.success("✅ Saved!")
                    st.rerun()
        
        st.markdown("---")
        
        # Stock Selection
        st.markdown("### 📈 Stock Selection")
        
        market_type = st.selectbox(
            "Market",
            ["US Market", "Indian Market (NSE)"]
        )
        
        if market_type == "US Market":
            popular_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
        else:
            popular_stocks = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]
        
        ticker = st.selectbox("Stock", popular_stocks)
        custom_ticker = st.text_input("Custom ticker")
        
        if custom_ticker:
            ticker = custom_ticker.upper()
        
        # Reset analysis if ticker changed
        if ticker != st.session_state.selected_ticker:
            st.session_state.ai_analysis_done = False
            st.session_state.news_loaded = False
            st.session_state.selected_ticker = ticker
        
        period = st.selectbox(
            "Period",
            ["1d", "5d", "1mo", "3mo", "6mo", "1y"],
            index=2
        )
        
        st.markdown("---")
        
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.ai_analysis_done = False
            st.session_state.news_loaded = False
            st.rerun()
        
        st.markdown("---")
        
        # Status
        st.markdown("### 📌 Status")
        if st.session_state.get('azure_api_key'):
            st.success("🟢 Azure AI")
        else:
            st.error("🔴 Azure AI")
        
        if st.session_state.get('tavily_api_key'):
            st.success("🟢 News API")
        else:
            st.warning("🟡 News API")
    
    # Main content - Compact header
    st.markdown(f'<h2 class="compact-header">📊 {ticker} Analysis Dashboard</h2>', unsafe_allow_html=True)
    
    # Metrics
    display_stock_metrics(ticker)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", 
        "📉 Technical", 
        "🤖 AI Analysis",
        "📰 News"
    ])
    
    with tab1:
        # Main price chart - Show immediately
        st.markdown('<div class="section-header">Price Action & Volume</div>', unsafe_allow_html=True)
        
        with st.spinner("Loading price chart..."):
            price_chart = create_price_chart(ticker, period)
            if price_chart:
                st.plotly_chart(price_chart, use_container_width=True)
            else:
                st.error("Unable to load price chart")
        
        # Two columns for additional charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-header">Volume Analysis</div>', unsafe_allow_html=True)
            with st.spinner("Loading volume chart..."):
                vol_chart = create_volume_analysis_chart(ticker)
                if vol_chart:
                    st.plotly_chart(vol_chart, use_container_width=True)
                else:
                    st.error("Unable to load volume chart")
        
        with col2:
            st.markdown('<div class="section-header">Price Distribution</div>', unsafe_allow_html=True)
            with st.spinner("Loading distribution chart..."):
                dist_chart = create_price_distribution_chart(ticker)
                if dist_chart:
                    st.plotly_chart(dist_chart, use_container_width=True)
                else:
                    st.error("Unable to load distribution chart")
        
        # Quick metrics row
        st.markdown('<div class="section-header">Quick Indicators</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rsi_data = calculate_rsi.invoke({"ticker": ticker})
            if "error" not in rsi_data:
                st.markdown(
                    create_info_card(
                        "RSI (14)",
                        f"{rsi_data['rsi']:.2f}",
                        rsi_data['interpretation']
                    ),
                    unsafe_allow_html=True
                )
        
        with col2:
            macd_data = calculate_macd.invoke({"ticker": ticker})
            if "error" not in macd_data:
                st.markdown(
                    create_info_card(
                        "MACD",
                        macd_data['interpretation'].split('-')[0].strip(),
                        f"MACD: {macd_data['macd']:.4f}"
                    ),
                    unsafe_allow_html=True
                )
        
        with col3:
            vol_data = analyze_volume_trend.invoke({"ticker": ticker})
            if "error" not in vol_data:
                st.markdown(
                    create_info_card(
                        "Volume Trend",
                        f"{vol_data['volume_change_percent']:+.1f}%",
                        vol_data['signal']
                    ),
                    unsafe_allow_html=True
                )
        
        # Monthly returns heatmap
        st.markdown('<div class="section-header">Monthly Returns Pattern</div>', unsafe_allow_html=True)
        with st.spinner("Loading returns heatmap..."):
            heatmap = create_returns_heatmap(ticker)
            if heatmap:
                st.plotly_chart(heatmap, use_container_width=True)
            else:
                st.error("Unable to load returns heatmap")
    
    with tab2:
        # Technical indicators chart (large)
        st.markdown('<div class="section-header">RSI & MACD Indicators</div>', unsafe_allow_html=True)
        indicators = create_indicators_chart(ticker)
        if indicators:
            st.plotly_chart(indicators, use_container_width=True)
        
        # Bollinger Bands chart (large)
        st.markdown('<div class="section-header">Bollinger Bands Analysis</div>', unsafe_allow_html=True)
        bb_chart = create_bollinger_bands_chart(ticker)
        if bb_chart:
            st.plotly_chart(bb_chart, use_container_width=True)
        
        # Technical data in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-header">Bollinger Bands Data</div>', unsafe_allow_html=True)
            bb_data = calculate_bollinger_bands.invoke({"ticker": ticker})
            if "error" not in bb_data:
                st.write(f"**Upper Band:** ${bb_data['upper_band']:.2f}")
                st.write(f"**Middle Band (SMA 20):** ${bb_data['middle_band']:.2f}")
                st.write(f"**Lower Band:** ${bb_data['lower_band']:.2f}")
                st.write(f"**Current Price:** ${bb_data['current_price']:.2f}")
                st.info(bb_data['interpretation'])
        
        with col2:
            st.markdown('<div class="section-header">Support & Resistance Levels</div>', unsafe_allow_html=True)
            sr_data = get_support_resistance.invoke({"ticker": ticker})
            if "error" not in sr_data:
                st.write(f"**Pivot Point:** ${sr_data['pivot_point']:.2f}")
                st.write(f"**Resistance 1:** ${sr_data['resistance_levels']['R1']:.2f}")
                st.write(f"**Resistance 2:** ${sr_data['resistance_levels']['R2']:.2f}")
                st.write(f"**Support 1:** ${sr_data['support_levels']['S1']:.2f}")
                st.write(f"**Support 2:** ${sr_data['support_levels']['S2']:.2f}")
        
        # Moving Averages
        st.markdown('<div class="section-header">Moving Averages</div>', unsafe_allow_html=True)
        ma_data = calculate_moving_averages.invoke({"ticker": ticker, "periods": "20,50,200"})
        if "error" not in ma_data:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("SMA 20", f"${ma_data['moving_averages']['SMA_20']:.2f}")
            with col2:
                st.metric("SMA 50", f"${ma_data['moving_averages']['SMA_50']:.2f}")
            with col3:
                st.metric("SMA 200", f"${ma_data['moving_averages']['SMA_200']:.2f}")
            with col4:
                st.metric("Current Price", f"${ma_data['current_price']:.2f}")
    
    with tab3:
        if not st.session_state.get('azure_api_key'):
            st.warning("⚠️ Configure Azure OpenAI in sidebar to enable AI Analysis")
            st.info("💡 AI Analysis provides comprehensive stock evaluation using advanced language models")
        else:
            # Auto-load analysis
            load_ai_analysis(ticker)
            
            if st.session_state.ai_analysis_result:
                messages = st.session_state.ai_analysis_result.get('messages', [])
                for msg in messages:
                    if hasattr(msg, 'content') and msg.type == 'ai':
                        st.markdown("### 🎯 AI-Powered Stock Analysis")
                        st.markdown(msg.content)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔄 Regenerate Analysis", use_container_width=True):
                    st.session_state.ai_analysis_done = False
                    st.rerun()
            with col2:
                st.info("💡 AI analysis updates automatically when you change stocks")
    
    with tab4:
        if not st.session_state.get('tavily_api_key'):
            st.info("💡 Add Tavily API key in sidebar to enable real-time market news")
            st.warning("⚠️ News feature requires Tavily API configuration")
        else:
            # Auto-load news
            load_market_news(ticker)
            
            if st.session_state.news_result:
                st.markdown("### 📰 Latest Market News & Analysis")
                for idx, result in enumerate(st.session_state.news_result):
                    messages = result.get('messages', [])
                    for msg in messages:
                        if hasattr(msg, 'content') and msg.type == 'ai':
                            news_titles = [
                                f"📌 {ticker} Latest Updates",
                                "🌐 Market Overview",
                                "📊 Analysis & Forecasts"
                            ]
                            st.markdown(f"""
                            <div class="news-card">
                                <div class="news-title">{news_titles[idx]}</div>
                                <div class="news-content">{msg.content}</div>
                                <div class="news-source">Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
                            </div>
                            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔄 Refresh News", use_container_width=True):
                    st.session_state.news_loaded = False
                    st.rerun()
            with col2:
                st.info("💡 News updates automatically when you change stocks")


if __name__ == "__main__":
    main()