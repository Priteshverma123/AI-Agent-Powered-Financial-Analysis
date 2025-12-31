"""
Enhanced Financial Analysis Tools
Provides comprehensive stock analysis with technical indicators
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from langchain_core.tools import tool
import logging

logger = logging.getLogger(__name__)


@tool
def get_stock_data(ticker: str, period: str = "1mo") -> Dict:
    """
    Get current stock data with basic information.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'RELIANCE.NS')
        period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y')
    
    Returns:
        Dictionary with current stock information
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        info = stock.info
        
        if hist.empty:
            return {"error": f"No data found for {ticker}"}
        
        current_price = hist['Close'].iloc[-1]
        previous_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        change = current_price - previous_close
        change_pct = (change / previous_close) * 100
        
        return {
            "ticker": ticker,
            "company_name": info.get("longName", ticker),
            "current_price": round(float(current_price), 2),
            "previous_close": round(float(previous_close), 2),
            "change": round(float(change), 2),
            "change_percent": round(float(change_pct), 2),
            "volume": int(hist['Volume'].iloc[-1]),
            "market_cap": info.get("marketCap"),
            "sector": info.get("sector", "N/A"),
            "pe_ratio": info.get("trailingPE"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "dividend_yield": info.get("dividendYield"),
        }
    except Exception as e:
        logger.error(f"Error fetching stock data: {e}")
        return {"error": str(e)}


@tool
def calculate_moving_averages(ticker: str, periods: str = "20,50,200") -> Dict:
    """
    Calculate Simple Moving Averages (SMA) for a stock.
    
    Args:
        ticker: Stock ticker symbol
        periods: Comma-separated MA periods (e.g., '20,50,200')
    
    Returns:
        Dictionary with moving averages and signals
    """
    try:
        period_list = [int(p.strip()) for p in periods.split(",")]
        max_period = max(period_list)
        
        stock = yf.Ticker(ticker)
        # Get enough data for the longest MA
        hist = stock.history(period=f"{max_period + 50}d")
        
        if hist.empty:
            return {"error": f"No data found for {ticker}"}
        
        result = {
            "ticker": ticker,
            "current_price": round(float(hist['Close'].iloc[-1]), 2),
            "moving_averages": {},
            "signals": []
        }
        
        # Calculate MAs
        for period in period_list:
            ma_value = hist['Close'].rolling(window=period).mean().iloc[-1]
            result["moving_averages"][f"SMA_{period}"] = round(float(ma_value), 2)
        
        # Generate signals
        current_price = result["current_price"]
        for period in period_list:
            ma = result["moving_averages"][f"SMA_{period}"]
            if current_price > ma:
                result["signals"].append(f"Price above SMA_{period} - Bullish")
            else:
                result["signals"].append(f"Price below SMA_{period} - Bearish")
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating moving averages: {e}")
        return {"error": str(e)}


@tool
def calculate_rsi(ticker: str, period: int = 14) -> Dict:
    """
    Calculate Relative Strength Index (RSI) for a stock.
    
    Args:
        ticker: Stock ticker symbol
        period: RSI period (default: 14)
    
    Returns:
        Dictionary with RSI value and interpretation
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{period + 30}d")
        
        if hist.empty:
            return {"error": f"No data found for {ticker}"}
        
        # Calculate price changes
        delta = hist['Close'].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # Interpret RSI
        if current_rsi > 70:
            interpretation = "Overbought - Potential sell signal"
        elif current_rsi < 30:
            interpretation = "Oversold - Potential buy signal"
        else:
            interpretation = "Neutral"
        
        return {
            "ticker": ticker,
            "rsi": round(float(current_rsi), 2),
            "period": period,
            "interpretation": interpretation,
            "current_price": round(float(hist['Close'].iloc[-1]), 2)
        }
        
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return {"error": str(e)}


@tool
def calculate_macd(ticker: str) -> Dict:
    """
    Calculate MACD (Moving Average Convergence Divergence) for a stock.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with MACD values and signals
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")
        
        if hist.empty:
            return {"error": f"No data found for {ticker}"}
        
        # Calculate EMAs
        ema_12 = hist['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = hist['Close'].ewm(span=26, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = ema_12 - ema_26
        
        # Calculate signal line (9-day EMA of MACD)
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_histogram = histogram.iloc[-1]
        
        # Generate signal
        if current_macd > current_signal and histogram.iloc[-2] < current_histogram:
            signal = "Bullish crossover - Potential buy signal"
        elif current_macd < current_signal and histogram.iloc[-2] > current_histogram:
            signal = "Bearish crossover - Potential sell signal"
        else:
            signal = "No clear signal"
        
        return {
            "ticker": ticker,
            "macd": round(float(current_macd), 4),
            "signal_line": round(float(current_signal), 4),
            "histogram": round(float(current_histogram), 4),
            "interpretation": signal,
            "current_price": round(float(hist['Close'].iloc[-1]), 2)
        }
        
    except Exception as e:
        logger.error(f"Error calculating MACD: {e}")
        return {"error": str(e)}


@tool
def calculate_bollinger_bands(ticker: str, period: int = 20, std_dev: int = 2) -> Dict:
    """
    Calculate Bollinger Bands for a stock.
    
    Args:
        ticker: Stock ticker symbol
        period: Moving average period (default: 20)
        std_dev: Number of standard deviations (default: 2)
    
    Returns:
        Dictionary with Bollinger Bands and interpretation
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{period + 50}d")
        
        if hist.empty:
            return {"error": f"No data found for {ticker}"}
        
        # Calculate middle band (SMA)
        middle_band = hist['Close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        std = hist['Close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        current_price = hist['Close'].iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_middle = middle_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        
        # Interpret position
        if current_price > current_upper:
            interpretation = "Price above upper band - Overbought"
        elif current_price < current_lower:
            interpretation = "Price below lower band - Oversold"
        elif current_price > current_middle:
            interpretation = "Price above middle band - Bullish"
        else:
            interpretation = "Price below middle band - Bearish"
        
        return {
            "ticker": ticker,
            "current_price": round(float(current_price), 2),
            "upper_band": round(float(current_upper), 2),
            "middle_band": round(float(current_middle), 2),
            "lower_band": round(float(current_lower), 2),
            "interpretation": interpretation,
            "bandwidth": round(float((current_upper - current_lower) / current_middle * 100), 2)
        }
        
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {e}")
        return {"error": str(e)}


@tool
def get_support_resistance(ticker: str, period: str = "3mo") -> Dict:
    """
    Calculate support and resistance levels using pivot points.
    
    Args:
        ticker: Stock ticker symbol
        period: Historical period to analyze
    
    Returns:
        Dictionary with support and resistance levels
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            return {"error": f"No data found for {ticker}"}
        
        # Get recent high, low, close
        high = hist['High'].max()
        low = hist['Low'].min()
        close = hist['Close'].iloc[-1]
        
        # Calculate pivot point
        pivot = (high + low + close) / 3
        
        # Calculate support and resistance levels
        resistance_1 = (2 * pivot) - low
        support_1 = (2 * pivot) - high
        resistance_2 = pivot + (high - low)
        support_2 = pivot - (high - low)
        resistance_3 = high + 2 * (pivot - low)
        support_3 = low - 2 * (high - pivot)
        
        return {
            "ticker": ticker,
            "current_price": round(float(close), 2),
            "pivot_point": round(float(pivot), 2),
            "resistance_levels": {
                "R1": round(float(resistance_1), 2),
                "R2": round(float(resistance_2), 2),
                "R3": round(float(resistance_3), 2)
            },
            "support_levels": {
                "S1": round(float(support_1), 2),
                "S2": round(float(support_2), 2),
                "S3": round(float(support_3), 2)
            }
        }
        
    except Exception as e:
        logger.error(f"Error calculating support/resistance: {e}")
        return {"error": str(e)}


@tool
def analyze_volume_trend(ticker: str, period: str = "1mo") -> Dict:
    """
    Analyze volume trends and patterns.
    
    Args:
        ticker: Stock ticker symbol
        period: Period to analyze
    
    Returns:
        Dictionary with volume analysis
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            return {"error": f"No data found for {ticker}"}
        
        avg_volume = hist['Volume'].mean()
        current_volume = hist['Volume'].iloc[-1]
        volume_change = ((current_volume - avg_volume) / avg_volume) * 100
        
        # Analyze price-volume relationship
        price_change = hist['Close'].pct_change().iloc[-1] * 100
        
        if volume_change > 50 and price_change > 0:
            signal = "Strong buying pressure - Bullish"
        elif volume_change > 50 and price_change < 0:
            signal = "Strong selling pressure - Bearish"
        elif volume_change < -30:
            signal = "Low volume - Weak trend"
        else:
            signal = "Normal volume activity"
        
        return {
            "ticker": ticker,
            "current_volume": int(current_volume),
            "average_volume": int(avg_volume),
            "volume_change_percent": round(float(volume_change), 2),
            "price_change_percent": round(float(price_change), 2),
            "signal": signal
        }
        
    except Exception as e:
        logger.error(f"Error analyzing volume: {e}")
        return {"error": str(e)}


@tool
def get_comprehensive_analysis(ticker: str) -> Dict:
    """
    Get a comprehensive technical analysis combining multiple indicators.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with complete technical analysis
    """
    try:
        # Get all analyses
        stock_data = get_stock_data.invoke({"ticker": ticker, "period": "1mo"})
        ma_data = calculate_moving_averages.invoke({"ticker": ticker, "periods": "20,50,200"})
        rsi_data = calculate_rsi.invoke({"ticker": ticker, "period": 14})
        macd_data = calculate_macd.invoke({"ticker": ticker})
        bb_data = calculate_bollinger_bands.invoke({"ticker": ticker, "period": 20})
        volume_data = analyze_volume_trend.invoke({"ticker": ticker, "period": "1mo"})
        
        # Calculate overall sentiment
        bullish_signals = 0
        bearish_signals = 0
        
        # Count signals
        for signal in ma_data.get("signals", []):
            if "Bullish" in signal:
                bullish_signals += 1
            else:
                bearish_signals += 1
        
        if rsi_data.get("rsi", 50) < 30:
            bullish_signals += 1
        elif rsi_data.get("rsi", 50) > 70:
            bearish_signals += 1
        
        if "Bullish" in macd_data.get("interpretation", ""):
            bullish_signals += 1
        elif "Bearish" in macd_data.get("interpretation", ""):
            bearish_signals += 1
        
        if "Bullish" in volume_data.get("signal", ""):
            bullish_signals += 1
        elif "Bearish" in volume_data.get("signal", ""):
            bearish_signals += 1
        
        # Overall sentiment
        total_signals = bullish_signals + bearish_signals
        if total_signals > 0:
            bullish_pct = (bullish_signals / total_signals) * 100
            if bullish_pct > 60:
                overall_sentiment = "BULLISH"
            elif bullish_pct < 40:
                overall_sentiment = "BEARISH"
            else:
                overall_sentiment = "NEUTRAL"
        else:
            overall_sentiment = "NEUTRAL"
        
        return {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "overall_sentiment": overall_sentiment,
            "bullish_signals": bullish_signals,
            "bearish_signals": bearish_signals,
            "stock_info": stock_data,
            "moving_averages": ma_data,
            "rsi": rsi_data,
            "macd": macd_data,
            "bollinger_bands": bb_data,
            "volume_analysis": volume_data
        }
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        return {"error": str(e)}