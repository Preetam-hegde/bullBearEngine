import pandas as pd
import numpy as np
import json

def calculate_technical_indicators(data):
    """Calculate comprehensive technical indicators using pure pandas/numpy"""
    df = data.copy()
    
    # Ensure Close is numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    
    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Price-based indicators
    df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
    df['Price_Change'] = df['Close'] - df['Open']
    df['Price_Change_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
    
    # Volatility (Average True Range approximation)
    df['High_Low'] = df['High'] - df['Low']
    df['High_Close'] = np.abs(df['High'] - df['Close'].shift())
    df['Low_Close'] = np.abs(df['Low'] - df['Close'].shift())
    df['True_Range'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
    df['ATR'] = df['True_Range'].rolling(window=14).mean()
    
    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
    return df

def generate_market_analysis(data, info, symbol):
    """Generate AI-powered market analysis"""
    latest = data.iloc[-1]
    prev = data.iloc[-2]
    
    # Price movement
    price_change = latest['Close'] - prev['Close']
    price_change_pct = (price_change / prev['Close']) * 100
    
    # Technical analysis
    rsi = latest.get('RSI', 50)
    sma_20 = latest.get('SMA_20', latest['Close'])
    sma_50 = latest.get('SMA_50', latest['Close'])
    bb_upper = latest.get('BB_upper', latest['Close'])
    bb_lower = latest.get('BB_lower', latest['Close'])
    
    # Volume analysis
    avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
    volume_ratio = latest['Volume'] / avg_volume if avg_volume > 0 else 1
    
    # MACD analysis
    macd = latest.get('MACD', 0)
    macd_signal = latest.get('MACD_signal', 0)
    
    # Generate analysis
    analysis = []
    
    # Price trend
    if price_change_pct > 3:
        analysis.append(f"🚀 {symbol} shows exceptional bullish momentum with a {price_change_pct:.2f}% surge")
    elif price_change_pct > 1:
        analysis.append(f"🟢 {symbol} demonstrates strong upward movement (+{price_change_pct:.2f}%)")
    elif price_change_pct > 0:
        analysis.append(f"🟡 {symbol} shows modest gains (+{price_change_pct:.2f}%)")
    elif price_change_pct > -1:
        analysis.append(f"🟡 {symbol} experiences slight decline ({price_change_pct:.2f}%)")
    elif price_change_pct > -3:
        analysis.append(f"🔴 {symbol} shows moderate bearish pressure ({price_change_pct:.2f}%)")
    else:
        analysis.append(f"🔻 {symbol} faces significant selling pressure ({price_change_pct:.2f}%)")
    
    # RSI analysis
    if rsi > 80:
        analysis.append(f"🚨 RSI at {rsi:.1f} indicates severely overbought conditions - potential reversal ahead")
    elif rsi > 70:
        analysis.append(f"⚠️ RSI at {rsi:.1f} shows overbought territory - exercise caution")
    elif rsi < 20:
        analysis.append(f"🛒 RSI at {rsi:.1f} signals severely oversold - strong buying opportunity")
    elif rsi < 30:
        analysis.append(f"💡 RSI at {rsi:.1f} suggests oversold conditions - potential buying opportunity")
    elif 40 <= rsi <= 60:
        analysis.append(f"⚖️ RSI at {rsi:.1f} indicates balanced momentum")
    else:
        analysis.append(f"📊 RSI at {rsi:.1f} shows {('bullish' if rsi > 50 else 'bearish')} bias")
    
    # Moving average analysis
    if latest['Close'] > sma_20 > sma_50:
        analysis.append("📈 Strong bullish alignment - price above both 20 and 50-day MAs")
    elif latest['Close'] < sma_20 < sma_50:
        analysis.append("📉 Bearish trend confirmed - price below key moving averages")
    elif latest['Close'] > sma_20 and sma_20 < sma_50:
        analysis.append("🔄 Mixed signals - short-term bullish but longer-term bearish")
    else:
        analysis.append("➡️ Consolidation phase - awaiting directional breakout")
    
    # Bollinger Bands analysis
    if latest['Close'] > bb_upper:
        analysis.append("📊 Price trading above upper Bollinger Band - potential overbought")
    elif latest['Close'] < bb_lower:
        analysis.append("📊 Price near lower Bollinger Band - potential oversold bounce")
    
    # MACD analysis
    if macd > macd_signal and macd > 0:
        analysis.append("⚡ MACD shows strong bullish momentum")
    elif macd < macd_signal and macd < 0:
        analysis.append("⚡ MACD indicates bearish momentum")
    elif macd > macd_signal:
        analysis.append("⚡ MACD bullish crossover - momentum improving")
    else:
        analysis.append("⚡ MACD bearish crossover - momentum weakening")
    
    # Volume analysis
    if volume_ratio > 2:
        analysis.append("🔥 Exceptional volume surge confirms strong conviction")
    elif volume_ratio > 1.5:
        analysis.append("📊 High volume validates price movement")
    elif volume_ratio < 0.5:
        analysis.append("📊 Below-average volume suggests weak conviction")
    else:
        analysis.append("📊 Normal volume levels")
    
    # Market cap context
    market_cap = info.get('marketCap', 0)
    if market_cap:
        if market_cap > 200e9:  # > 200B
            analysis.append("🏢 Large-cap stability with lower volatility expected")
        elif market_cap > 10e9:  # > 10B
            analysis.append("🏢 Mid-cap stock with balanced growth-stability profile")
        else:
            analysis.append("🏢 Small-cap stock with higher growth potential and volatility")
    
    return analysis

def get_technical_analysis_report(data, info, symbol):
    """
    Combines technical indicators and market analysis into a structured report.
    Returns a dictionary suitable for JSON serialization.
    """
    # Calculate indicators
    df = calculate_technical_indicators(data)
    
    # Generate insights
    insights = generate_market_analysis(df, info, symbol)
    
    # Get latest metrics
    latest = df.iloc[-1]
    
    metrics = {
        "current_price": latest['Close'],
        "rsi": latest.get('RSI'),
        "macd": latest.get('MACD'),
        "sma_20": latest.get('SMA_20'),
        "sma_50": latest.get('SMA_50'),
        "bb_upper": latest.get('BB_upper'),
        "bb_lower": latest.get('BB_lower'),
        "volume": latest['Volume'],
        "volume_ratio": latest.get('Volume_ratio')
    }
    
    # Replace NaN with None for JSON compatibility
    metrics = {k: (None if pd.isna(v) else v) for k, v in metrics.items()}
    
    return {
        "symbol": symbol,
        "insights": insights,
        "metrics": metrics
    }
