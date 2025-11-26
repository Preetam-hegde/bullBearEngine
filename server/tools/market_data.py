import yfinance as yf
import pandas as pd
import json

def get_market_data(ticker: str, period: str = "1mo", interval: str = "1d") -> str:
    """
    Fetches historical market data for a given ticker.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL", "MSFT").
        period (str): The data period to download (e.g., "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max").
        interval (str): The data interval (e.g., "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo").
        
    Returns:
        str: JSON string of the market data.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        
        if hist.empty:
            return json.dumps({"error": f"No data found for ticker {ticker}"})
            
        # Reset index to make Date a column and convert to string for JSON serialization
        hist.reset_index(inplace=True)
        hist['Date'] = hist['Date'].astype(str)
        
        return hist.to_json(orient="records")
    except Exception as e:
        return json.dumps({"error": str(e)})
