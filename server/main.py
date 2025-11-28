from mcp.server.fastmcp import FastMCP
from tools.market_data import get_market_data
from tools.company_info import get_company_info
from tools.technical_analysis import get_technical_analysis_report
from tools.prediction import get_price_prediction
from tools.performance_analysis import generate_performance_analysis
from tools.visualize import generate_market_chart, generate_separate_charts
import json

# Initialize the MCP Server
mcp = FastMCP("BullBearEngine")

@mcp.tool()
def fetch_market_data(ticker: str, period: str = "1mo", interval: str = "1d") -> str:
    """
    Fetches historical market data for a given ticker.
    Returns a JSON data and info.
    """
    return get_market_data(ticker, period, interval)

@mcp.tool()
def get_stock_info(ticker: str) -> str:
    """
    Retrieves company profile, sector, industry, and key financial metrics.
    """
    return get_company_info(ticker)

@mcp.tool()
def get_technical_analysis(ticker: str, period: str = "1mo", interval: str = "1d") -> str:
    """
    Calculates technical indicators (RSI, MACD, BB) and generates AI analysis.
    """
    # We need to fetch data first, then pass it to the report generator
    # But the helper function in technical_analysis.py takes (data, info, symbol)
    # Let's see if we can refactor or just do the work here.
    # Actually, let's check technical_analysis.py again.
    # It has get_technical_analysis_report(data, info, symbol).
    # So we need to fetch data here.
    
    result = get_market_data(ticker, period, interval)
    if isinstance(result, str):
        return result
    data, info = result
    return json.dumps(get_technical_analysis_report(data, info, ticker))

@mcp.tool()
def predict_price(ticker: str, period: str = "5y", interval: str = "1d") -> str:
    """
    Predicts future price movements using Random Forest.
    """
    return get_price_prediction(ticker, period, interval)

@mcp.tool()
def analyze_performance(ticker: str, period: str = "1y", interval: str = "1d") -> str:
    """
    Analyzes historical performance (Sharpe, Volatility, Drawdown).
    """
    return generate_performance_analysis(ticker, period, interval)

@mcp.tool()
def show_market_data(ticker: str, period: str = "1mo", interval: str = "1d") -> str:
    """
    Generates a visual chart for the market data of a given ticker.
    Returns a JSON string of the Plotly figure.
    """
    return generate_market_chart(ticker, period, interval)

@mcp.tool()
def show_separate_charts(ticker: str, period: str = "1mo", interval: str = "1d") -> str:
    """
    Generates separate charts for Price, Volume, MACD, and RSI/Stoch.
    Returns a JSON string containing multiple Plotly figures.
    """
    return generate_separate_charts(ticker, period, interval)

if __name__ == "__main__":
    mcp.run()
