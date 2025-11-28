import sys
import os
import json

# Add parent directory to path to import server tools
# In a real MCP setup, this would be replaced by an MCP Client connecting via Stdio/SSE
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from server.tools.market_data import get_market_data as _get_market_data
from server.tools.visualize import generate_market_chart as _generate_market_chart, generate_separate_charts as _generate_separate_charts
from server.tools.performance_analysis import generate_performance_analysis as _generate_performance_analysis
from server.tools.technical_analysis import get_technical_analysis_report as _get_technical_analysis_report
from server.tools.prediction import get_price_prediction as _get_price_prediction
from server.tools.company_info import get_company_info as _get_company_info

async def get_market_data_tool(ticker: str, period: str = "1mo", interval: str = "1d") -> str:
    """Tool to fetch market data."""
    # Simulate async network call
    return _get_market_data(ticker, period, interval)

async def get_technical_analysis_tool(ticker: str, period: str = "1y", interval: str = "1d") -> str:
    """Tool to generate comprehensive technical analysis report."""
    try:
        # We need to fetch data first
        result = _get_market_data(ticker, period, interval)
        if isinstance(result, str): # Error message
            return result
        
        data, info = result
        report = _get_technical_analysis_report(data, info, ticker)
        return json.dumps(report)
    except Exception as e:
        return json.dumps({"error": str(e)})

async def get_price_prediction_tool(ticker: str, period: str = "5y", interval: str = "1d") -> str:
    """Tool to generate price prediction."""
    return _get_price_prediction(ticker, period, interval)

async def get_company_info_tool(ticker: str) -> str:
    """Tool to get company information."""
    return _get_company_info(ticker)

async def generate_market_chart_tool(ticker: str, period: str = "1mo", interval: str = "1d") -> str:
    """Tool to generate market chart."""
    return _generate_market_chart(ticker, period, interval)

async def generate_separate_charts_tool(ticker: str, period: str = "1mo", interval: str = "1d") -> str:
    """Tool to generate separate market charts."""
    return _generate_separate_charts(ticker, period, interval)

async def generate_performance_analysis_tool(ticker: str, period: str = "1y", interval: str = "1d") -> str:
    """Tool to generate performance analysis."""
    return _generate_performance_analysis(ticker, period, interval)
