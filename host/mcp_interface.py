import sys
import os
import json

# Add parent directory to path to import server tools
# In a real MCP setup, this would be replaced by an MCP Client connecting via Stdio/SSE
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from server.tools.market_data import get_market_data as _get_market_data
from server.tools.market_curve import fit_market_curve as _fit_market_curve
from server.tools.sr_zones import detect_sr_zones as _detect_sr_zones

async def get_market_data_tool(ticker: str, period: str = "1mo", interval: str = "1d") -> str:
    """Tool to fetch market data."""
    # Simulate async network call
    return _get_market_data(ticker, period, interval)

async def fit_market_curve_tool(data: str) -> str:
    """Tool to analyze market trend."""
    try:
        data_list = json.loads(data)
        return _fit_market_curve(data_list)
    except Exception as e:
        return json.dumps({"error": str(e)})

async def detect_sr_zones_tool(data: str) -> str:
    """Tool to detect support and resistance zones."""
    try:
        data_list = json.loads(data)
        return _detect_sr_zones(data_list)
    except Exception as e:
        return json.dumps({"error": str(e)})
