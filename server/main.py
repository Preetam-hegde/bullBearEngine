from mcp.server.fastmcp import FastMCP
from tools.market_data import get_market_data
from tools.market_curve import fit_market_curve
from tools.sr_zones import detect_sr_zones
import json

# Initialize the MCP Server
mcp = FastMCP("BullBearEngine")

@mcp.tool()
def fetch_market_data(ticker: str, period: str = "1mo", interval: str = "1d") -> str:
    """
    Fetches historical market data for a given ticker.
    Returns a JSON string of the data.
    """
    return get_market_data(ticker, period, interval)

@mcp.tool()
def analyze_market_trend(data: str) -> str:
    """
    Analyzes the market trend (uptrend, downtrend, ranging) from market data.
    Input data should be a JSON string of the market data list.
    """
    try:
        data_list = json.loads(data)
        return fit_market_curve(data_list)
    except Exception as e:
        return json.dumps({"error": f"Invalid data format: {str(e)}"})

@mcp.tool()
def find_sr_levels(data: str) -> str:
    """
    Detects Support and Resistance zones from market data.
    Input data should be a JSON string of the market data list.
    """
    try:
        data_list = json.loads(data)
        return detect_sr_zones(data_list)
    except Exception as e:
        return json.dumps({"error": f"Invalid data format: {str(e)}"})

if __name__ == "__main__":
    mcp.run()
