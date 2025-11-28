import json
from .market_data import get_market_data
from .create_chart import create_advanced_chart, create_separate_charts

def generate_market_chart(ticker: str, period: str = "1mo", interval: str = "1d") -> str:
    """
    Fetches market data and generates a Plotly chart JSON.
    """
    try:
        result = get_market_data(ticker, period, interval)
        if isinstance(result, str):
            return result
        data, info = result
        fig = create_advanced_chart(data, ticker)
        return fig.to_json()
    except Exception as e:
        return json.dumps({"error": str(e)})

def generate_separate_charts(ticker: str, period: str = "1mo", interval: str = "1d") -> str:
    """
    Fetches market data and generates separate Plotly chart JSONs.
    """
    try:
        result = get_market_data(ticker, period, interval)
        if isinstance(result, str):
            return result
        data, info = result
        charts = create_separate_charts(data, ticker)
        # Convert all figures to JSON
        charts_json = {k: v.to_json() for k, v in charts.items()}
        return json.dumps(charts_json)
    except Exception as e:
        return json.dumps({"error": str(e)})
