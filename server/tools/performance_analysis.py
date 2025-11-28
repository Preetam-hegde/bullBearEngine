import json
from .market_data import get_market_data
from .perfomance_metric import create_performance_metrics

def generate_performance_analysis(ticker: str, period: str = "1y", interval: str = "1d") -> str:
    """
    Fetches market data and generates performance metrics and a chart.
    
    Args:
        ticker (str): The stock ticker symbol.
        period (str): The data period. Default is "1y" for performance.
        interval (str): The data interval.
        
    Returns:
        str: JSON string containing metrics and Plotly figure, or JSON error message.
    """
    try:
        # get_market_data returns (DataFrame, info_dict) or JSON error string
        result = get_market_data(ticker, period, interval)
        
        if isinstance(result, str):
            return result
            
        data, info = result
        
        # Create the metrics and chart
        metrics, fig = create_performance_metrics(data, ticker)
        
        # Return JSON representation
        return json.dumps({
            "metrics": metrics,
            "chart": fig.to_json()
        })
    except Exception as e:
        return json.dumps({"error": str(e)})
