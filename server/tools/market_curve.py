import numpy as np
from scipy.stats import linregress
import pandas as pd
import json

def fit_market_curve(data: list) -> str:
    """
    Analyzes the market trend using linear regression.
    
    Args:
        data (list): List of dictionaries containing market data (must have 'Close' prices).
        
    Returns:
        str: JSON string describing the trend (uptrend, downtrend, ranging) and slope.
    """
    try:
        df = pd.DataFrame(data)
        if 'Close' not in df.columns:
            return json.dumps({"error": "Data must contain 'Close' prices"})
            
        prices = df['Close'].values
        x = np.arange(len(prices))
        slope, intercept, r_value, p_value, std_err = linregress(x, prices)
        
        trend = "ranging"
        if slope > 0.1: # Thresholds can be adjusted
            trend = "uptrend"
        elif slope < -0.1:
            trend = "downtrend"
            
        result = {
            "trend": trend,
            "slope": slope,
            "r_squared": r_value**2,
            "std_err": std_err
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})
