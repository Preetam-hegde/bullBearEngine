import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import json

def detect_sr_zones(data: list, n_clusters: int = 5) -> str:
    """
    Detects Support and Resistance zones using K-Means clustering.
    
    Args:
        data (list): List of dictionaries containing market data (must have 'High' and 'Low' prices).
        n_clusters (int): Number of zones to detect.
        
    Returns:
        str: JSON string containing the support and resistance levels.
    """
    try:
        df = pd.DataFrame(data)
        if 'High' not in df.columns or 'Low' not in df.columns:
            return json.dumps({"error": "Data must contain 'High' and 'Low' prices"})
            
        # Combine High and Low prices to find price concentrations
        prices = np.concatenate([df['High'].values, df['Low'].values]).reshape(-1, 1)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        kmeans.fit(prices)
        
        centers = kmeans.cluster_centers_.flatten()
        centers.sort()
        
        result = {
            "sr_levels": centers.tolist()
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})
