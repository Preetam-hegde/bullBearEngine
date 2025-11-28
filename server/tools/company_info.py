import json
from .market_data import get_market_data

def get_company_info(ticker: str) -> str:
    """
    Fetches and formats company information.
    """
    try:
        # We can use get_market_data to get info, or just yfinance directly.
        # get_market_data returns (data, info)
        result = get_market_data(ticker, period="1d") # Period doesn't matter for info
        if isinstance(result, str):
            return result
            
        _, info = result
        
        company_details = {
            "Company Name": info.get('longName', 'N/A'),
            "Sector": info.get('sector', 'N/A'),
            "Industry": info.get('industry', 'N/A'),
            "Country": info.get('country', 'N/A'),
            "Website": info.get('website', 'N/A'),
            "Employees": info.get('fullTimeEmployees', 'N/A'),
            "Summary": info.get('longBusinessSummary', 'N/A')
        }
        
        financial_metrics = {
            "P/E Ratio": info.get('trailingPE', 'N/A'),
            "Forward P/E": info.get('forwardPE', 'N/A'),
            "PEG Ratio": info.get('pegRatio', 'N/A'),
            "Price to Book": info.get('priceToBook', 'N/A'),
            "Dividend Yield": info.get('dividendYield', 'N/A'),
            "Beta": info.get('beta', 'N/A'),
            "52W High": info.get('fiftyTwoWeekHigh', 'N/A'),
            "52W Low": info.get('fiftyTwoWeekLow', 'N/A'),
            "Market Cap": info.get('marketCap', 'N/A')
        }
        
        return json.dumps({
            "symbol": ticker,
            "details": company_details,
            "financials": financial_metrics
        })
    except Exception as e:
        return json.dumps({"error": str(e)})
