import sys
import os
import json

# Add the current directory to sys.path so we can import tools
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from tools.prediction import get_price_prediction
except ImportError:
    # If running from within tools directory or similar, adjust path
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from server.tools.prediction import get_price_prediction

def test_prediction():
    ticker = "ORCL"
    print(f"Testing prediction for {ticker}...")
    
    try:
        # Run prediction
        result_json = get_price_prediction(ticker)
        
        # Parse result
        result = json.loads(result_json)
        
        if "error" in result:
            print(f"Error returned: {result['error']}")
            if "traceback" in result:
                print(f"Traceback: {result['traceback']}")
            return

        print("\nPrediction successful!")
        print("-" * 50)
        print(f"Current Price: ${result.get('current_price', 'N/A')}")
        
        ensemble = result.get('ensemble_prediction', {})
        if isinstance(ensemble, dict):
            print(f"Ensemble Prediction: ${ensemble.get('prediction', 'N/A')}")
            print(f"Direction: {ensemble.get('direction', 'N/A')}")
            print(f"Confidence: {ensemble.get('consensus_strength', 'N/A')}%")
        
        print("\nIndividual Models:")
        models = result.get('individual_models', {})
        for name, data in models.items():
            if isinstance(data, dict) and 'error' not in data:
                print(f"- {name}: ${data.get('prediction', 'N/A')} ({data.get('direction', 'N/A')})")
            else:
                print(f"- {name}: Error or invalid data")

        print("-" * 50)
        # print("Full result keys:", json.dumps(result.keys(), indent=2))
        print("Full result:", json.dumps(result, indent=2))
        
        if 'all_model_results' in result:
            print("SUCCESS: 'all_model_results' key found in response.")
        else:
            print("FAILURE: 'all_model_results' key NOT found in response.")

    except Exception as e:
        print(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction()
