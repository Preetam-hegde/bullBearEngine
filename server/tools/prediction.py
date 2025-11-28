import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MultiModelPricePredictor:
    def __init__(self):
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.models = {}
        self.initialize_models()
        
    def initialize_models(self):
        """Initialize multiple models optimized for financial time series"""
        self.models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': XGBRegressor(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'LightGBM': LGBMRegressor(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            ),
            'AdaBoost': AdaBoostRegressor(
                n_estimators=50,
                learning_rate=0.5,
                random_state=42
            ),
            'Ridge Regression': Ridge(
                alpha=1.0,
                random_state=42
            ),
            'Lasso Regression': Lasso(
                alpha=0.1,
                random_state=42,
                max_iter=2000
            ),
            'ElasticNet': ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                random_state=42,
                max_iter=2000
            ),
            'SVR': SVR(
                kernel='rbf',
                C=100,
                gamma='scale',
                epsilon=0.1
            ),
            'KNN': KNeighborsRegressor(
                n_neighbors=10,
                weights='distance',
                metric='euclidean'
            )
        }
        
    def prepare_ml_features(self, data):
        """Prepare comprehensive features for machine learning"""
        df = data.copy()
        
        # Ensure we have technical indicators
        from .technical_analysis import calculate_technical_indicators
        if 'RSI' not in df.columns:
            df = calculate_technical_indicators(df)
            
        # Returns and momentum
        df['Returns'] = df['Close'].pct_change()
        df['Returns_5d'] = df['Close'].pct_change(5)
        df['Returns_10d'] = df['Close'].pct_change(10)
        df['Returns_20d'] = df['Close'].pct_change(20)
        df['Returns_60d'] = df['Close'].pct_change(60)
        
        # Logarithmic returns (more stable)
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Lag features
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
            df[f'Returns_lag_{lag}'] = df['Returns'].shift(lag)
            df[f'High_lag_{lag}'] = df['High'].shift(lag)
            df[f'Low_lag_{lag}'] = df['Low'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'Close_mean_{window}'] = df['Close'].rolling(window).mean()
            df[f'Close_std_{window}'] = df['Close'].rolling(window).std()
            df[f'Volume_mean_{window}'] = df['Volume'].rolling(window).mean()
            df[f'Volume_std_{window}'] = df['Volume'].rolling(window).std()
            df[f'High_mean_{window}'] = df['High'].rolling(window).mean()
            df[f'Low_mean_{window}'] = df['Low'].rolling(window).mean()
            
        # Exponential moving averages
        for span in [12, 26, 50]:
            df[f'EMA_{span}'] = df['Close'].ewm(span=span).mean()
            
        # Price position relative to moving averages
        for window in [20, 50, 100, 200]:
            sma_col = f'SMA_{window}' if f'SMA_{window}' in df.columns else f'Close_mean_{window}'
            if sma_col in df.columns:
                df[f'Price_vs_SMA{window}'] = (df['Close'] - df[sma_col]) / df[sma_col] * 100
        
        # Volatility features
        for window in [10, 20, 30, 60]:
            df[f'Volatility_{window}d'] = df['Returns'].rolling(window).std()
            df[f'Volatility_{window}d_zscore'] = (
                df[f'Volatility_{window}d'] - df[f'Volatility_{window}d'].rolling(100).mean()
            ) / df[f'Volatility_{window}d'].rolling(100).std()
        
        # Market regime indicators
        df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['Price_Range_20d'] = df['High_Low_Range'].rolling(20).mean()
        df['Price_Range_50d'] = df['High_Low_Range'].rolling(50).mean()
        
        # True Range and Average True Range
        df['True_Range'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        df['ATR_14'] = df['True_Range'].rolling(14).mean()
        df['ATR_28'] = df['True_Range'].rolling(28).mean()
        
        # Volume patterns
        df['Volume_trend'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['Volume_volatility'] = df['Volume'].rolling(20).std() / df['Volume'].rolling(20).mean()
        df['Volume_price_trend'] = df['Volume'] * df['Returns']
        
        # Price momentum
        for period in [10, 20, 50, 100]:
            df[f'Momentum_{period}d'] = df['Close'] - df['Close'].shift(period)
            df[f'ROC_{period}d'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period) * 100
        
        # Bollinger Bands
        for window in [20, 50]:
            rolling_mean = df['Close'].rolling(window).mean()
            rolling_std = df['Close'].rolling(window).std()
            df[f'BB_upper_{window}'] = rolling_mean + (rolling_std * 2)
            df[f'BB_lower_{window}'] = rolling_mean - (rolling_std * 2)
            df[f'BB_width_{window}'] = (df[f'BB_upper_{window}'] - df[f'BB_lower_{window}']) / rolling_mean
            df[f'BB_position_{window}'] = (df['Close'] - df[f'BB_lower_{window}']) / (df[f'BB_upper_{window}'] - df[f'BB_lower_{window}'])
        
        # Stochastic Oscillator
        for window in [14, 28]:
            low_min = df['Low'].rolling(window).min()
            high_max = df['High'].rolling(window).max()
            df[f'Stochastic_{window}'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
        
        # Williams %R
        for window in [14, 28]:
            high_max = df['High'].rolling(window).max()
            low_min = df['Low'].rolling(window).min()
            df[f'Williams_R_{window}'] = -100 * (high_max - df['Close']) / (high_max - low_min)
        
        # Commodity Channel Index (CCI)
        for window in [20, 40]:
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = tp.rolling(window).mean()
            mad = tp.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean())
            df[f'CCI_{window}'] = (tp - sma_tp) / (0.015 * mad)
        
        # On-Balance Volume (OBV)
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['OBV_EMA'] = df['OBV'].ewm(span=20).mean()
        
        # Accumulation/Distribution Line
        clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        clv = clv.fillna(0)
        df['AD_Line'] = (clv * df['Volume']).cumsum()
        
        # Money Flow Index (MFI)
        for window in [14, 28]:
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            raw_mf = tp * df['Volume']
            pos_mf = raw_mf.where(tp > tp.shift(1), 0).rolling(window).sum()
            neg_mf = raw_mf.where(tp < tp.shift(1), 0).rolling(window).sum()
            mfi_ratio = pos_mf / neg_mf
            df[f'MFI_{window}'] = 100 - (100 / (1 + mfi_ratio))
        
        # Day of week and month effects
        df['DayOfWeek'] = pd.to_datetime(df.index).dayofweek
        df['DayOfMonth'] = pd.to_datetime(df.index).day
        df['Month'] = pd.to_datetime(df.index).month
        df['Quarter'] = pd.to_datetime(df.index).quarter
        
        # Trend strength
        df['Trend_20d'] = df['Close'].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        df['Trend_50d'] = df['Close'].rolling(50).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        
        return df
    
    def select_features(self, df):
        """Select relevant features for modeling"""
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits',
                       'Returns', 'Returns_5d', 'Returns_10d', 'Returns_20d', 'Returns_60d',
                       'Log_Returns', 'High_Low_Range', 'True_Range']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
    
    def walk_forward_validation(self, df, feature_cols, model_name, n_splits=5):
        """Perform walk-forward time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        predictions = []
        actuals = []
        
        X = df[feature_cols].ffill().bfill()
        y = df['Close'].shift(-1)[:-1]
        X = X[:-1]
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            train_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
            test_mask = ~(X_test.isna().any(axis=1) | y_test.isna())
            
            X_train_clean = X_train[train_mask]
            y_train_clean = y_train[train_mask]
            X_test_clean = X_test[test_mask]
            y_test_clean = y_test[test_mask]
            
            if len(X_train_clean) < 50 or len(X_test_clean) < 5:
                continue
                
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_clean)
            X_test_scaled = scaler.transform(X_test_clean)
            
            model = self.models[model_name]
            model.fit(X_train_scaled, y_train_clean)
            
            score = model.score(X_test_scaled, y_test_clean)
            scores.append(score)
            
            preds = model.predict(X_test_scaled)
            predictions.extend(preds)
            actuals.extend(y_test_clean.values)
        
        # Calculate additional metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        # Directional accuracy
        direction_correct = np.sum(np.sign(predictions[1:] - actuals[:-1]) == 
                                   np.sign(actuals[1:] - actuals[:-1]))
        directional_accuracy = direction_correct / (len(actuals) - 1) * 100 if len(actuals) > 1 else 0
        
        return {
            'r2_mean': np.mean(scores),
            'r2_std': np.std(scores),
            'mse': mse,
            'mae': mae,
            'mape': mape,
            'directional_accuracy': directional_accuracy
        }
    
    def train_and_predict_all_models(self, data):
        """Train all models and generate predictions with ensemble"""
        df = self.prepare_ml_features(data)
        df = df.dropna()
        
        if len(df) < 200:
            return {"error": "Insufficient data for prediction (need at least 200 data points)"}
        
        feature_cols = self.select_features(df)
        
        if len(feature_cols) < 10:
            return {"error": "Not enough features generated"}
        
        X = df[feature_cols].ffill().bfill()
        y = df['Close'].shift(-1)
        
        X_train_full = X[:-1]
        y_train_full = y[:-1]
        X_pred = X.iloc[-1:]
        
        mask = ~(X_train_full.isna().any(axis=1) | y_train_full.isna())
        X_train_full = X_train_full[mask]
        y_train_full = y_train_full[mask]
        
        if len(X_train_full) < 100:
            return {"error": "Insufficient valid training data"}
        
        current_price = df['Close'].iloc[-1]
        results = {}
        predictions = []
        weights = []
        
        # Train and predict with each model
        for model_name, model in self.models.items():
            try:
                # Walk-forward validation
                cv_metrics = self.walk_forward_validation(df, feature_cols, model_name, n_splits=5)
                
                # Train final model
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train_full)
                model.fit(X_train_scaled, y_train_full)
                
                # Predict
                X_pred_scaled = scaler.transform(X_pred)
                prediction = model.predict(X_pred_scaled)[0]
                
                # Calculate prediction confidence based on directional accuracy
                confidence = cv_metrics['directional_accuracy']
                
                change_pct = ((prediction - current_price) / current_price) * 100
                direction = "UP" if prediction > current_price else "DOWN"
                
                results[model_name] = {
                    'prediction': float(prediction),
                    'predicted_change_pct': float(change_pct),
                    'direction': direction,
                    'confidence_score': float(confidence),
                    'validation_metrics': {
                        'r2_score': float(cv_metrics['r2_mean']),
                        'r2_std': float(cv_metrics['r2_std']),
                        'mae': float(cv_metrics['mae']),
                        'mape': float(cv_metrics['mape']),
                        'directional_accuracy': float(cv_metrics['directional_accuracy'])
                    }
                }
                
                # Collect for ensemble (weight by directional accuracy)
                if cv_metrics['directional_accuracy'] > 45:  # Only use models better than random
                    predictions.append(prediction)
                    weights.append(cv_metrics['directional_accuracy'])
                    
            except Exception as e:
                results[model_name] = {'error': str(e)}
        
        # Ensemble prediction (weighted average)
        if predictions:
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            ensemble_pred = np.average(predictions, weights=weights)
            ensemble_std = np.sqrt(np.average((predictions - ensemble_pred)**2, weights=weights))
            
            ensemble_change_pct = ((ensemble_pred - current_price) / current_price) * 100
            ensemble_direction = "UP" if ensemble_pred > current_price else "DOWN"
            
            # Consensus strength
            up_count = sum(1 for p in predictions if p > current_price)
            consensus = max(up_count, len(predictions) - up_count) / len(predictions) * 100
            
            ensemble = {
                'prediction': float(ensemble_pred),
                'confidence_interval_95': {
                    'lower': float(ensemble_pred - 1.96 * ensemble_std),
                    'upper': float(ensemble_pred + 1.96 * ensemble_std)
                },
                'predicted_change_pct': float(ensemble_change_pct),
                'direction': ensemble_direction,
                'consensus_strength': float(consensus),
                'num_models_used': len(predictions),
                'prediction_std': float(ensemble_std)
            }
        else:
            ensemble = {'error': 'No reliable models for ensemble'}
        
        # Find best performing model
        valid_models = {k: v for k, v in results.items() if 'error' not in v}
        if valid_models:
            best_model = max(valid_models.items(), 
                           key=lambda x: x[1]['validation_metrics']['directional_accuracy'])
            best_model_name = best_model[0]
            best_model_accuracy = best_model[1]['validation_metrics']['directional_accuracy']
        else:
            best_model_name = None
            best_model_accuracy = 0
        
        return {
            'current_price': float(current_price),
            'ensemble_prediction': ensemble,
            'individual_models': results,
            'all_model_results': results,
            'best_model': {
                'name': best_model_name,
                'directional_accuracy': float(best_model_accuracy)
            },
            'features_used': len(feature_cols),
            'training_samples': len(X_train_full),
            'timestamp': datetime.now().isoformat(),
            'disclaimer': 'These predictions are for informational purposes only. Past performance does not guarantee future results. Do not make investment decisions based solely on these predictions.'
        }

def get_price_prediction(ticker: str, period: str = "5y", interval: str = "1d") -> str:
    """
    Generates price predictions using multiple ML models optimized for stocks.
    Returns ensemble prediction and individual model results.
    """
    try:
        from .market_data import get_market_data
        
        result = get_market_data(ticker, period, interval)
        if isinstance(result, str):
            return result
            
        data, info = result
        
        predictor = MultiModelPricePredictor()
        result = predictor.train_and_predict_all_models(data)
        
        if "error" in result:
            return json.dumps(result)
            
        result["symbol"] = ticker
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": str(e.__traceback__)})