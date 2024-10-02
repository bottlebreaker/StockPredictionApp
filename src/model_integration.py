# model_integration.py

from StockPredictionApp.src.lstm_model import predict_lstm
from StockPredictionApp.src.xgboost_model import predict_xgboost


def combine_predictions(lstm_preds, xgb_preds, alpha=0.5):
    """
    Combines LSTM and XGBoost predictions.
    - `alpha`: weight for LSTM predictions (default: 0.5)
    """
    return alpha * lstm_preds + (1 - alpha) * xgb_preds

def predict_combined(lstm_model, xgb_model, X_lstm, X_xgb, days=10):
    """
    Predicts using both LSTM and XGBoost models for the next 'days' days and combines the results.
    """
    lstm_preds = predict_lstm(lstm_model, X_lstm, days=days)
    xgb_preds = predict_xgboost(xgb_model, X_xgb, days=days)
    
    # Combine predictions (simple average for this example, but can use weighted or other methods)
    combined_preds = [(l + x) / 2 for l, x in zip(lstm_preds, xgb_preds)]
    alpha = 0.7  # Set alpha dynamically
    combined_preds = combine_predictions(lstm_preds, xgb_preds, alpha)

    return combined_preds
