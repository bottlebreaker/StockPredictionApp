# xgboost_model.py

import xgboost as xgb
from sklearn.model_selection import train_test_split

def train_xgboost(X, y, test_size=0.2):
    """
    Trains an XGBoost model with GPU support.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'tree_method': 'gpu_hist',  # Using GPU for training
        'predictor': 'gpu_predictor',
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    }
    
    model = xgb.train(params, dtrain, evals=[(dtest, 'eval')], num_boost_round=100, early_stopping_rounds=10)
    return model

def predict_xgboost(model, X_test):
    """
    Predicts the stock prices using the trained XGBoost model.
    """
    dtest = xgb.DMatrix(X_test)
    return model.predict(dtest)
