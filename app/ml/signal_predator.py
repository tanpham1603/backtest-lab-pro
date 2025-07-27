# app/ml/signal_predictor.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
class SignalPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200)
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)
    def feature_importances(self):
        return self.model.feature_importances_
