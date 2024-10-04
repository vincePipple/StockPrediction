from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
import pandas as pd
import uvicorn

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
import yfinance as yf
from joblib import load

from app.model import preprocess_stock_data, fetch_stock_data

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Stock Price Prediction API! In order to find today's closing price prediction, add /predict/{ticker} to the url."}

@app.get("/predict/{ticker}")
def predict_stock_price(ticker:str):
    """
    Stock price prediction model based on Random Forest algorithm
    
    Args:
        ticker: stock ticker about which predictions should be made

    Returns:
        prediction: predicted closing price of the stock
    
    """
    try:
        data = fetch_stock_data(ticker, period = '1y', interval = '1d')

        if data is None or data.empty:
            raise HTTPException(status_code=400, detail="Invalid stock ticker")
        
        data = preprocess_stock_data(data)
        data.reset_index(inplace=True)

        X = data[['MA10', 'Pct_Change']]
        y = data['Close']

        model = RandomForestRegressor()
        model.fit(X,y)

        X_latest = pd.DataFrame([X.iloc[-1]], columns=X.columns)

        prediction = model.predict(X_latest)
        print("PREDICTION: ", prediction)
        return {
            "ticker": ticker, 
            "predicted_close_price": prediction[0]
        }
    except Exception as e:
        return {"error": str(e)}