from joblib import dump
import pandas as pd

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException

from app.model import fetch_stock_data, train_and_predict

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
        data = fetch_stock_data(ticker, period = '5y', interval = '1d')

        if data is None or data.empty:
            raise HTTPException(status_code=400, detail="Invalid stock ticker")
        
        prediction_one, prediction_two = train_and_predict(data = data)

        return {
            "ticker": ticker, 
            "predicted closing price (using RandomForest)": prediction_one,
            "predicted closing price (using LinearRegression)": prediction_two
        }
    except Exception as e:
        return {"error": str(e)}