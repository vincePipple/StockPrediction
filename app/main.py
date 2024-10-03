from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
import pandas as pd
import uvicorn

from fastapi import FastAPI
import yfinance as yf
from joblib import load

from model import fetch_stock_data, preprocess_stock_data

data = fetch_stock_data('ASML')
cleaned_data = preprocess_stock_data(data)

print(f"{cleaned_data}\n")

def train_stock_model(data: pd.DataFrame):

    X = data[['MA10', 'Pct_Change']]
    y = data['Close'].shift(-1)

    X, y = X[:-1], y[:-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    dump(model, 'data/model.joblib')
    print("Model trained and saved!")

train_stock_model(cleaned_data)


app = FastAPI()

model = load('data/model.joblib')

@app.get("/predict/{ticker}")
def predict_stock_price(ticker:str):

    data = yf.download(ticker, period = '10d', interval = '1d')
    data = preprocess_stock_data(data)

    X_latest = data[['MA10', 'Pct_Change']].iloc[-1].values.reshape(1, -1)

    prediction = model.predict(X_latest)

    return {"ticker": ticker, "predicted_close_price": prediction[0]}