import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def fetch_stock_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Retrieving stock data

    Args:
        ticker: stock ticker about which predictions should be made
        period: the amount of time back data is retrieved
        interval: size of the time period
    """

    stock_data = yf.download(ticker, period=period, interval=interval)
    if stock_data.empty:
        return None
    
    stock_data.reset_index(inplace=True)

    return stock_data

def preprocess_stock_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates moving average with a time window of 10 days 
    Calculates percentage change for closing prices

    Args:
        data: Pandas dataframe with stock data
    """
    data = data.dropna()

    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['Pct_Change'] = data['Close'].pct_change()

    data = data.dropna()

    return data

def train_and_predict(data: pd.DataFrame) -> int:
    """
    Trains a machine learning model and makes a prediction for today's closing price

    Args:
        data: Pandas dataframe containing stock information
    """
    data = preprocess_stock_data(data)
    data.reset_index(inplace=True)

    X = data[['MA10', 'Pct_Change']]
    y = data['Close']

    model_one = RandomForestRegressor()
    model_two = LinearRegression()
    model_one.fit(X,y)
    model_two.fit(X,y)

    X_latest = pd.DataFrame([X.iloc[-1]], columns=X.columns)

    prediction_one = model_one.predict(X_latest)
    prediction_two = model_two.predict(X_latest)

    return prediction_one[0], prediction_two[0]