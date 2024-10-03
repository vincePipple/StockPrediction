import yfinance as yf
import pandas as pd


def fetch_stock_data(ticker: str, period = '1y', intervals = '1d') -> pd.DataFrame:

    stock_data = yf.download(ticker, period=period, interval=intervals)
    stock_data.reset_index(inplace=True)

    return stock_data


def preprocess_stock_data(data: pd.DataFrame) -> pd.DataFrame:

    data = data.dropna()

    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['Pct_Change'] = data['Close'].pct_change()

    data = data.dropna()

    return data

