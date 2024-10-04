from unittest.mock import patch
import pandas as pd
import os
import sys

# Add the parent directory to the sys.path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import preprocess_stock_data, fetch_stock_data

# Create mock stock data as a DataFrame for testing
mock_stock_data = pd.DataFrame({
    "Date": ["2024-10-01", "2024-10-02"],
    "Open": [100.0, 102.0],
    "High": [105.0, 107.0],
    "Low": [98.0, 100.0],
    "Close": [104.0, 106.0],
    "Volume": [3000000, 3500000],
})

@patch("model.yf.download") 
def test_fetch_stock_data_valid_ticker(mock_download):
    mock_download.return_value = mock_stock_data 
    result = fetch_stock_data("AAPL", period = '1y', interval = '1d') 
    assert not result.empty

@patch("model.yf.download")
def test_fetch_stock_data_invalid_ticker(mock_download):
    mock_download.return_value = pd.DataFrame
    result = fetch_stock_data("INVALID", period = '1y', interval = '1d')
    assert result is None  

def test_preprocess_stock_data():
    df = preprocess_stock_data(mock_stock_data)
    assert "Close" in df.columns 
    assert "MA10" in df.columns
    assert "Pct_Change" in df.columns
