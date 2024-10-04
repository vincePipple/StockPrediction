from fastapi.testclient import TestClient

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  

# Create a TestClient to interact with the FastAPI app during testing
client = TestClient(app)

def test_root():
    response = client.get("/") 
    assert response.status_code == 200  
    assert response.json() == {"message": "Welcome to the Stock Price Prediction API! In order to find today's closing price prediction, add /predict/{ticker} to the url."}  # Check welcome message

def test_predict_valid_ticker():
    response = client.get("/predict/AAPL") 
    assert response.status_code == 200  
    
    json_response = response.json()
    assert "predicted_close_price" in json_response  
    assert isinstance(json_response["predicted_close_price"], float) 

def test_predict_invalid_ticker():
    response = client.get("/predict/INVALID")  
    json_response = response.json()
    assert json_response == {"error": "400: Invalid stock ticker"} 