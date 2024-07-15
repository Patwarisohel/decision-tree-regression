import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker):
    stock_data = yf.download(ticker, start='2020-01-01', end='2023-01-01')
    return stock_data

data = fetch_stock_data('AAPL')
print(data.head())

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

def train_model(data):
    data = data[['Open', 'High', 'Low', 'Close']]
    data['Prediction'] = data['Close'].shift(-1)
    data = data.dropna()
    
    X = data[['Open', 'High', 'Low', 'Close']].values
    y = data['Prediction'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

model, X_train, X_test, y_train, y_test = train_model(data)

from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objs as go
import numpy as np

def evaluate_model(model, X_train, X_test, y_train, y_test):
    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)
    
    mse_train = mean_squared_error(y_train, predictions_train)
    r2_train = r2_score(y_train, predictions_train)
    
    mse_test = mean_squared_error(y_test, predictions_test)
    r2_test = r2_score(y_test, predictions_test)
    
    fig_train = go.Figure()
    fig_train.add_trace(go.Scatter(x=np.arange(len(y_train)), y=y_train, mode='lines', name='Actual Train'))
    fig_train.add_trace(go.Scatter(x=np.arange(len(predictions_train)), y=predictions_train, mode='lines', name='Predicted Train'))

    fig_test = go.Figure()
    fig_test.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_test, mode='lines', name='Actual Test'))
    fig_test.add_trace(go.Scatter(x=np.arange(len(predictions_test)), y=predictions_test, mode='lines', name='Predicted Test'))
    
    return mse_train, r2_train, mse_test, r2_test, fig_train, fig_test

mse_train, r2_train, mse_test, r2_test, fig_train, fig_test = evaluate_model(model, X_train, X_test, y_train, y_test)
print(f'Train Mean Squared Error: {mse_train}')
print(f'Train R^2 Score: {r2_train}')
print(f'Test Mean Squared Error: {mse_test}')
print(f'Test R^2 Score: {r2_test}')
fig_train.show()
fig_test.show()

def predict_future(model, ticker):
    future_data = yf.download(ticker, start='2023-01-01', end='2024-07-01')
    future_data = future_data[['Open', 'High', 'Low', 'Close']]
    
    predictions = model.predict(future_data.values)
    
    future_data['Predicted Close'] = predictions
    
    return future_data

future_predictions = predict_future(model, 'AAPL')

fig_combined = go.Figure()
fig_combined.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Close'))
fig_combined.add_trace(go.Scatter(x=future_predictions.index, y=future_predictions['Predicted Close'], mode='lines', name='Predicted Future Close'))

fig_combined.show()

import streamlit as st

def main():
    st.title('Stock Price Prediction App')
    
    st.subheader('Stock Data')
    ticker = st.text_input('Enter Stock Ticker', 'AAPL')
    data = fetch_stock_data(ticker)
    st.write(data.tail())

    st.subheader('Model Training and Evaluation')
    model, X_train, X_test, y_train, y_test = train_model(data)
    mse_train, r2_train, mse_test, r2_test, fig_train, fig_test = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    st.write(f'Train Mean Squared Error: {mse_train}')
    st.write(f'Train R^2 Score: {r2_train}')
    st.write(f'Test Mean Squared Error: {mse_test}')
    st.write(f'Test R^2 Score: {r2_test}')
    
    st.plotly_chart(fig_train)
    st.plotly_chart(fig_test)

    st.subheader('Predict Future Prices')
    future_predictions = predict_future(model, ticker)
    
    fig_combined = go.Figure()
    fig_combined.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Close'))
    fig_combined.add_trace(go.Scatter(x=future_predictions.index, y=future_predictions['Predicted Close'], mode='lines', name='Predicted Future Close'))
    
    st.plotly_chart(fig_combined)

if __name__ == '__main__':
    main()

