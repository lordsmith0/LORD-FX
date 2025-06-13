import streamlit as st
import pandas as pd
from alpha_vantage.foreignexchange import ForeignExchange
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- CONFIG ---
st.set_page_config(page_title="LORD SMITH AI Forex Signals", layout="centered")
st.title("📈 LORD SMITH Forex Signal AI")
st.subheader("AI-generated Buy/Sell/Hold signals using EUR/USD data")

# --- API KEY INPUT ---
API_KEY = st.text_input("🔑 Enter your Alpha Vantage API Key:", type="password")

if API_KEY:
    try:
        fx = ForeignExchange(key=API_KEY, output_format='pandas')
        data, _ = fx.get_currency_exchange_daily(from_symbol='EUR', to_symbol='USD', outputsize='full')
        data = data.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close'
        }).sort_index().dropna()

        data['MA_10'] = data['Close'].rolling(window=10).mean()
        data['Return'] = data['Close'].pct_change()
        data['Signal'] = data['Return'].shift(-1).apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        data.dropna(inplace=True)

        features = ['Close', 'MA_10']
        X = data[features]
        y = data['Signal']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        def get_latest_signal():
            X_live = X.tail(1)
            signal = model.predict(X_live)[0]
            confidence = model.predict_proba(X_live).max()
            return signal, confidence

        signal, confidence = get_latest_signal()
        st.success(f"📊 Latest Signal: **{'BUY' if signal == 1 else 'SELL' if signal == -1 else 'HOLD'}**")
        st.info(f"🤖 Model Confidence: **{confidence:.2f}**")

        st.line_chart(data[['Close', 'MA_10']].tail(100))

    except Exception as e:
        st.error(f"Error: {e}")
