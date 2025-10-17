import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")


feature_columns = ['Open-Close', 'High-Low']


data = pd.read_csv("NSE-TATAGLOBAL.csv")
data['Daily Change %'] = data['Close'].pct_change() * 100
avg_volatility = data['Daily Change %'].abs().mean() / 100


def predict_stock(open_price, close_price, high_price, low_price):
    open_close = open_price - close_price
    high_low = high_price - low_price
    X_new = pd.DataFrame([[open_close, high_low]], columns=feature_columns)
    X_new_scaled = scaler.transform(X_new)
    pred = model.predict(X_new_scaled)
    confidence = max(model.predict_proba(X_new_scaled)[0]) * 100 if hasattr(model, "predict_proba") else 50
    return (1 if pred[0]==1 else -1), confidence

def predict_next_five_days(open_price, close_price, high_price, low_price):
    results = []
    current_open, current_close, current_high, current_low = open_price, close_price, high_price, low_price
    for day in range(1,6):
        direction, confidence = predict_stock(current_open, current_close, current_high, current_low)
        change = np.random.uniform(0.5*avg_volatility, 1.5*avg_volatility)
        if direction == 1:  # UP
            current_open = current_close
            current_close *= (1 + change)
            current_high = current_close * (1 + 0.6*change)
            current_low = current_close * (1 - 0.6*change)
            results.append((day, "UP 📈", round(current_close,2), round(confidence,2), current_low, current_high))
        else:  # DOWN
            current_open = current_close
            current_close *= (1 - change)
            current_high = current_close * (1 + 0.6*change)
            current_low = current_close * (1 - 0.6*change)
            results.append((day, "DOWN 📉", round(current_close,2), round(confidence,2), current_low, current_high))
    df_results = pd.DataFrame(results, columns=["Day", "Prediction", "Predicted Close (INR)", "Confidence (%)", "Low", "High"])
    return df_results


st.set_page_config(page_title="5-Day Stock Prediction", layout="wide", page_icon="📈")
st.title("💹 TATAGLOBAL 5-Day Stock Prediction ")

st.write("Average daily volatility: {:.2f}%".format(avg_volatility*100))


input_col, output_col = st.columns([1,2])  # Left column smaller, right column bigger

with input_col:
    st.subheader("Enter Today's Stock Prices")
    open_price = st.number_input("Opening Price (INR)", min_value=0.0, format="%.2f")
    close_price = st.number_input("Closing Price (INR)", min_value=0.0, format="%.2f")
    high_price = st.number_input("High Price (INR)", min_value=0.0, format="%.2f")
    low_price = st.number_input("Low Price (INR)", min_value=0.0, format="%.2f")
    predict_button = st.button("Predict Next 5 Days")

with output_col:
    if predict_button:
        df_pred = predict_next_five_days(open_price, close_price, high_price, low_price)
        st.subheader("Predicted Movement for Next 5 Days")
        st.dataframe(df_pred[["Day","Prediction","Predicted Close (INR)","Confidence (%)"]])

        
        plt.style.use('dark_background')
        fig, ax1 = plt.subplots(figsize=(12,7))
        plt.subplots_adjust(top=0.9, bottom=0.15)

      
        ax1.plot(df_pred['Day'], df_pred['Predicted Close (INR)'], marker='o', linewidth=2, color='#00ffff', label='Predicted Close')

      
        for i in range(len(df_pred)):
            color = '#39ff14' if "UP" in df_pred['Prediction'][i] else '#ff073a'
            ax1.scatter(df_pred['Day'][i], df_pred['Predicted Close (INR)'][i], color=color, s=150, edgecolors='white', linewidth=1.5)

       
        ax1.fill_between(df_pred['Day'], df_pred['Low'], df_pred['High'], color='#6e0dd0', alpha=0.3, label='Predicted Volatility Range')

        ax1.set_xlabel("Future Days", fontsize=13)
        ax1.set_ylabel("Predicted Closing Price (INR)", fontsize=13)
        ax1.set_title("5-Day Stock Prediction with Confidence & Volatility (TATAGLOBAL)", fontsize=16, pad=15)
        ax1.legend(loc="upper left")
        ax1.grid(True, linestyle='--', alpha=0.5)

        
        ax2 = ax1.twinx()
        colors = ['#39ff14' if c >= 75 else '#00bfff' if c >= 50 else '#ff073a' for c in df_pred['Confidence (%)']]
        ax2.bar(df_pred['Day'], df_pred['Confidence (%)'], color=colors, alpha=0.5, width=0.4, label='Confidence (%)')
        ax2.set_ylabel("Model Confidence (%)", fontsize=13)
        ax2.set_ylim(0,100)
        ax2.legend(loc="upper right")

        st.pyplot(fig)
