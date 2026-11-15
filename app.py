import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from statsmodels.tsa.statespace.sarimax import SARIMAX


st.set_page_config(
    page_title="StockScope",
    page_icon="📈",
    layout="wide"
)

st.title("📈 StockScope")
st.write(
    "Forecast a stock's next closing price and compare "
    "ARIMA with a simple previous-price baseline."
)


# Download stock prices
@st.cache_data(ttl=3600)
def load_data(ticker):
    data = yf.download(
        ticker,
        start="2019-01-01",
        end="2026-10-01",
        auto_adjust=False,
        multi_level_index=False,
        progress=False
    )

    if data is None or data.empty:
        raise ValueError("No data was downloaded. Please try again later.")

    stock = data[["Close"]].reset_index()
    stock.columns = ["date", "close"]

    return (
        stock.dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )


# Fit ARIMA and compare it with the baseline
@st.cache_data
def make_predictions(stock):
    train = stock.iloc[:-60].copy()
    test = stock.iloc[-60:].copy()

    model = SARIMAX(
        train["close"].values,
        order=(1, 1, 0),
        trend="n"
    ).fit(disp=False)

    model_predictions = []
    naive_predictions = []

    last_price = train["close"].iloc[-1]

    for actual_price in test["close"]:
        # Predict before seeing the actual price
        prediction = float(
            np.asarray(model.forecast(1))[0]
        )

        model_predictions.append(prediction)
        naive_predictions.append(last_price)

        # Update after the actual price becomes available
        model = model.append(
            [float(actual_price)],
            refit=False
        )

        last_price = actual_price

    test["ARIMA"] = model_predictions
    test["Previous close"] = naive_predictions

    model_mae = np.mean(
        np.abs(test["close"] - test["ARIMA"])
    )

    naive_mae = np.mean(
        np.abs(test["close"] - test["Previous close"])
    )

    # Use all available data for the next-session forecast
    full_model = SARIMAX(
        stock["close"].values,
        order=(1, 1, 0),
        trend="n"
    ).fit(disp=False)

    next_prediction = float(
        np.asarray(full_model.forecast(1))[0]
    )

    return test, model_mae, naive_mae, next_prediction


# Stock selection
ticker = st.selectbox(
    "Choose a stock",
    ["AAPL", "MSFT", "AMZN", "GOOGL", "DIS", "NFLX"]
)

st.caption(
    "Historical data: January 2019 through September 2026. "
    "The final 60 trading days are used for testing."
)


# Run the calculations
try:
    with st.spinner("Loading prices and making forecasts..."):
        stock = load_data(ticker)

        if len(stock) < 160:
            raise ValueError("Not enough data to fit and test the model.")

        test, model_mae, naive_mae, next_prediction = (
            make_predictions(stock)
        )

except Exception as error:
    st.error(f"Unable to complete the forecast: {error}")
    st.stop()


# Organize the app into tabs
forecast_tab, comparison_tab, about_tab = st.tabs(
    ["Forecast", "Model Comparison", "About"]
)


with forecast_tab:
    st.subheader(f"{ticker}: Next Trading Session")

    last_price = stock["close"].iloc[-1]
    last_date = stock["date"].iloc[-1]

    st.caption(
        f"Forecast for the next trading session after "
        f"{last_date:%B %d, %Y}, not necessarily tomorrow."
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Last closing price",
        f"${last_price:,.2f}"
    )

    col2.metric(
        "Predicted next closing price",
        f"${next_prediction:,.2f}"
    )

    st.subheader("Historical Closing Prices")

    st.line_chart(
        stock.set_index("date")[["close"]]
        .rename(columns={"close": "Closing price ($)"})
    )

    st.info(
        "This model estimates the next closing price. "
        "Incorrect predictions could lead to financial losses "
        "if used for trading."
    )


with comparison_tab:
    st.subheader("Performance on the Final 60 Trading Days")

    col1, col2 = st.columns(2)

    col1.metric(
        "ARIMA MAE",
        f"${model_mae:.2f}"
    )

    col2.metric(
        "Previous-Price Baseline MAE",
        f"${naive_mae:.2f}"
    )

    st.caption(
        "MAE is the average absolute prediction error "
        "in dollars. Lower is better."
    )

    if model_mae < naive_mae:
        st.success(
            "ARIMA beat the previous-price baseline "
            "during this test period."
        )
    else:
        st.info(
            "ARIMA did not beat the previous-price baseline "
            "during this test period."
        )

    chart_data = test.set_index("date")[
        ["close", "ARIMA", "Previous close"]
    ].rename(columns={"close": "Actual"})

    st.line_chart(chart_data)

    st.write(
        "Both methods predict one trading day at a time. "
        "After each prediction, the actual price becomes "
        "available for the next forecast."
    )


with about_tab:
    st.write(
        "**Data source:** Yahoo Finance, downloaded using yfinance."
    )

    st.write(
        "**Model:** ARIMA(1,1,0), without seasonality or drift."
    )

    st.write(
        "**Baseline:** The next closing price equals "
        "the most recent closing price."
    )

    st.write(
        "**Testing:** The final 60 trading days are held "
        "back from initial model fitting."
    )

    st.write(
        "**Limitations:** Results cover one test period. "
        "A lower MAE does not guarantee profitable trades, "
        "and transaction costs are not included."
    )
