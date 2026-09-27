import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load the trained ARIMA model
model = joblib.load('arima_model.joblib')

st.title('Demand Forecasting App')
st.write('Forecast future demand using an ARIMA model.')

# User input for number of months to forecast
months_to_forecast = st.slider(
    'Select the number of months to forecast:',
    min_value=1,
    max_value=24,
    value=6
)

if st.button('Generate Forecast'):
    # Make future predictions
    future_forecast = model.predict(n_periods=months_to_forecast)

    # Create a DataFrame for the forecast with appropriate date index
    # Assuming the last date in your training data was the last date in the original df
    # You might need to adjust this if your training data ends earlier
    last_date_in_data = pd.to_datetime('2024-12-01') # Based on the last date in your dataframe df
    future_dates = pd.date_range(start=last_date_in_data + pd.DateOffset(months=1), periods=months_to_forecast, freq='MS')
    
    forecast_df = pd.DataFrame({'Forecasted Demand (000L)': future_forecast}, index=future_dates)

    st.subheader(f'Forecast for the next {months_to_forecast} months:')
    st.dataframe(forecast_df)

    # Plotting the forecast
