import yfinance as yf
import pandas as pd
import statsmodels.api as sm

# Step 1: Fetch stock data
stock_symbol = "AAPL"  # Change to your stock
df = yf.download(stock_symbol, period="1y", interval="1d")

# Step 2: Reset index and rename columns
df = df.reset_index()
df.rename(columns={'Date': 'Stock Date', 'Close': 'Stock Price'}, inplace=True)

# Step 3: Handle missing values properly
df['Stock Price'] = df['Stock Price'].ffill()

# Step 4: Ensure correct data types
df['Stock Date'] = pd.to_datetime(df['Stock Date'])

# Step 5: Set date index with frequency
df.set_index('Stock Date', inplace=True)
df = df.asfreq('D')

# Step 6: Train ARIMA model
df['Stock Price'] = df['Stock Price'].astype(float)
model = sm.tsa.ARIMA(df['Stock Price'].dropna(), order=(5,1,0))  
model_fit = model.fit()

# Step 7: Forecast future prices
forecast_steps = 365
forecast = model_fit.forecast(steps=forecast_steps) 

# Step 8: Create future dates
forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), 
                               periods=forecast_steps, freq='D')

# Step 9: Create forecast DataFrame
forecast_df = pd.DataFrame({'Stock Date': forecast_dates, 'Predicted Price': forecast.tolist()})

# Step 10: Ensure numeric conversion
df = df.reset_index()
forecast_df['Predicted Price'] = pd.to_numeric(forecast_df['Predicted Price'], errors='coerce')

# Step 11: Merge actual & predicted data
final_df = pd.concat([df, forecast_df], ignore_index=True, sort=False)

# Output for Power BI
print(final_df)
