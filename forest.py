import yfinance as yf
import pandas as pd
import numpy as np
import talib as ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


symbol = "TCS.NS"
period = "max"
interval = "1d"

df = yf.download(tickers=symbol, period=period, interval=interval, auto_adjust=True)

# Calculating technical indicators
high = df["High"].values
low = df["Low"].values
close = df["Close"].values
volume = df["Volume"].values

df['adx'] = ta.ADX(high, low, close, timeperiod=14)
df['atr'] = ta.ATR(high, low, close, timeperiod=14)
df['rsi'] = ta.RSI(close, timeperiod=14)
df['upper_band'], df['middle_band'], df['lower_band'] = ta.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
df['mom'] = ta.MOM(close, timeperiod=10)

# Shift Close by 1 to get tomorrow's close as target variable
df['TOM'] = df['Close'].shift(-1)

# Drop rows with NaN values
df.dropna(inplace=True)
# Remove last row which contains NaN in TOM
df = df.iloc[:-1]

# Features
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'adx', 'atr', 'rsi', 'upper_band', 'middle_band', 'lower_band', 'mom']
target = ['TOM']

X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)


# Train the model
model = RandomForestRegressor(n_estimators=100, max_features=20, max_depth=50, random_state=0)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
print("mse:", mse)