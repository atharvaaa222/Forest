import yfinance as yf
import pandas as pd
import numpy as np
import talib as ta
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

symbol = "^NSEBANK"
period = "max"
interval = "1d"
df = yf.download(tickers=symbol + '', period=period, interval=interval, ignore_tz=True,
                 group_by='column', auto_adjust=True, prepost=False, threads=True, proxy=None)
symbol1 = "RELIANCE"
period1 = "max"
interval1 = "1d"
df1 = yf.download(tickers=symbol1 + '.NS', period=period1, interval=interval1, ignore_tz=True,
                 group_by='column', auto_adjust=True, prepost=False, threads=True, proxy=None)

df.reset_index(inplace=True)
df1.reset_index(inplace=True)

df['TOM'] = df['Close'].shift(-1)
open = df["Open"].values
high = df["High"].values
low = df["Low"].values
close = df["Close"].values
volume = df["Volume"].values
df['TOM1'] = df1['Close']


typical_price = (high + low + close) / 3
cumulative_typical_price_volume = np.cumsum(typical_price * volume)
cumulative_volume = np.cumsum(volume)
df['vwap'] = cumulative_typical_price_volume / cumulative_volume
df['adx'] = ta.ADX(high, low, close, timeperiod=14)
df['atr'] = ta.ATR(high, low, close, timeperiod=14)
df['rsi'] = ta.RSI(close, timeperiod=14)
df['upper_band'], df['middle_band'], df['lower_band'] = ta.BBANDS(df['Close'].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
df['mom'] = ta.MOM(close, timeperiod=10)
df['tema'] = ta.TEMA(close, timeperiod=144)
################################################################################################################
df['m_pattern'] = ((df['Close'] < df['Open'].shift()) & (df['Close'] < df['Open'].shift(-1)) & (
        df['Open'] > df['Open'].shift()) & (df['Open'] > df['Open'].shift(-1)))
df['w_pattern'] = ((df['Close'] > df['Open'].shift()) & (df['Close'] > df['Open'].shift(-1)) & (
        df['Open'] < df['Open'].shift()) & (df['Open'] < df['Open'].shift(-1)))
df['bullish_engulfing'] = ta.CDLENGULFING(df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values)
df['bearish_engulfing'] = ta.CDLENGULFING(df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values)
df['hanging_man'] = ta.CDLHANGINGMAN(df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values)
df['hammer_pattern'] = ta.CDLHAMMER(df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values)
df['bullish_harami'] = ta.CDLHARAMI(df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values)
df['bearish_harami'] = ta.CDLHARAMICROSS(df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values)
df['morning_star'] = ta.CDLMORNINGSTAR(df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values)
df['evening_star'] = ta.CDLEVENINGSTAR(df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values)
df['doji'] = ta.CDLDOJI(df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values)
df['inside_bar'] = ta.CDL3INSIDE(df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values)
df['head_and_shoulders'] = ta.CDLINVERTEDHAMMER(df['Open'].values, df['High'].values, df['Low'].values,
                                                df['Close'].values)
df['symmetrical_triangle'] = ta.CDLXSIDEGAP3METHODS(df['Open'].values, df['High'].values, df['Low'].values,
                                                    df['Close'].values)

df.fillna(0, inplace=True)

predictors = ['Volume', 'Open', 'High', 'Low', 'Close', 'TOM1', 'adx', 'atr', 'rsi', 'vwap', 'm_pattern', 'w_pattern',
              'bullish_engulfing', 'bearish_engulfing', 'hammer_pattern', 'bullish_harami', 'bearish_harami',
              'morning_star', 'evening_star', 'doji', 'inside_bar', 'upper_band', 'middle_band', 'lower_band', 'mom', 'tema']
target = ['TOM']

df = df[:-1]

train_size = int(len(df) - 10)
test_size = len(df) - train_size

train = df.iloc[:train_size]
test = df.iloc[train_size:]

x_train = train[predictors]
y_train = train[target].values.ravel()
x_test = test[predictors]
y_test = test[target].values.ravel()

model = RandomForestRegressor(n_estimators=100, max_features=5, max_depth=25, random_state=0)
model.fit(x_train, y_train)

"""Apply Model"""

y_pred = model.predict(x_test)

y_pred_df = pd.DataFrame({'Predicted Value': y_pred})
# Combine predicted and actual values into a single DataFrame
results_df = pd.DataFrame({'TOM': y_test, 'Predicted Value': y_pred})
# Print the combined DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(results_df)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE:', rmse)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# mae
# r2
# plt.figure(figsize=(16, 8))
# plt.plot(df.index[train_size:], y_test, label="Actual")
# plt.plot(df.index[train_size:], y_pred, label="Predicted")
# plt.legend(loc="upper left")
# plt.xlabel("Date")
# plt.ylabel("Closing Price")
# plt.show()
