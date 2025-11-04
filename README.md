# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date:04/11/2025
### Name: Sarish Varshan V
### Reg No: 212223230196

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("/content/NFLX.csv")

data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

data.dropna(subset=['Date'], inplace=True)

data = data.sort_values(by='Date')

data.set_index('Date', inplace=True)

print(data.head())

target_col = 'Close'

plt.figure(figsize=(12, 6))
plt.plot(data.index, data[target_col], label=target_col, color='blue')
plt.xlabel('Date')
plt.ylabel(target_col)
plt.title(f'{target_col} Time Series')
plt.legend()
plt.grid()
plt.show()

def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

print("\n--- Stationarity Test for Value ---")
check_stationarity(data[target_col])

plt.figure(figsize=(10, 4))
plot_acf(data[target_col].dropna(), lags=30)
plt.title("Autocorrelation Function (ACF)")
plt.show()

plt.figure(figsize=(10, 4))
plot_pacf(data[target_col].dropna(), lags=30)
plt.title("Partial Autocorrelation Function (PACF)")
plt.show()

train_size = int(len(data) * 0.8)
train, test = data[target_col][:train_size], data[target_col][train_size:]

sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit(disp=False)

predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print(f'\nRoot Mean Squared Error (RMSE): {rmse:.4f}')

plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label='Actual', color='blue')
plt.plot(test.index, predictions, label='Predicted', color='red')
plt.xlabel('Date')
plt.ylabel(target_col)
plt.title(f'SARIMA Model Predictions for {target_col}')
plt.legend()
plt.grid()
plt.show()
```

### OUTPUT:

<img width="1002" height="547" alt="image" src="https://github.com/user-attachments/assets/42461792-676e-4316-bdcf-eb45d76283d6" />


<img width="290" height="149" alt="image" src="https://github.com/user-attachments/assets/13b6b639-4819-4477-8fe4-82b035893153" />

<img width="569" height="434" alt="image" src="https://github.com/user-attachments/assets/1f51858e-b5bc-42f9-994d-f3c409206a24" />

<img width="567" height="449" alt="image" src="https://github.com/user-attachments/assets/c9af8d92-6a47-4c42-8383-6c408fe68dd3" />
<img width="1005" height="570" alt="image" src="https://github.com/user-attachments/assets/575ae141-66ac-4246-8660-e9b5e131a07f" />


### RESULT:
Thus the program run successfully based on the SARIMA model.
