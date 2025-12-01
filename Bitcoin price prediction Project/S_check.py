import pandas as pd
from stationarity_check import check_stationarity

# Load Bitcoin data
data = pd.read_csv('./data/BTC_USD Bitfinex Historical Data.csv')

# Preprocess closing price data (remove thousand separators and convert to float)
close_prices = data['Close'].str.replace(',', '').astype(float)

# 1. Check stationarity of original series
print("=== Stationarity Test for Original Data ===")
check_stationarity(close_prices, title='Original Closing Price')

# 2. Calculate first-order difference and check stationarity
print("\n=== Stationarity Test for First-Order Difference ===")
diff_prices = close_prices.diff().dropna()
check_stationarity(diff_prices, title='First-Order Difference of Closing Price')

# 3. Save processed data after confirming stationarity
if diff_prices.isna().sum() == 0:  # Ensure no NaN values
    data = data.iloc[1:]  # Remove first row of original data
    data['Closing - first-order difference'] = diff_prices.values  # Add difference column
    data.to_csv('./data/BTC_USD Bitfinex Historical Data_processed.csv', index=False)
    print("\nProcessed data file has been saved")
else:
    print("\nWarning: NaN values exist in the data, please check the processing workflow")
    