import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import matplotlib

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows common font
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue

# Load Bitcoin data
data = pd.read_csv('./data/BTC_USD Bitfinex Historical Data.csv')

# Preprocess closing price data (remove thousand separators and convert to float)
close_prices = data['Close'].str.replace(',', '').astype(float)

# Set time index and handle missing values
close_prices.index = pd.to_datetime(data['Date'])
close_prices = close_prices.asfreq('D').ffill()

# Seasonal decomposition
print("=== Seasonal Decomposition Analysis ===")
try:
    # Complete data seasonal decomposition
    decomp = seasonal_decompose(close_prices, model='additive', period=7)
    fig = decomp.plot()
    fig.suptitle('Bitcoin Closing Price Seasonal Decomposition', y=1.02)
    fig.set_size_inches((12, 10))
    fig.tight_layout()
    
    # Evaluate seasonality significance by observing seasonal component amplitude
    seasonal_component = decomp.seasonal
    print("\n=== Seasonality Significance Assessment ===")
    print(f"Seasonal component amplitude range: {seasonal_component.min():.2f} to {seasonal_component.max():.2f}")
    if (seasonal_component.max() - seasonal_component.min()) > 0.1 * close_prices.std():
        print("Conclusion: Large seasonal component variation, significant seasonality may exist")
    else:
        print("Conclusion: Small seasonal component variation, seasonality may not be significant")
    
    plt.show()

    # Last 30 days seasonal decomposition
    print("\n=== Last 30 Days Seasonal Decomposition ===")
    decomp_recent = seasonal_decompose(close_prices.tail(30), model='additive', period=7)
    fig_recent = decomp_recent.plot()
    fig_recent.suptitle('Bitcoin Closing Price Last 30 Days Seasonal Decomposition', y=1.02)
    fig_recent.set_size_inches((12, 10))
    fig_recent.tight_layout()
    plt.grid(True)
    plt.show()

except ValueError as e:
    print(f"Seasonal decomposition error: {e}")
    print("Please check if data contains missing values or if period setting is reasonable")
# #=== Seasonality Significance Assessment ===
# Seasonal component amplitude range: -26.27 to 51.53
# Conclusion: Small seasonal component variation, seasonality may not be significant
