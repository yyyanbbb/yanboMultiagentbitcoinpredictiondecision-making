import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        
    def load_data(self):
        """Load data and set date index"""
        self.data = pd.read_csv(self.data_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        # Process price columns to remove commas
        price_columns = ['Close', 'Open', 'High', 'Low']
        for col in price_columns:
            self.data[col] = self.data[col].str.replace(',', '').astype(float)
        
        # Process volume column (contains 'K' unit)
        self.data['Volume'] = self.data['Volume'].apply(self._convert_volume)
        
        # Process change percentage, remove '%' and convert to float
        self.data['Change'] = self.data['Change'].str.rstrip('%').astype(float) / 100
        
        self.data.set_index('Date', inplace=True)
        return self
    
    def _convert_volume(self, volume_str):
        """Convert volume string to numeric value"""
        try:
            # Remove commas
            volume_str = volume_str.replace(',', '')
            # Check if it contains 'K'
            if 'K' in volume_str:
                # Remove 'K' and multiply the value by 1000
                return float(volume_str.replace('K', '')) * 1000
            return float(volume_str)
        except (ValueError, AttributeError):
            return np.nan
        
    def handle_missing_values(self):
        """Handle missing values
        - For numeric data, use forward fill
        - If missing values exist at the start, use backward fill
        """
        self.data = self.data.ffill().bfill()
        return self
        
    def handle_outliers_iqr(self, threshold=1.5):
        """Handle outliers using the IQR method
        
        Args:
            threshold: IQR multiplier threshold (default 1.5)
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            q1 = self.data[col].quantile(0.25)
            q3 = self.data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            # Replace outliers with boundary values
            self.data[col] = np.where(self.data[col] < lower_bound, lower_bound,
                                    np.where(self.data[col] > upper_bound, upper_bound, self.data[col]))
        return self
    
    def create_date_features(self):
        """Create date features"""
        # Extract year, month, day, weekday, etc.
        self.data['year'] = self.data.index.year
        self.data['month'] = self.data.index.month
        self.data['day'] = self.data.index.day
        self.data['dayofweek'] = self.data.index.dayofweek
        self.data['quarter'] = self.data.index.quarter
        return self
    
    def calculate_technical_indicators(self):
        """Calculate technical indicators"""
        # Simple Moving Average (SMA)
        self.data['SMA_5'] = self._calculate_sma(self.data['Close'], 5)
        self.data['SMA_20'] = self._calculate_sma(self.data['Close'], 20)
        
        # Relative Strength Index (RSI)
        self.data['RSI_14'] = self._calculate_rsi(self.data['Close'], 14)
        
        # MACD
        self.data['MACD'], self.data['Signal_Line'] = self._calculate_macd(self.data['Close'])
        
        # Bollinger Bands
        self.data['BB_middle'] = self._calculate_sma(self.data['Close'], 20)
        rolling_std = self.data['Close'].rolling(window=20).std()
        self.data['BB_upper'] = self.data['BB_middle'] + (rolling_std * 2)
        self.data['BB_lower'] = self.data['BB_middle'] - (rolling_std * 2)
        
        # Volume Rate of Change
        self.data['Volume_ROC'] = self._calculate_roc(self.data['Volume'], 1)
        
        return self
    
    def _calculate_sma(self, series, window):
        """Calculate Simple Moving Average"""
        values = series.values  # Convert to numpy array for performance
        sma = np.zeros(len(values))
        for i in range(window-1, len(values)):
            sma[i] = np.mean(values[i-window+1:i+1])
        return sma
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI from scratch"""
        values = prices.values  # Convert to numpy array
        deltas = np.diff(values)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        rs = up/down
        rsi = np.zeros_like(values)
        rsi[:period] = 100. - 100./(1.+rs)

        for i in range(period, len(values)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up*(period-1) + upval)/period
            down = (down*(period-1) + downval)/period
            rs = up/down if down != 0 else np.inf
            rsi[i] = 100. - 100./(1.+rs)
            
        return rsi
    
    def _calculate_macd(self, prices, slow=26, fast=12, signal=9):
        """Calculate MACD from scratch"""
        values = prices.values  # Convert to numpy array
        
        # Calculate EMA
        def ema(data, span):
            alpha = 2.0 / (span + 1)
            ema_values = np.zeros_like(data)
            ema_values[0] = data[0]
            for i in range(1, len(data)):
                ema_values[i] = alpha * data[i] + (1 - alpha) * ema_values[i-1]
            return ema_values
        
        # Calculate MACD line
        fast_ema = ema(values, fast)
        slow_ema = ema(values, slow)
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = ema(macd_line, signal)
        
        return macd_line, signal_line
    
    def _calculate_roc(self, series, period=1):
        """Calculate Rate of Change (ROC)"""
        values = series.values  # Convert to numpy array
        roc = np.zeros_like(values)
        for i in range(period, len(values)):
            roc[i] = (values[i] - values[i-period]) / values[i-period]
        return roc
    
    def normalize_features(self):
        """Normalize/Standardize features"""
        # Use Min-Max normalization
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            min_val = self.data[column].min()
            max_val = self.data[column].max()
            if max_val - min_val != 0:  # Avoid division by zero
                self.data[column] = (self.data[column] - min_val) / (max_val - min_val)
        return self
    
    def split_data(self, train_ratio=0.7, val_ratio=0.15):
        """Split data into training, validation, and test sets
        Args:
            train_ratio: Training set ratio (default 0.7)
            val_ratio: Validation set ratio (default 0.15)
        Returns:
            Training set, validation set, test set
        """
        train_size = int(len(self.data) * train_ratio)
        val_size = int(len(self.data) * val_ratio)
        
        train_data = self.data.iloc[:train_size]
        val_data = self.data.iloc[train_size:train_size+val_size]
        test_data = self.data.iloc[train_size+val_size:]
        
        return train_data, val_data, test_data

def visualize_data(data):
    """Create candlestick chart and technical indicators visualization for Bitcoin prices"""
    # Create a copy of the data to avoid modifying the original data
    df = data.copy()
    
    # Preprocess data
    # Process price columns
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = df[col].str.replace(',', '').astype(float)
    
    # Process volume column
    df['Volume'] = df['Volume'].apply(lambda x: float(x.replace('K', '').replace(',', '')) * 1000 if isinstance(x, str) and 'K' in x else float(x.replace(',', '') if isinstance(x, str) else x))
    
    # Calculate technical indicators for display
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    
    # Create subplots with 4 chart areas
    fig = make_subplots(rows=4, cols=1, 
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.5, 0.2, 0.15, 0.15],
                        subplot_titles=('Bitcoin Price Trend (with 20-day and 50-day Moving Averages)',
                                      'Volume',
                                      'Price Rate of Change (%)',
                                      'Cumulative Return (%)'))

    # 1. Add candlestick chart
    fig.add_trace(go.Candlestick(x=df['Date'],
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                name='OHLC',
                                increasing_line_color='red',
                                decreasing_line_color='green'),
                  row=1, col=1)
    
    # Add moving averages
    fig.add_trace(go.Scatter(x=df['Date'], 
                            y=df['SMA20'], 
                            name='20-day MA',
                            line=dict(color='orange', width=1)),
                  row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df['Date'], 
                            y=df['SMA50'], 
                            name='50-day MA',
                            line=dict(color='blue', width=1)),
                  row=1, col=1)

    # 2. Add volume bar chart
    colors = ['red' if close >= open_ else 'green' 
              for close, open_ in zip(df['Close'], df['Open'])]
    
    fig.add_trace(go.Bar(x=df['Date'],
                        y=df['Volume'],
                        name='Volume',
                        marker_color=colors),
                  row=2, col=1)

    # 3. Add daily return rate
    df['Daily Return'] = df['Close'].pct_change() * 100
    fig.add_trace(go.Bar(x=df['Date'],
                        y=df['Daily Return'],
                        name='Daily Return',
                        marker_color=['red' if x >= 0 else 'green' for x in df['Daily Return']]),
                  row=3, col=1)

    # 4. Add cumulative return
    df['Cumulative Return'] = ((1 + df['Daily Return']/100).cumprod() - 1) * 100
    fig.add_trace(go.Scatter(x=df['Date'],
                            y=df['Cumulative Return'],
                            name='Cumulative Return',
                            line=dict(color='purple', width=1)),
                  row=4, col=1)

    # Update layout
    fig.update_layout(
        title='Bitcoin Price Analysis Charts',
        yaxis_title='Price (USD)',
        yaxis2_title='Volume',
        yaxis3_title='Daily Return (%)',
        yaxis4_title='Cumulative Return (%)',
        xaxis_rangeslider_visible=False,
        height=1200,  # Increase chart height
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Update Y-axis format
    fig.update_yaxes(tickformat=",", row=1, col=1)  # Add thousand separator to price axis
    fig.update_yaxes(tickformat=",", row=2, col=1)  # Add thousand separator to volume axis
    fig.update_yaxes(tickformat=".2f", row=3, col=1)  # Keep two decimal places for return rate
    fig.update_yaxes(tickformat=".2f", row=4, col=1)  # Keep two decimal places for cumulative return

    # Update styles for each subplot
    for i in range(1, 5):
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', row=i, col=1)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', row=i, col=1)
        
    # Show the chart
    fig.show()
    
    # Create feature correlation heatmap
    plt.figure(figsize=(12, 10))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Set Chinese font
    plt.rcParams['axes.unicode_minus'] = False  # Solve negative sign display issue
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.show()

def main():
    # Instantiate the preprocessor
    preprocessor = DataPreprocessor('data/BTC_USD Bitfinex Historical Data_processed.csv')
    
    try:
        # Load raw data
        raw_data = pd.read_csv('data/BTC_USD Bitfinex Historical Data_processed.csv')
        
        # Display the first few rows of raw data
        print("\n=== Raw Data Preview (First 5 Rows) ===")
        print(raw_data.head())
        
        # Display basic information about the data
        print("\n=== Basic Data Information ===")
        print(raw_data.info())
        
        # Display statistical description of the data
        print("\n=== Statistical Description of Data ===")
        print(raw_data.describe().to_string())
        
        # Display the number of missing values in each column
        print("\n=== Number of Missing Values in Each Column ===")
        print(raw_data.isnull().sum())
        
        # Create visualization charts
        print("\nGenerating price trend chart...")
        visualize_data(raw_data)
        
        input("\nPress Enter to continue processing data...")
        
        # Execute preprocessing steps
        preprocessor.load_data()
        print("\nData loading completed, raw data shape:", preprocessor.data.shape)
        
        preprocessor.handle_missing_values()
        print("Missing values handled")
        
        preprocessor.create_date_features()
        print("Date feature engineering completed")
        
        preprocessor.calculate_technical_indicators()
        print("Technical indicators calculated")
        
        preprocessor.normalize_features()
        print("Feature normalization completed")
        
        # Split into training, validation, and test sets
        train_data, val_data, test_data = preprocessor.split_data()
        print(f"Data split completed: Training set size {train_data.shape}, Validation set size {val_data.shape}, Test set size {test_data.shape}")
        
        # Save processed data
        train_data.to_csv('data/processed_train_data.csv')
        val_data.to_csv('data/processed_val_data.csv')
        test_data.to_csv('data/processed_test_data.csv')
        print("Processed data saved")
        
        # Display the list of processed features
        print("\n=== List of Processed Features ===")
        print(train_data.columns.tolist())
        
        # Display statistical description of processed data
        print("\n=== Statistical Description of Processed Data ===")
        print(train_data.describe())
        
    except Exception as e:
        print(f"Error occurred during processing: {str(e)}")

if __name__ == "__main__":
    main()