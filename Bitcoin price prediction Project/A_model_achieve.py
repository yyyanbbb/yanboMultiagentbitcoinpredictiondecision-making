import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import ARIMAModel, plot_differencing_analysis, grid_search_arima
import os

def load_data(train_path, test_path, val_path):
    """
    Load and preprocess data
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    val_df = pd.read_csv(val_path)
    
    # Extract closing price as time series data
    train_data = train_df['Close'].copy()
    test_data = test_df['Close'].copy()
    val_data = val_df['Close'].copy()
    
    return pd.Series(train_data), pd.Series(test_data), pd.Series(val_data)

def evaluate_predictions(y_true, y_pred):
    """
    Evaluate prediction results
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Ensure lengths match
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    # Calculate evaluation metrics
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Safe MAPE calculation with threshold
    epsilon = 1e-10  # Small threshold to prevent division by zero
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100
    
    # Direction accuracy calculation
    direction_true = np.diff(y_true) > 0
    direction_pred = np.diff(y_pred) > 0
    direction_accuracy = np.mean(direction_true == direction_pred)
    
    # R² calculation with safety check
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > epsilon else 0
    
    # Cap R² to reasonable range
    r2 = max(min(r2, 1.0), -1.0)  # Limit R² to [-1, 1] range
    
    print("\nPrediction Evaluation Results:")
    print(f"Evaluation Sample Count: {min_len}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Direction Accuracy: {direction_accuracy:.4f}")
    print(f"R²: {r2:.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'direction_accuracy': direction_accuracy,
        'r2': r2
    }

def plot_predictions(train_data, test_data, predictions, title="ARIMA Predictions"):
    """
    Plot prediction results
    """
    # Ensure prediction length matches test set
    min_len = min(len(test_data), len(predictions))
    test_data = test_data[:min_len]
    predictions = predictions[:min_len]
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(train_data)), train_data, label='Training Data')
    plt.plot(range(len(train_data), len(train_data) + len(test_data)), test_data, label='True Values')
    plt.plot(range(len(train_data), len(train_data) + len(predictions)), predictions, label='Predictions')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def save_arima_results(y_true, y_pred, metrics):
    """
    Save ARIMA model results to CSV files for comparison
    """
    # Create directory for results if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Save predictions
    pred_df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred
    })
    pred_df.to_csv('results/arima_predictions.csv', index=False)
    
    # Save metrics for model comparison
    metrics_df = pd.DataFrame({
        'Model': ['ARIMA'],
        'RMSE': [metrics['rmse']],
        'R²': [metrics['r2']],
        'MAE': [metrics['mae']],
        'Direction Accuracy': [metrics['direction_accuracy']]
    })
    metrics_df.to_csv('arima_results.csv', index=False)
    
    print("ARIMA model results saved to CSV files")

def enhanced_grid_search_arima(train_data, val_data, p_range=range(0, 5), d_range=range(0, 3), q_range=range(0, 5)):
    """
    Enhanced grid search using separate validation set
    """
    print("Performing enhanced grid search with separate validation set...")
    
    best_val_rmse = float('inf')
    best_model = None
    best_params = None
    
    # Progress tracking
    total_combinations = len(p_range) * len(d_range) * len(q_range)
    current_combination = 0
    
    for d in d_range:
        for p in p_range:
            for q in q_range:
                current_combination += 1
                if current_combination % 5 == 0:
                    print(f"Progress: {current_combination}/{total_combinations} combinations tested")
                
                try:
                    # Create and fit model on training data
                    model = ARIMAModel(p=p, d=d, q=q)
                    model.fit(train_data)
                    
                    # Predict on validation data
                    predictions = model.predict(steps=len(val_data))
                    
                    # Calculate validation RMSE
                    val_rmse = np.sqrt(np.mean((val_data - predictions) ** 2))
                    
                    # Track best model
                    if val_rmse < best_val_rmse:
                        best_val_rmse = val_rmse
                        best_model = model
                        best_params = {
                            'p': p, 
                            'd': d, 
                            'q': q, 
                            'val_rmse': val_rmse
                        }
                        print(f"New best model: ARIMA({p},{d},{q}) with validation RMSE: {val_rmse:.4f}")
                        
                except Exception as e:
                    print(f"Error with ARIMA({p},{d},{q}): {str(e)}")
    
    return best_model, best_params

class EnhancedARIMAModel(ARIMAModel):
    """Enhanced ARIMA model with robust fitting and prediction"""
    
    def __init__(self, p=1, d=1, q=1, stability_checks=True):
        super().__init__(p, d, q)
        self.stability_checks = stability_checks
        
    def fit(self, data):
        """Enhanced fitting with stability checks"""
        self.history = data.copy()
        
        # Check and handle stationarity
        if self.stability_checks:
            self.d = self._auto_determine_d(data)
            print(f"Auto-determined differencing order: d={self.d}")
        
        # Perform differencing
        diff_data = data.copy()
        for _ in range(self.d):
            diff_data = np.diff(diff_data)
        
        # Remove mean and save it
        self.mean = np.mean(diff_data)
        centered_data = diff_data - self.mean
        
        # Try-except blocks for robust parameter estimation
        try:
            self.ar_params = self._estimate_ar_params(centered_data)
        except Exception as e:
            print(f"Warning: Error estimating AR parameters: {e}")
            self.ar_params = np.zeros(self.p)
        
        # Calculate residuals for MA estimation
        residuals = centered_data.copy()
        if self.p > 0:
            for t in range(self.p, len(centered_data)):
                ar_pred = np.sum(self.ar_params * centered_data[t-self.p:t][::-1])
                residuals[t] = centered_data[t] - ar_pred
        
        # Estimate MA parameters with error handling
        try:
            self.ma_params = self._estimate_ma_params(residuals[self.p:])
        except Exception as e:
            print(f"Warning: Error estimating MA parameters: {e}")
            self.ma_params = np.zeros(self.q)
        
        # Save residuals and calculate variance
        self.resid = residuals[self.p:]
        self.sigma2 = np.var(self.resid) if len(self.resid) > 0 else 1.0
        
        # Check parameter stability
        if self.stability_checks:
            self._check_parameter_stability()
        
        return self
    
    def _auto_determine_d(self, data):
        """Automatically determine differencing order using ADF test"""
        from statsmodels.tsa.stattools import adfuller
        
        for d in range(4):  # Try up to 3rd order differencing
            # Apply d-order differencing
            diff_data = data.copy()
            for _ in range(d):
                diff_data = np.diff(diff_data)
            
            # Perform ADF test
            adf_result = adfuller(diff_data)
            p_value = adf_result[1]
            
            # If series is stationary, return this d value
            if p_value < 0.05:
                return d
                
        # If no differencing achieves stationarity, return 1 as default
        return 1
    
    def _check_parameter_stability(self):
        """Check and ensure parameter stability"""
        if self.p > 0:
            # Create AR polynomial
            ar_poly = np.r_[1, -self.ar_params]
            
            # Check if all roots are outside the unit circle
            roots = np.roots(ar_poly)
            if np.any(np.abs(roots) < 1.01):  # Allow small margin for numerical error
                print("Warning: AR parameters may lead to non-stationary process")
                # Shrink parameters toward zero for stability
                self.ar_params *= 0.95

def check_data_quality(data, title="Data Quality Check"):
    """Check time series data quality"""
    print(f"\n{title}")
    print("-" * 50)
    
    # Basic statistics
    print(f"Data length: {len(data)}")
    print(f"Min value: {np.min(data):.6f}")
    print(f"Max value: {np.max(data):.6f}")
    print(f"Mean: {np.mean(data):.6f}")
    print(f"Std dev: {np.std(data):.6f}")
    
    # Check for zeros or near-zeros
    near_zero_threshold = 1e-6
    zero_count = np.sum(np.abs(data) < near_zero_threshold)
    print(f"Values near zero: {zero_count} ({zero_count/len(data)*100:.4f}%)")
    
    # Check for missing or invalid values
    nan_count = np.sum(np.isnan(data))
    inf_count = np.sum(np.isinf(data))
    print(f"NaN values: {nan_count}")
    print(f"Inf values: {inf_count}")
    
    # Check for outliers using IQR method
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = np.sum((data < lower_bound) | (data > upper_bound))
    print(f"Potential outliers: {outliers} ({outliers/len(data)*100:.4f}%)")
    
    # Check stationarity
    from statsmodels.tsa.stattools import adfuller
    adf_result = adfuller(data)
    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"p-value: {adf_result[1]:.4f}")
    print(f"Series is {'stationary' if adf_result[1] < 0.05 else 'non-stationary'}")

def main():
    # 1. Load data
    print("Loading data...")
    train_data, test_data, val_data = load_data(
        'data/processed_train_data.csv',
        'data/processed_test_data.csv',
        'data/processed_val_data.csv'
    )
    
    # 2. Check data quality
    check_data_quality(train_data, "Training Data Quality")
    check_data_quality(test_data, "Test Data Quality")
    
    # 3. Time series analysis
    print("\nPerforming time series analysis...")
    plot_differencing_analysis(train_data, lag_num=40)
    
    # 4. Enhanced grid search for optimal parameters
    print("\nPerforming enhanced grid search for optimal ARIMA parameters...")
    best_model, best_params = enhanced_grid_search_arima(
        train_data,
        val_data,
        p_range=range(0, 5),
        d_range=range(0, 3), 
        q_range=range(0, 5)
    )
    
    print("\nOptimal ARIMA parameters:")
    print(f"p (AR order): {best_params['p']}")
    print(f"d (Differencing order): {best_params['d']}")
    print(f"q (MA order): {best_params['q']}")
    print(f"Validation RMSE: {best_params['val_rmse']:.4f}")
    
    # 5. Train enhanced model with optimal parameters
    print("\nTraining enhanced ARIMA model with optimal parameters...")
    model = EnhancedARIMAModel(
        p=best_params['p'], 
        d=best_params['d'], 
        q=best_params['q'],
        stability_checks=True
    )
    model.fit(train_data)
    
    # 6. Generate predictions
    print("\nGenerating predictions...")
    predictions = model.predict(steps=len(test_data))
    
    # 7. Evaluate predictions with robust metrics
    metrics = evaluate_predictions(test_data.values, predictions)
    
    # 8. Save results
    save_arima_results(test_data.values, predictions, metrics)
    
    # 9. Plot results
    plot_predictions(train_data, test_data, predictions)
    
    # 10. Model diagnostics
    print("\nPlotting model diagnostics...")
    model.plot_diagnostics()

if __name__ == "__main__":
    main()