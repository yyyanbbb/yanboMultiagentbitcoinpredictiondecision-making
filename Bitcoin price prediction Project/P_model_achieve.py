import pandas as pd
import numpy as np
from models import ProphetModel
import matplotlib.pyplot as plt
import os

def load_data(train_path, test_path, val_path):
    """
    Load and preprocess data
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    val_df = pd.read_csv(val_path)
    
    # Prepare format needed for Prophet: ds (date) and y (target value)
    train_df = train_df.rename(columns={'Date': 'ds', 'Close': 'y'})
    test_df = test_df.rename(columns={'Date': 'ds', 'Close': 'y'})
    val_df = val_df.rename(columns={'Date': 'ds', 'Close': 'y'})
    
    return train_df, test_df, val_df

def evaluate_prophet_predictions(y_true, y_pred):
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
    
    # Calculate metrics
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Calculate direction accuracy
    direction_true = np.diff(y_true) > 0
    direction_pred = np.diff(y_pred) > 0
    direction_accuracy = np.mean(direction_true == direction_pred)
    
    # Calculate R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    print("\nProphet Model Evaluation Results:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    print(f"Direction Accuracy: {direction_accuracy:.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'direction_accuracy': direction_accuracy
    }

def save_prophet_results(y_true, y_pred, metrics):
    """
    Save Prophet model results to CSV files for comparison
    """
    # Create directory for results if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Save predictions
    pred_df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred
    })
    pred_df.to_csv('results/prophet_predictions.csv', index=False)
    
    # Save metrics for model comparison
    metrics_df = pd.DataFrame({
        'Model': ['Prophet'],
        'RMSE': [metrics['rmse']],
        'R²': [metrics['r2']],
        'MAE': [metrics['mae']],
        'Direction Accuracy': [metrics['direction_accuracy']]
    })
    metrics_df.to_csv('prophet_results.csv', index=False)
    
    print("Prophet model results saved to CSV files")

def prophet_modeling(train_df, test_df, holidays_df=None, period_days=7, fourier_order=10):
    """
    Train Prophet model and make predictions
    """
    model = ProphetModel(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        changepoint_range=1,
        holidays=holidays_df,
        seasonality_mode='multiplicative'
    )
    
    # Add custom seasonality
    model.add_seasonality(
        name='custom_seasonality',
        period=period_days,
        fourier_order=fourier_order,
        mode='multiplicative',
        prior_scale=0.5
    )
    
    # Train model
    model.fit(train_df)
    
    # Predict
    forecast = model.predict(periods=len(test_df), freq='D')
    
    # Plot forecast
    fig1 = model.plot_forecast(forecast, xlabel='Date', ylabel="BTC Price")
    plt.show()
    
    # Plot components
    fig2 = model.plot_components(forecast)
    plt.show()
    
    return forecast['yhat'][-len(test_df):]

def main():
    # Load data
    train_df, test_df, val_df = load_data(
        'data/processed_train_data.csv',
        'data/processed_test_data.csv',
        'data/processed_val_data.csv'
    )
    
    # Train model and make predictions
    print("Training Prophet model and generating predictions...")
    predictions = prophet_modeling(
        train_df=train_df,
        test_df=test_df,
        period_days=7,
        fourier_order=10
    )
    
    # Evaluate predictions
    metrics = evaluate_prophet_predictions(test_df['y'].values, predictions)
    
    # Save results for model comparison
    save_prophet_results(test_df['y'].values, predictions, metrics)

if __name__ == '__main__':
    main()