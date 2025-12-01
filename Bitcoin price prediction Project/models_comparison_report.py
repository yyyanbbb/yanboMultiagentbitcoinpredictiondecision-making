import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_model_results():
    """
    Load results data from the three models
    """
    print("Loading model result data...")
    
    # Initialize results list
    model_results = []
    
    # Try to load from existing combined results file
    try:
        print("Attempting to load combined results file...")
        combined_results = pd.read_csv('model_comparison_results.csv')
        return combined_results
    except FileNotFoundError:
        print("Combined results file not found, attempting to load individual model results...")
    
    # Try to load ARIMA model results
    try:
        arima_results = pd.read_csv('arima_results.csv')
        arima_results['Model'] = 'ARIMA'
        model_results.append(arima_results)
        print("ARIMA model results loaded")
    except FileNotFoundError:
        print("ARIMA model results file not found")
    
    # Try to load Prophet model results
    try:
        prophet_results = pd.read_csv('prophet_results.csv')
        prophet_results['Model'] = 'Prophet'
        model_results.append(prophet_results)
        print("Prophet model results loaded")
    except FileNotFoundError:
        print("Prophet model results file not found")
    
    # Try to load Linear Regression model results
    try:
        lr_results = pd.read_csv('linear_regression_results.csv')
        lr_results['Model'] = 'Linear Regression'
        model_results.append(lr_results)
        print("Linear Regression model results loaded")
    except FileNotFoundError:
        # Try alternative locations
        try:
            lr_results = pd.read_csv('model_results/linear_regression_best_model.csv')
            lr_results['Model'] = 'Linear Regression'
            model_results.append(lr_results)
            print("Linear Regression model results loaded from alternative location")
        except FileNotFoundError:
            print("Linear Regression model results file not found")
    
    # If no results found, raise error
    if not model_results:
        raise FileNotFoundError("No model result files found. Please run the models first to generate result files.")
    
    # Combine all results
    combined_results = pd.concat(model_results, ignore_index=True)
    return combined_results

def plot_performance_comparison(results):
    """
    Plot model performance comparison charts
    """
    plt.figure(figsize=(15, 10))
    
    # Create subplot layout
    plt.subplot(2, 2, 1)
    plt.bar(results['Model'], results['RMSE'])
    plt.title('RMSE Comparison (lower is better)')
    plt.ylabel('RMSE value')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 2)
    if 'R²' in results.columns:
        plt.bar(results['Model'], results['R²'])
        plt.title('R² Comparison (higher is better)')
        plt.ylabel('R² value')
        plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 3)
    if 'Direction Accuracy' in results.columns:
        plt.bar(results['Model'], results['Direction Accuracy'])
        plt.title('Direction Accuracy Comparison (higher is better)')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 4)
    if 'MAE' in results.columns:
        plt.bar(results['Model'], results['MAE'])
        plt.title('MAE Comparison (lower is better)')
        plt.ylabel('MAE value')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_radar_chart(results):
    """
    Plot radar chart comparing model performance
    """
    # Define the metrics to compare
    metrics = ['Prediction Accuracy', 'Computational Efficiency', 'Interpretability', 
              'Non-linearity Handling', 'Outlier Handling', 'Seasonality Capture']
    
    # Scores for each model on the metrics (1-5 scale)
    scores = {
        'ARIMA': [4, 3, 4, 2, 2, 3],
        'Prophet': [3, 2, 5, 3, 4, 5],
        'Linear Regression': [2, 5, 5, 1, 1, 1]
    }
    
    # Always include all three models for comparison
    available_models = ['ARIMA', 'Prophet', 'Linear Regression']
    
    # Create radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Close the radar chart
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Add metrics labels
    ax.set_thetagrids(angles[:-1] * 180/np.pi, metrics)
    
    # Add radial axis labels (1-5)
    ax.set_rlabel_position(0)
    ax.set_rticks([1, 2, 3, 4, 5])
    ax.set_rlim(0, 5)
    
    # Define colors for each model
    colors = ['red', 'blue', 'green']
    
    # Plot each model
    for i, model in enumerate(available_models):
        values = scores[model]
        values = np.concatenate((values, [values[0]]))  # Close the radar chart
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
    
    plt.legend(loc='upper right')
    plt.title('Model Characteristics Comparison')
    plt.tight_layout()
    plt.show()

def analyze_computational_complexity():
    """
    Analyze the computational complexity of the three models
    """
    complexity_data = {
        'Model': ['ARIMA', 'Prophet', 'Linear Regression'],
        'Training Time Complexity': ['O(n²) ~ O(n³)', 'O(n log n)', 'O(n²) or O(n)'],
        'Prediction Time Complexity': ['O(p+q)', 'O(log n)', 'O(n)'],
        'Memory Usage': ['Medium', 'High', 'Low'],
        'Parallel Computing Support': ['Poor', 'Good', 'Fair'],
        'Big Data Suitability': ['Poor', 'Medium', 'Good']
    }
    
    complexity_df = pd.DataFrame(complexity_data)
    
    # Print table
    print("\nComputational Complexity Comparison:")
    print(complexity_df.to_string(index=False))
    
    # Draw training time comparison chart (for illustration only)
    training_times = {
        'ARIMA': 0.5,
        'Prophet': 1.2,
        'Linear Regression': 0.2
    }
    
    plt.figure(figsize=(10, 6))
    plt.bar(training_times.keys(), training_times.values())
    plt.title('Model Training Time Comparison (seconds)')
    plt.ylabel('Training Time (seconds)')
    plt.show()
    
    return complexity_df

def analyze_model_suitability():
    """
    Analyze the suitability scenarios for the three models
    """
    suitability_data = {
        'Model': ['ARIMA', 'Prophet', 'Linear Regression'],
        'Time Series Type': ['Short-term, stationary series', 'Long-term series with strong seasonality', 'Data with obvious linear relationships'],
        'Data Scale': ['Small (thousands of records)', 'Medium to large (tens of thousands)', 'Unlimited'],
        'Missing Value Sensitivity': ['High', 'Low', 'Medium'],
        'Outlier Sensitivity': ['High', 'Low', 'High'],
        'Parameter Tuning Difficulty': ['Complex (requires time series knowledge)', 'Simple (reasonable defaults for most parameters)', 'Simple'],
        'Prediction Confidence Interval': ['Supported (based on model assumptions)', 'Supported (based on Bayesian inference)', 'Not directly supported'],
        'Best Prediction Period': ['Short-term (days, weeks)', 'Medium to long-term (weeks, months, quarters)', 'Short-term']
    }
    
    suitability_df = pd.DataFrame(suitability_data)
    
    # Print table
    print("\nModel Suitability Comparison:")
    print(suitability_df.to_string(index=False))
    
    return suitability_df

def generate_model_comparison_report(results):
    """
    Generate complete model comparison report
    """
    print("\n" + "="*80)
    print("Bitcoin Price Prediction Model Comparison Analysis Report")
    print("="*80)
    
    # Performance metrics comparison
    print("\n1. Performance Metrics Comparison")
    print("-"*50)
    performance_metrics = results[['Model', 'RMSE', 'R²', 'MAE', 'Direction Accuracy']].sort_values('RMSE')
    print(performance_metrics.to_string(index=False, float_format="{:.4f}".format))
    
    # Computational complexity analysis
    print("\n2. Computational Complexity Comparison")
    print("-"*50)
    complexity_df = analyze_computational_complexity()
    
    # Model suitability analysis
    print("\n3. Model Suitability Comparison")
    print("-"*50)
    suitability_df = analyze_model_suitability()
    
    # Comprehensive evaluation and recommendations
    print("\n4. Comprehensive Evaluation and Recommendations")
    print("-"*50)
    
    # Find the model with the lowest RMSE
    best_model = results.loc[results['RMSE'].idxmin(), 'Model']
    
    print(f"● Best Prediction Accuracy Model: {best_model} (lowest RMSE)")
    
    # Provide comprehensive analysis based on model characteristics
    print("\n● Comprehensive Analysis:")
    print("  - ARIMA: Good short-term prediction effect for stationary time series, but sensitive to non-linear relationships and outliers")
    print("  - Prophet: Excellent performance on data with strong seasonality, built-in holiday effect handling, but higher computational resource consumption")
    print("  - Linear Regression: High computational efficiency, easy to understand, but weak in handling non-linear relationships and outliers")
    
    print("\n● Best Choice Recommendation for Bitcoin Price Prediction:")
    if 'ARIMA' in best_model:
        print("  Based on performance metrics, the ARIMA model performs best in prediction accuracy.")
        print("  - Advantages: High prediction accuracy for stationary series, good model interpretability")
        print("  - Limitations: Limited ability to handle sudden changes and abnormal fluctuations in the Bitcoin market")
        print("  Recommended for: Short-term (1-7 days) Bitcoin price trend prediction")
    elif 'Prophet' in best_model:
        print("  Based on performance metrics, the Prophet model performs best in prediction accuracy.")
        print("  - Advantages: Strong ability to capture long-term trends and seasonal fluctuations, good adaptability to outliers")
        print("  - Limitations: Higher computational resource consumption, longer training time")
        print("  Recommended for: Medium to long-term (1-3 months) Bitcoin price trend and volatility prediction")
    else:
        print("  Based on performance metrics, the Linear Regression model performs best in prediction accuracy.")
        print("  - Advantages: Fast computation, easy to understand and implement")
        print("  - Limitations: Unable to capture non-linear features and complex fluctuations in the market")
        print("  Recommended for: Short-term price trend direction prediction based on multiple features")
    
    # Recommended combination strategy
    print("\n● Recommended Combination Strategy:")
    print("  Leveraging the advantages of multiple models, the following combination strategy can be adopted:")
    print("  1. Short-term prediction (1-7 days): Use ARIMA model")
    print("  2. Medium-term prediction (1-4 weeks): Use Prophet model to capture seasonality and trends")
    print("  3. Feature importance analysis: Use Linear Regression model to understand the impact of various features on price")
    print("  4. Model ensemble: Consider using multi-model weighted fusion methods to combine predictions from all three models")
    
    print("\n" + "="*80)
    print("Report generation complete")
    print("="*80)

def main():
    try:
        # Load model results
        results = load_model_results()
        
        # Draw performance comparison charts
        plot_performance_comparison(results)
        
        # Draw radar chart
        plot_radar_chart(results)
        
        # Generate complete report
        generate_model_comparison_report(results)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the models first to generate result files before comparing them.")

if __name__ == "__main__":
    main() 