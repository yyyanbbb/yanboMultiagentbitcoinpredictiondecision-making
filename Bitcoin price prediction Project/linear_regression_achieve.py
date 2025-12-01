import pandas as pd
import numpy as np
from models import LinearRegression
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import seaborn as sns

# Configure matplotlib for Chinese characters (assuming 'SimHei' font is available)
# Keep these if the environment supports SimHei, otherwise remove or change font
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class MinMaxScaler:
    """
    Custom MinMaxScaler implementation.
    Scales features to a given range, typically [0, 1].
    """
    def __init__(self):
        self.min_ = None
        self.max_ = None
        self.scale_ = None

    def fit(self, X):
        """
        Compute the minimum, maximum, and scale to be used for scaling.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        self.scale_ = self.max_ - self.min_
        # Avoid division by zero for constant features
        self.scale_[self.scale_ == 0] = 1
        return self

    def transform(self, X):
        """
        Scale features using the computed minimum, maximum, and scale.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        X_scaled = (X - self.min_) / self.scale_
        return X_scaled

    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """
        Undo the scaling of X according to the computed minimum and scale.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        X_original = X * self.scale_ + self.min_
        return X_original

class KFold:
   
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X):
        """
        Generate indices to split data into training and test sets.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        current = 0

        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            yield train_indices, test_indices
            current = stop

def mean_squared_error(y_true, y_pred):
    """
    Calculates the Mean Squared Error (MSE).

    MSE = (1/n) * Σ(y_true - y_pred)²

    Arguments:
    - y_true: array of true values
    - y_pred: array of predicted values

    Returns:
    - MSE value, non-negative, smaller is better
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    """
    Calculates the Mean Absolute Error (MAE).

    MAE = (1/n) * Σ|y_true - y_pred|

    Arguments:
    - y_true: array of true values
    - y_pred: array of predicted values

    Return:
    - MAE value, non-negative, smaller is better
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    """
    Calculates the R² score (Coefficient of Determination).

    R² = 1 - SS_res / SS_tot
    Where:
    SS_res = Residual sum of squares = Σ(y_true - y_pred)²
    SS_tot = Total sum of squares = Σ(y_true - y_mean)²

    Arguments:
    - y_true: array of true values
    - y_pred: array of predicted values

    Returns:
    - R² score, best value is 1.0, can be negative
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate the residual sum of squares
    ss_res = np.sum((y_true - y_pred) ** 2)

    # Calculate the total sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    # Avoid division by zero
    if ss_tot == 0:
        return 0.0

    # Calculate the R² score
    r2 = 1 - (ss_res / ss_tot)

    return r2

def load_and_preprocess_data(train_path, val_path, test_path):
    """
    Loads and preprocesses data, lowering feature filtering standards to retain more features.
    """
    # Load data
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)

    # Select features and target variable
    feature_columns = [
        'Open', 'High', 'Low', 'Volume', 'Change',
        'SMA_5', 'SMA_20', 'RSI_14', 'MACD', 'Signal_Line',
        'BB_middle', 'BB_upper', 'BB_lower', 'Volume_ROC'
    ]
    target_column = 'Close'

    # Separate features and target
    X_train = train_data[feature_columns].astype(float)
    y_train = train_data[target_column].astype(float)

    X_val = val_data[feature_columns].astype(float)
    y_val = val_data[target_column].astype(float)

    X_test = test_data[feature_columns].astype(float)
    y_test = test_data[target_column].astype(float)

    # Handle infinite and NaN values
    X_train = pd.DataFrame(np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0), columns=feature_columns)
    X_val = pd.DataFrame(np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0), columns=feature_columns)
    X_test = pd.DataFrame(np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0), columns=feature_columns)

    # Calculate the correlation matrix between features
    correlation_matrix = X_train.corr().abs()

    # Print highly correlated feature pairs
    print("\nHighly Correlated Feature Pairs (correlation > 0.8):")
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if correlation_matrix.iloc[i, j] > 0.8:
                print(f"{correlation_matrix.columns[i]} - {correlation_matrix.columns[j]}: {correlation_matrix.iloc[i, j]:.4f}")

    # Select features (more relaxed collinearity criteria)
    # Method 1: Increase correlation threshold, keep more features
    correlation_threshold = 0.95  # Increased from 0.8 to 0.95

    # Calculate the correlation of each feature with the target
    target_correlations = abs(pd.concat([X_train, pd.Series(y_train, name=target_column)], axis=1).corr()[target_column])
    target_correlations = target_correlations.sort_values(ascending=False)

    # Select features one by one, using a more relaxed collinearity filter
    selected_features = []
    for feature in target_correlations.index:
        if feature == target_column:
            continue

        # Check the correlation of the current feature with already selected features
        if not selected_features:
            selected_features.append(feature)
        else:
            correlations = correlation_matrix.loc[feature, selected_features]
            if not any(correlations > correlation_threshold):
                selected_features.append(feature)

    print("\nSelected Features (correlation threshold=0.95):")
    print(selected_features)

    # Method 2: Force inclusion of certain basic features
    forced_features = ['Open', 'High', 'Low', 'SMA_5', 'RSI_14', 'MACD', 'BB_upper', 'Volume']
    for feature in forced_features:
        if feature not in selected_features and feature in feature_columns:
            selected_features.append(feature)

    print("\nFeatures After Forcing Inclusion:")
    print(selected_features)

    # Use the selected features
    X_train = X_train[selected_features]
    X_val = X_val[selected_features]
    X_test = X_test[selected_features]

    # Feature scaling
    feature_scaler = MinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_val_scaled = feature_scaler.transform(X_val)
    X_test_scaled = feature_scaler.transform(X_test)

    # Target variable scaling
    target_scaler = MinMaxScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_val_scaled = target_scaler.transform(y_val.values.reshape(-1, 1)).ravel()
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

    # Print data range information
    print("\nScaled Data Range:")
    print(f"X_train range: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
    print(f"y_train range: [{y_train_scaled.min():.2f}, {y_train_scaled.max():.2f}]")

    return (X_train_scaled, y_train_scaled), (X_val_scaled, y_val_scaled), (X_test_scaled, y_test_scaled), target_scaler, feature_scaler, selected_features

def train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, target_scaler, selected_features):
    """
    Trains and evaluates Linear Regression models, comparing the effects of no regularization,
    L1, L2, and Elastic Net regularization.
    Includes a hyperparameter tuning process using a wider range of regularization strengths.
    """
    # Define basic configurations for models to test
    model_configs = [
        {
            'name': 'No Regularization',
            'params': {'reg_type': None, 'reg_lambda': 0.0}
        },
        {
            'name': 'L1 (Lasso)',
            'base_params': {'reg_type': 'l1'},
            'tune_params': {'reg_lambda': [0.1, 1.0, 10.0]}  # Reduced search space
        },
        {
            'name': 'L2 (Ridge)',
            'base_params': {'reg_type': 'l2'},
            'tune_params': {'reg_lambda': [0.1, 1.0, 10.0]}  # Reduced search space
        },
        {
            'name': 'Elastic Net',
            'base_params': {'reg_type': 'elastic_net'},
            'tune_params': {
                'reg_lambda': [0.01, 0.05, 0.1],  # Lowered regularization strength range
                'l1_ratio': [0.1, 0.3]  # Lowered L1 ratio, increasing L2 influence
            }
        }
    ]

    # Dictionary to save results for each model
    results = {}

    # Train and evaluate each model
    for config in model_configs:
        print(f"\n{'-' * 20} {config['name']} {'-' * 20}")

        # If hyperparameter tuning is needed
        if 'tune_params' in config:
            print("Performing hyperparameter tuning...")
            best_params = grid_search_cv(
                X_train, y_train,
                base_params=config['base_params'],
                param_grid=config['tune_params'],
                cv=5,
                verbose=True  # Print more tuning information
            )
            print(f"Best parameters: {best_params}")
            model_params = {**config['base_params'], **best_params}
        else:
            model_params = config['params']

        # Set learning rate and max iterations based on model type
        learning_rate = 0.01  # Default learning rate
        max_iterations = 300  # Reduced default iterations
        learning_rate_decay = 0.01  # Default learning rate decay

        # Optimize parameters for different regularization types
        if config['name'] == 'L1 (Lasso)':
            learning_rate = 0.03  # Lasso might need a higher learning rate
            max_iterations = 500  # Reduced iterations
            learning_rate_decay = 0.015  # Slightly higher decay for Lasso
        elif config['name'] == 'L2 (Ridge)':
            learning_rate = 0.05  # Ridge can use a higher learning rate
            learning_rate_decay = 0.008  # Lower decay for Ridge
        elif config['name'] == 'Elastic Net':
            learning_rate = 0.04  # Increased learning rate for Elastic Net
            max_iterations = 800  # Increased iterations for Elastic Net
            learning_rate_decay = 0.012  # Medium decay for Elastic Net

        # Initialize model with best or preset parameters
        model = LinearRegression(
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            batch_size=32,
            learning_rate_decay=learning_rate_decay,  # Add learning rate decay
            **model_params
        )

        # Cross-validation
        print("Performing cross-validation...")
        cv_scores = cross_validate(model, X_train, y_train, k=5)
        print(f"Cross-validation R² scores: {cv_scores}")
        print(f"Average cross-validation R² score: {np.mean(cv_scores):.4f}")

        # Train on the full training set
        print("Training model on the full training set...")
        model.fit(X_train, y_train)

        # Get and print model coefficients
        coeffs = model.theta[1:]  # Skip intercept
        print("\nModel Coefficients (excluding intercept):")
        for feature, coef in zip(selected_features, coeffs):
            print(f"{feature}: {coef:.6f}")

        # Evaluate on the validation set
        val_predictions = model.predict(X_val)
        val_mse = mean_squared_error(y_val, val_predictions)
        val_r2 = r2_score(y_val, val_predictions)

        print("\nValidation Set Evaluation Metrics:")
        print(f"Validation Set MSE: {val_mse:.6f}")
        print(f"Validation Set R² Score: {val_r2:.6f}")

        # Evaluate on the test set
        test_predictions = model.predict(X_test)

        # Inverse scale predictions and true values to calculate actual metrics
        test_predictions_rescaled = target_scaler.inverse_transform(test_predictions.reshape(-1, 1)).flatten()
        y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        # Evaluation metrics (on scaled data)
        mse = mean_squared_error(y_test, test_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, test_predictions)
        r2 = r2_score(y_test, test_predictions) # This is the R2 on scaled data being plotted

        # Calculate rescaled RMSE (in actual price units)
        mse_rescaled = mean_squared_error(y_test_rescaled, test_predictions_rescaled)
        rmse_rescaled = np.sqrt(mse_rescaled)

        # Calculate directional prediction accuracy - only predicts direction (up/down)
        direction_correct = np.sum(np.sign(test_predictions[1:] - test_predictions[:-1]) ==
                                     np.sign(y_test[1:] - y_test[:-1]))
        direction_accuracy = direction_correct / (len(y_test) - 1)

        print("\nTest Set Evaluation Metrics:")
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R2 Score: {r2:.6f}")
        print(f"Actual Price RMSE: ${rmse_rescaled:.6f}")
        print(f"Direction Prediction Accuracy: {direction_accuracy:.6f}")

        # Calculate performance in different price ranges
        low_idx = y_test_rescaled < np.percentile(y_test_rescaled, 25)
        mid_idx = (y_test_rescaled >= np.percentile(y_test_rescaled, 25)) & (y_test_rescaled <= np.percentile(y_test_rescaled, 75))
        high_idx = y_test_rescaled > np.percentile(y_test_rescaled, 75)

        low_rmse = np.sqrt(mean_squared_error(y_test_rescaled[low_idx], test_predictions_rescaled[low_idx])) if np.any(low_idx) else 0
        mid_rmse = np.sqrt(mean_squared_error(y_test_rescaled[mid_idx], test_predictions_rescaled[mid_idx])) if np.any(mid_idx) else 0
        high_rmse = np.sqrt(mean_squared_error(y_test_rescaled[high_idx], test_predictions_rescaled[high_idx])) if np.any(high_idx) else 0

        print("\nRMSE in Different Price Ranges:")
        print(f"Low Price Range (0-25%): ${low_rmse:.6f}")
        print(f"Mid Price Range (25-75%): ${mid_rmse:.6f}")
        print(f"High Price Range (75-100%): ${high_rmse:.6f}")

        # Save results
        results[config['name']] = {
            'model': model,
            'predictions': test_predictions,
            'predictions_rescaled': test_predictions_rescaled,
            'coefficients': {feature: coef for feature, coef in zip(selected_features, coeffs)},
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'direction_accuracy': direction_accuracy
            },
            'actual_rmse': rmse_rescaled
        }

    # Find the model with the highest R2 score on the validation set
    best_model_name = max(results.keys(), key=lambda name: r2_score(y_val, results[name]['model'].predict(X_val)))
    print(f"\nBest Model based on Validation Set R² Score: {best_model_name}")

    # Plot prediction results comparison for different models
    plt.figure(figsize=(15, 6))
    plt.plot(y_test_rescaled, label='Actual Price', color='blue', alpha=0.5)

    colors = ['red', 'green', 'purple', 'orange']
    for (name, result), color in zip(results.items(), colors):
        plt.plot(result['predictions_rescaled'],
                 label=f'{name} (RMSE: ${result["actual_rmse"]:.6f}, Direction Acc: {result["metrics"]["direction_accuracy"]:.2f})',
                 color=color, alpha=0.7)

    plt.title('Bitcoin Price Prediction - Model Comparison')
    plt.xlabel('Time')
    plt.ylabel('Bitcoin Price (USD)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

    # Compare coefficients of each model
    plt.figure(figsize=(20, 15))  # Increase figure size for better spacing

    # Plot coefficients for each feature across different models
    for i, feature in enumerate(selected_features, 1):
        plt.subplot(int(np.ceil(len(selected_features) / 3)), 3, i)  # Use 3 columns for better spacing

        # Collect coefficient values for this feature from each model
        feature_coeffs = []
        model_names = []
        for name, result in results.items():
            feature_coeffs.append(result['coefficients'][feature])
            model_names.append(name)

        # Plot bar chart
        plt.bar(model_names, feature_coeffs)
        plt.title(f'Feature: {feature}')
        plt.ylabel('Coefficient Value')
        plt.xticks(rotation=30)  # Rotate x-axis labels for better readability
        plt.grid(True, alpha=0.3)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig('feature_coefficients_comparison_updated.png')
    plt.show()

    # Plot feature importance for each model
    plt.figure(figsize=(15, 10))

    # Divide plot into subplots equal to the number of models
    n_models = len(results)
    n_rows = (n_models + 1) // 2  # Ceiling division

    for i, (name, result) in enumerate(results.items(), 1):
        plt.subplot(n_rows, 2, i)

        # Create dataframe for feature importance plotting
        importance_df = pd.DataFrame({
            'feature': selected_features,
            'importance': np.abs(list(result['coefficients'].values()))
        }).sort_values('importance', ascending=False)

        # Plot feature importance
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.title(f'Feature Importance - {name}')
        plt.xlabel('|Coefficient Value|')
        plt.tight_layout()

    plt.savefig('feature_importance.png')
    plt.show()

    # Plot scatter plot of predicted vs actual values
    plt.figure(figsize=(15, 10))
    for i, (name, result) in enumerate(results.items(), 1):
        plt.subplot(2, 2, i)
        plt.scatter(y_test_rescaled, result['predictions_rescaled'], alpha=0.5)
        plt.plot([min(y_test_rescaled), max(y_test_rescaled)],
                 [min(y_test_rescaled), max(y_test_rescaled)],
                 'r--', label='Perfect Prediction')
        plt.title(f'{name} - Predicted vs Actual')
        plt.xlabel('Actual Price (USD)')
        plt.ylabel('Predicted Price (USD)')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('prediction_scatter.png')
    plt.show()

    return results

def grid_search_cv(X, y, base_params, param_grid, cv=5, verbose=False):
    """
    Simple grid search cross-validation implementation, with optional detailed tuning output.
    """
    if verbose:
        print(f"Starting grid search with {cv}-fold cross-validation")
        print(f"Base parameters: {base_params}")
        print(f"Grid search parameters: {param_grid}")

    # Generate all possible parameter combinations
    param_combinations = []

    # Case for single parameter
    if len(param_grid) == 1:
        param_name = list(param_grid.keys())[0]
        param_values = param_grid[param_name]
        for value in param_values:
            param_combinations.append({param_name: value})

    # Case for two parameters (for Elastic Net)
    elif len(param_grid) == 2:
        param_names = list(param_grid.keys())
        for value1 in param_grid[param_names[0]]:
            for value2 in param_grid[param_names[1]]:
                param_combinations.append({
                    param_names[0]: value1,
                    param_names[1]: value2
                })
    else:
         # Handle cases with more than 2 parameters if needed, or raise an error
         raise NotImplementedError("Grid search is currently implemented for 1 or 2 parameters only.")


    if verbose:
        print(f"Generated {len(param_combinations)} parameter combinations")

    # Store best parameters and score
    best_score = -float('inf')
    best_params = None
    all_params_scores = []  # Store scores for all parameter combinations

    # Evaluate each parameter combination using cross-validation
    for params in param_combinations:
        cv_scores = []
        fold_indices = split_k_fold(X, cv)

        for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Set model parameters
            model_params = {**base_params, **params}

            # Adjust learning rate based on regularization type
            lr = 0.01
            max_iter = 500
            lr_decay = 0.01  # Default learning rate decay
            
            if model_params.get('reg_type') == 'l1':
                lr = 0.03
                lr_decay = 0.015  # Slightly higher decay for Lasso
            elif model_params.get('reg_type') == 'l2':
                lr = 0.05
                lr_decay = 0.008  # Lower decay for Ridge
            elif model_params.get('reg_type') == 'elastic_net':
                lr = 0.04
                max_iter = 600  # Increased iterations for Elastic Net
                lr_decay = 0.012  # Medium decay for Elastic Net

            model = LinearRegression(
                learning_rate=lr,
                max_iterations=max_iter,  # Use iterations adjusted for model type
                batch_size=32,
                learning_rate_decay=lr_decay,  # Add learning rate decay
                **model_params
            )

            # Train and evaluate
            model.fit(X_train_fold, y_train_fold)
            predictions = model.predict(X_val_fold)
            score = r2_score(y_val_fold, predictions)
            cv_scores.append(score)

            if verbose:
                print(f"  Fold {fold_idx+1}/{cv}: Parameters {params}, R² Score: {score:.6f}")

        # Calculate average score
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        all_params_scores.append((params, mean_score, std_score))

        print(f"Parameters: {params}, Average R² Score: {mean_score:.6f} ± {std_score:.6f}")

        # Update best parameters
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            if verbose:
                print(f"  New best parameters: {best_params}, Score: {best_score:.6f}")

    # Print all parameter combinations results, sorted
    if verbose:
        print("\nPerformance of all parameter combinations (sorted descending):")
        sorted_results = sorted(all_params_scores, key=lambda x: x[1], reverse=True)
        for idx, (params, score, std) in enumerate(sorted_results):
            print(f"{idx+1}. Parameters: {params}, Score: {score:.6f} ± {std:.6f}")

    return best_params

def cross_validate(model, X, y, k=5):
    """Simple cross-validation implementation."""
    scores = []
    fold_indices = split_k_fold(X, k)

    for train_idx, val_idx in fold_indices:
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Create a new model instance to avoid influence from previous training
        model_copy = LinearRegression(
            learning_rate=model.learning_rate,
            max_iterations=model.max_iterations,
            batch_size=model.batch_size,
            reg_type=model.reg_type,
            reg_lambda=model.reg_lambda,
            l1_ratio=model.l1_ratio,
            learning_rate_decay=model.learning_rate_decay  # Add learning rate decay
        )

        model_copy.fit(X_train_fold, y_train_fold)
        predictions = model_copy.predict(X_val_fold)
        score = r2_score(y_val_fold, predictions)
        scores.append(score)

    return scores

def split_k_fold(X, k):
    """Splits data into k folds, returns train and validation indices."""
    m = X.shape[0]
    indices = np.random.permutation(m)
    fold_size = m // k

    folds = []
    for i in range(k):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < k - 1 else m

        val_indices = indices[val_start:val_end]
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])

        folds.append((train_indices, val_indices))

    return folds

def plot_results(y_true, results, title):
    """
    Plots the prediction results comparison for different models.
    """
    plt.figure(figsize=(15, 6))
    plt.plot(y_true, label='Actual', color='blue', alpha=0.5)

    colors = ['red', 'green', 'purple', 'orange']
    for (name, result), color in zip(results.items(), colors):
        plt.plot(result['predictions'], label=f'{name} Predictions',
                 color=color, alpha=0.5)

    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Bitcoin Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_feature_importance(importance_values, feature_names, model_name):
    """
    Plots a bar chart of feature importance.
    """
    plt.figure(figsize=(12, 6))

    # Create dataframe for feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values
    })

    # Sort by absolute importance
    importance_df['abs_importance'] = abs(importance_df['importance'])
    importance_df = importance_df.sort_values('abs_importance', ascending=True)

    # Plot bar chart
    plt.barh(range(len(importance_df)), importance_df['importance'])
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Feature Importance in Bitcoin Price Prediction ({model_name})')
    plt.tight_layout()
    plt.show()

def save_lr_results(y_true, y_pred, model_name, metrics, feature_importance=None):
    """Saves the prediction results and evaluation metrics for a Linear Regression model."""
    # Create directory for saving results if it doesn't exist
    if not os.path.exists('model_results'):
        os.makedirs('model_results')

    # Save prediction results
    results_df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred
    })
    results_df.to_csv(f'model_results/linear_regression_predictions_{model_name}.csv')

    # Save evaluation metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f'model_results/linear_regression_metrics_{model_name}.csv')

    # If feature importance exists, save it too
    if feature_importance is not None:
        importance_df = pd.DataFrame(feature_importance)
        importance_df.to_csv(f'model_results/linear_regression_feature_importance_{model_name}.csv')

def main():
    """
    Main function: Loads data, preprocesses it, trains models, and evaluates them.
    """
    print("=" * 80)
    print("Bitcoin Price Prediction - Linear Regression Model Comparison")
    print("=" * 80)

    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")

    train_path = 'data/processed_train_data.csv'
    val_path = 'data/processed_val_data.csv'
    test_path = 'data/processed_test_data.csv'

    (X_train, y_train), (X_val, y_val), (X_test, y_test), target_scaler, feature_scaler, selected_features = load_and_preprocess_data(
        train_path, val_path, test_path
    )

    # Print data shapes
    print(f"\nData Shapes:")
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    # Train and evaluate models
    print("\n2. Training and evaluating models...")
    results = train_and_evaluate_model(
        X_train, y_train, X_val, y_val, X_test, y_test, target_scaler, selected_features
    )

    # Display model comparison results table
    print("\n3. Model Comparison Results Summary:")
    comparison_data = []
    for name, result in results.items():
        metrics = result['metrics']
        comparison_data.append({
            'Model': name,
            'R²': metrics['r2'],
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae'],
            'Direction Accuracy': metrics['direction_accuracy'],
            'Actual RMSE': result['actual_rmse']
        })

    # Create comparison dataframe and print
    comparison_df = pd.DataFrame(comparison_data)
    print("\nModel Performance Comparison Table:")
    print(comparison_df.to_string(index=False, float_format="{:.6f}".format))

    # Save comparison results
    comparison_df.to_csv('model_comparison_results.csv', index=False, encoding='utf-8-sig')
    print("\nResults saved to 'model_comparison_results.csv'")
    
    # Find the best model (with highest R² or lowest RMSE)
    best_model_idx = comparison_df['R²'].idxmax()
    best_model_name = comparison_df.loc[best_model_idx, 'Model']
    best_model_results = comparison_df.loc[best_model_idx:best_model_idx]
    
    # Save best model results in the format expected by the comparison script
    best_model_results['Model'] = 'Linear Regression'  # Simplify model name for comparison
    best_model_results.to_csv('linear_regression_results.csv', index=False)
    print(f"\nBest model ({best_model_name}) results saved as 'linear_regression_results.csv'")

    # Summarize the characteristics of the four models
    print("\n4. Model Characteristics Summary:")
    print("No Regularization: Relies entirely on data fitting, potentially prone to overfitting.")
    print("L1 (Lasso): Tends to produce sparse solutions, shrinking coefficients of unimportant features to zero.")
    print("L2 (Ridge): Shrinks all feature coefficients proportionally, providing stable solutions.")
    print("Elastic Net: Combines the advantages of L1 and L2, balancing sparsity and stability.")

    # Show the number of non-zero coefficients for each model
    print("\nNumber of Non-Zero Coefficients per Model:")
    for name, result in results.items():
        non_zero_count = sum(1 for coef in result['coefficients'].values() if abs(coef) > 1e-6)
        total_count = len(result['coefficients'])
        print(f"{name}: {non_zero_count}/{total_count} ({non_zero_count/total_count:.2%})")

    print("\nAnalysis complete.")

    # Visualize model comparison results
    print("\nPlotting model performance comparison charts...")
    visualize_model_comparison(results)

def visualize_model_comparison(results):
    """
    Generates model comparison visualization charts.
    """
    # Extract required data
    model_names = list(results.keys())
    r2_scores = [results[name]['metrics']['r2'] for name in model_names]
    direction_accuracy = [results[name]['metrics']['direction_accuracy'] for name in model_names]
    rmse_values = [results[name]['actual_rmse'] for name in model_names]

    # Create performance comparison charts
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # R2 score comparison
    ax[0].bar(model_names, r2_scores, color=['red', 'green', 'blue', 'purple'])
    ax[0].set_title(r'$R^2$ Score Comparison')  # Use LaTeX-style math rendering
    ax[0].set_ylabel(r'$R^2$ Score')
    # Adjust y-axis limits to better visualize large differences
    min_r2 = min(r2_scores) if r2_scores else 0
    max_r2 = max(r2_scores) if r2_scores else 1
    ax[0].set_ylim(min_r2 - 10, max_r2 + 10)  # Extend limits to include outliers
    ax[0].grid(True, alpha=0.3)

    # Direction accuracy comparison
    ax[1].bar(model_names, direction_accuracy, color=['red', 'green', 'blue', 'purple'])
    ax[1].set_title('Direction Prediction Accuracy Comparison')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_ylim(0.4, 1.0)  # Direction accuracy is typically between 0.5-1.0
    ax[1].grid(True, alpha=0.3)

    # RMSE comparison
    ax[2].bar(model_names, rmse_values, color=['red', 'green', 'blue', 'purple'])
    ax[2].set_title('RMSE Comparison (Lower is better)')
    ax[2].set_ylabel('RMSE (USD)')
    ax[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_performance_comparison.png')
    plt.show()

    # Feature coefficient heatmap
    plt.figure(figsize=(12, 8))

    # Prepare heatmap data
    # Get features from the first model's coefficients
    features = list(next(iter(results.values()))['coefficients'].keys())
    coef_data = {}

    for name in model_names:
        # Ensure the feature order is consistent
        coef_data[name] = [results[name]['coefficients'].get(feature, 0) for feature in features] # Use .get with default 0 for safety

    # Convert to DataFrame
    coef_df = pd.DataFrame(coef_data, index=features)

    # Plot heatmap
    sns.heatmap(coef_df, annot=True, fmt='.3f', cmap='coolwarm', center=0)
    plt.title('Feature Coefficient Comparison Across Models')
    plt.tight_layout()
    plt.savefig('coefficient_heatmap.png')
    plt.show()

if __name__ == "__main__":
    main()