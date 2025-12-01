from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from typing import Tuple, List, Optional

def compute_acf(data, nlags):
    """
    Calculate Autocorrelation Function (ACF)
    """
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    data = data - mean
    acf = np.zeros(nlags + 1)
    variance = np.sum(data ** 2) / n
    
    for lag in range(nlags + 1):
        c = np.sum(data[lag:] * data[:(n-lag)]) / n
        acf[lag] = c / variance if variance != 0 else 0
    return acf

def compute_pacf(data, nlags):
    """
    Calculate Partial Autocorrelation Function (PACF) using the Durbin-Levinson algorithm
    """
    pacf = np.zeros(nlags + 1)
    pacf[0] = 1.0
    
    # Calculate autocorrelation function
    acf = compute_acf(data, nlags)
    
    # Durbin-Levinson algorithm
    for k in range(1, nlags + 1):
        phi = np.zeros(k)
        phi[-1] = acf[k]
        
        for j in range(k-1):
            phi[j] = pacf[j+1]
        
        if k > 1:
            for j in range(k-1):
                phi[j] = pacf[j+1] - phi[-1] * pacf[k-j-1]
        
        pacf[k] = phi[-1]
        
    return pacf

def acf_pacf_draw(df, lag_num=40, acf_plot=True, pacf_plot=True, title="", ylim=1):
    """
    Plot ACF and PACF graphs for time series
    
    Parameters:
        df: pandas DataFrame or Series containing time series data
        lag_num: int, maximum lag order
        acf_plot: bool, whether to plot ACF
        pacf_plot: bool, whether to plot PACF
        title: str, chart title
        ylim: float, y-axis limit range
    """
    # Determine number of subplots
    num_plots = 1 + int(acf_plot) + int(pacf_plot)
    fig, axes = plt.subplots(1, num_plots, figsize=(15, 4))
    
    # Convert data to one-dimensional array
    data = np.array(df).squeeze()
    
    # Original series plot
    axes[0].plot(data)
    axes[0].set_title('Original Series')
    
    plot_idx = 1
    if acf_plot:
        # Calculate ACF
        acf_values = compute_acf(data, lag_num)
        lags = np.arange(len(acf_values))
        
        # Plot ACF
        axes[plot_idx].stem(lags, acf_values)
        axes[plot_idx].axhline(y=0, linestyle='-', color='black')
        axes[plot_idx].axhline(y=1.96/np.sqrt(len(data)), linestyle='--', color='gray')
        axes[plot_idx].axhline(y=-1.96/np.sqrt(len(data)), linestyle='--', color='gray')
        axes[plot_idx].set_title('ACF')
        axes[plot_idx].set_ylim(-ylim, ylim)
        plot_idx += 1
        
    if pacf_plot:
        # Calculate PACF
        pacf_values = compute_pacf(data, lag_num)
        lags = np.arange(len(pacf_values))
        
        # Plot PACF
        axes[plot_idx].stem(lags, pacf_values)
        axes[plot_idx].axhline(y=0, linestyle='-', color='black')
        axes[plot_idx].axhline(y=1.96/np.sqrt(len(data)), linestyle='--', color='gray')
        axes[plot_idx].axhline(y=-1.96/np.sqrt(len(data)), linestyle='--', color='gray')
        axes[plot_idx].set_title('PACF')
        axes[plot_idx].set_ylim(-ylim, ylim)
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_differencing_analysis(data, lag_num=100):
    """
    Analyze and plot differencing results for time series
    
    Parameters:
        data: pandas DataFrame or Series, original time series data
        lag_num: int, maximum lag order
    """
    # Original series
    acf_pacf_draw(data, lag_num, True, True, 'Original Series')
    
    # First-order differencing
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        diff1 = data.diff().dropna()
    else:
        diff1 = np.diff(data)
    acf_pacf_draw(diff1, lag_num, True, True, '1st Order Differencing')
    
    # Second-order differencing
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        diff2 = diff1.diff().dropna()
    else:
        diff2 = np.diff(diff1)
    acf_pacf_draw(diff2, lag_num, True, True, '2nd Order Differencing')

class LinearRegression:
    """
    Linear regression model implementation with regularization support
    """
    def __init__(self, learning_rate=0.01, max_iterations=1000, batch_size=32, 
                 reg_type=None, reg_lambda=0.0, l1_ratio=0.5, learning_rate_decay=0.0):
        """
        Initialize linear regression model
        
        Parameters:
        - learning_rate: Learning rate for gradient descent
        - max_iterations: Maximum number of iterations
        - batch_size: Batch size for mini-batch gradient descent
        - reg_type: Regularization type, can be None, 'l1', 'l2', or 'elastic_net'
        - reg_lambda: Regularization strength
        - l1_ratio: L1 regularization ratio in elastic net (only used for elastic_net)
        - learning_rate_decay: Decay rate for learning rate (0.0 means no decay)
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.reg_type = reg_type
        self.reg_lambda = reg_lambda
        self.l1_ratio = l1_ratio
        self.learning_rate_decay = learning_rate_decay
        self.theta = None
        
        # Model parameters
        self.feature_means = None
        self.feature_stds = None
        self.loss_history = []
        self.val_loss_history = []
        
    def _standardize(self, X, fit=False):
        """Feature standardization"""
        if not self.standardize:
            return X
            
        if fit:
            self.feature_means = np.mean(X, axis=0)
            self.feature_stds = np.std(X, axis=0)
            self.feature_stds[self.feature_stds < 1e-8] = 1.0
            
        return (X - self.feature_means) / self.feature_stds
        
    def _add_bias(self, X):
        """
        Add bias term (intercept) to feature matrix
        """
        return np.c_[np.ones(X.shape[0]), X]
        
    def _compute_loss(self, X, y, theta):
        """
        Calculate loss function value (including regularization term)
        """
        m = X.shape[0]
        predictions = X.dot(theta)
        mse = np.sum((predictions - y) ** 2) / (2 * m)
        
        # Calculate regularization term
        if self.reg_type is None:
            return mse
            
        if self.reg_type == 'l1':
            # Lasso regularization
            reg_term = self.reg_lambda * np.sum(np.abs(theta[1:])) / m
        elif self.reg_type == 'l2':
            # Ridge regularization
            reg_term = (self.reg_lambda * np.sum(theta[1:] ** 2)) / (2 * m)
        elif self.reg_type == 'elastic_net':
            # Elastic net regularization
            l1_term = self.reg_lambda * self.l1_ratio * np.sum(np.abs(theta[1:])) / m
            l2_term = self.reg_lambda * (1 - self.l1_ratio) * np.sum(theta[1:] ** 2) / (2 * m)
            reg_term = l1_term + l2_term
        else:
            reg_term = 0
            
        return mse + reg_term
        
    def _compute_gradient(self, X, y, theta):
        """
        Calculate gradient (including gradient of regularization term)
        """
        m = X.shape[0]
        error = X.dot(theta) - y
        gradient = X.T.dot(error) / m
        
        # Calculate gradient of regularization term
        if self.reg_type is None:
            return gradient
            
        elif self.reg_type == 'l1':
            # Lasso regularization gradient (subgradient approximation)
            # Don't regularize bias term theta[0]
            reg_grad = self.reg_lambda * np.sign(theta[1:]) / m
            gradient[1:] += reg_grad
        elif self.reg_type == 'l2':
            # Ridge regularization gradient
            # Don't regularize bias term theta[0]
            reg_grad = self.reg_lambda * theta[1:] / m
            gradient[1:] += reg_grad
        elif self.reg_type == 'elastic_net':
            # Elastic net regularization gradient
            # Don't regularize bias term theta[0]
            l1_grad_part = self.reg_lambda * self.l1_ratio * np.sign(theta[1:]) / m
            l2_grad_part = self.reg_lambda * (1 - self.l1_ratio) * theta[1:] / m
            gradient[1:] += l1_grad_part + l2_grad_part
            
        return gradient
        
    def _get_mini_batches(self, X, y, batch_size):
        m = X.shape[0]
        # Create shuffled indices to randomize the dataset order
        indices = np.random.permutation(m)
        # Calculate number of complete batches needed to cover the dataset
        num_batches = m // batch_size + (1 if m % batch_size != 0 else 0)
        # Initialize list to store all mini-batches
        batches = []
        # Split shuffled indices into batches
        for i in range(num_batches):
        # Calculate start and end index for the current batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, m)  # Prevent index overflow
            # Get indices for current batch
            batch_indices = indices[start_idx:end_idx]
            # Extract corresponding data and labels using batch indices
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            # Add current batch to the list of batches
            batches.append((X_batch, y_batch))
        return batches
                
    def _normal_equation(self, X, y):
        """
        Solve using normal equation (suitable for cases with fewer features)
        Including regularization term
        """
        m, n = X.shape
        
        if self.reg_type is None:
            # No regularization
            return np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
        elif self.reg_type == 'l2':
            # Analytical solution for Ridge regularization
            reg_matrix = np.eye(n)
            reg_matrix[0, 0] = 0  # Don't regularize the bias term
            return np.linalg.pinv(X.T.dot(X) + self.reg_lambda * reg_matrix).dot(X.T).dot(y)
        else:
            # L1 and elastic net don't have analytical solutions, use gradient descent
            return self._gradient_descent(X, y)
        
    def fit(self, X, y):
        """
        Train the linear regression model
        
        Parameters:
        - X: Training feature matrix
        - y: Training target vector
        """
        # Add bias term (intercept)
        X_with_bias = self._add_bias(X)
        m, n = X_with_bias.shape
        
        # Initialize model parameters
        self.theta = np.zeros(n)
        
        # Store initial learning rate for decay calculation
        initial_learning_rate = self.learning_rate
        
        # Use mini-batch gradient descent
        for iteration in range(self.max_iterations):
            # Apply learning rate decay if enabled
            if self.learning_rate_decay > 0:
                # Calculate current learning rate using decay formula: α_t = α_0 / (1 + decay · t)
                current_lr = initial_learning_rate / (1 + self.learning_rate_decay * iteration)
            else:
                current_lr = self.learning_rate
                
            # Generate mini-batches
            batches = self._get_mini_batches(X_with_bias, y, self.batch_size)
            
            for X_batch, y_batch in batches:
                # Calculate gradient
                gradient = self._compute_gradient(X_batch, y_batch, self.theta)
                
                # Update parameters using the current learning rate
                self.theta -= current_lr * gradient
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained model
        
        Parameters:
        - X: Feature matrix
        
        Returns:
        - Predicted values
        """
        if self.theta is None:
            raise ValueError("Model not yet trained")
        
        X_with_bias = self._add_bias(X)
        return X_with_bias.dot(self.theta)
    
    def score(self, X, y):
        """Calculate R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
        
    def rmse(self, X, y):
        """Calculate Root Mean Square Error"""
        y_pred = self.predict(X)
        return np.sqrt(np.mean((y - y_pred) ** 2))
        
    def mae(self, X, y):
        """Calculate Mean Absolute Error"""
        y_pred = self.predict(X)
        return np.mean(np.abs(y - y_pred))
        
    def plot_learning_curve(self):
        """Plot learning curve"""
        if not self.loss_history:
            raise ValueError("Model not yet trained")
            
        plt.figure(figsize=(12, 5))
        
        # Training loss
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_history, label='Training Loss')
        plt.title('Training Loss Curve')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        # Validation loss (if available)
        if self.val_loss_history:
            plt.subplot(1, 2, 2)
            plt.plot(self.val_loss_history, label='Validation Loss', color='r')
            plt.title('Validation Loss Curve')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    def get_feature_importance(self):
        """
        Get feature importance
        Returns a dictionary containing the weight of each feature
        """
        if self.theta is None:
            raise ValueError("Model not yet trained")
            
        importance = np.abs(self.theta[1:])  # Exclude bias term
        total = np.sum(importance)
        if total > 0:
            importance = importance / total
            
        return {f"feature_{i}": imp for i, imp in enumerate(importance)}

# Numerical stability test function
def test_numerical_stability(X, y):
    """
    Test numerical stability and compare with numpy.lstsq
    :return: (custom implementation parameters, numpy.lstsq parameters, condition number)
    """
    # Add bias term
    X = np.c_[np.ones(X.shape[0]), X]
    
    # Calculate condition number
    cond = np.linalg.cond(X.T.dot(X))
    
    # Custom implementation
    model = LinearRegression(method='gradient_descent')
    model.fit(X[:,1:], y)  # Remove bias term
    
    # Numpy implementation
    theta_np, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    
    return model.theta, theta_np, cond


class ARIMAModel:
    def __init__(self, p=1, d=1, q=1):
        """
        ARIMA model implementation
        :param p: Autoregressive order
        :param d: Differencing order
        :param q: Moving average order
        """
        self.p = p
        self.d = d
        self.q = q
        self.ar_params = None  # AR parameters
        self.ma_params = None  # MA parameters
        self.mean = None       # Mean term
        self.sigma2 = None     # Error variance
        self.history = None    # Training data
        self.resid = None      # Residual series
        
        # ADF test critical values (1%, 5%, 10%)
        self.adf_critical_values = {
            1: -3.43,
            5: -2.86,
            10: -2.57
        }
        
    def _compute_acf(self, data, nlags):
        """
        Calculate autocorrelation function (ACF)
        """
        n = len(data)
        mean = np.mean(data)
        data = data - mean
        acf = np.zeros(nlags + 1)
        variance = np.sum(data ** 2) / n
        
        for lag in range(nlags + 1):
            c = np.sum(data[lag:] * data[:(n-lag)]) / n
            acf[lag] = c / variance
        return acf
    
    def _compute_pacf(self, data, nlags):
        """
        Calculate partial autocorrelation function (PACF) using the Durbin-Levinson algorithm
        """
        pacf = np.zeros(nlags + 1)
        pacf[0] = 1.0
        
        # Calculate autocorrelation function
        acf = self._compute_acf(data, nlags)
        
        # Durbin-Levinson algorithm
        for k in range(1, nlags + 1):
            phi = np.zeros(k)
            phi[-1] = acf[k]
            
            for j in range(k-1):
                phi[j] = pacf[j+1]
            
            if k > 1:
                for j in range(k-1):
                    phi[j] = pacf[j+1] - phi[-1] * pacf[k-j-1]
            
            pacf[k] = phi[-1]
            
        return pacf
    
    def _estimate_ar_params(self, data):
        """
        Estimate AR parameters using Yule-Walker equations
        """
        if self.p == 0:
            return np.array([])
            
        r = self._compute_acf(data, self.p)
        R = np.zeros((self.p, self.p))
        
        for i in range(self.p):
            for j in range(self.p):
                R[i,j] = r[abs(i-j)]
                
        return np.linalg.solve(R, r[1:self.p+1])
    
    def _estimate_ma_params(self, residuals):
        """
        Estimate MA parameters using the innovation algorithm
        """
        if self.q == 0:
            return np.array([])
            
        n = len(residuals)
        theta = np.zeros(self.q)
        v = np.zeros(n)
        
        # Innovation algorithm
        for t in range(n):
            v[t] = residuals[t]
            if t > 0:
                for j in range(min(self.q, t)):
                    v[t] -= theta[j] * v[t-j-1]
        
        # Estimate MA parameters using least squares
        X = np.zeros((n-self.q, self.q))
        for i in range(self.q):
            X[:,i] = v[self.q-i-1:n-i-1]
        
        y = residuals[self.q:]
        theta = np.linalg.lstsq(X, y, rcond=None)[0]
        return theta
    
    def adf_test(self, data, max_lag=None):
        """
        Augmented Dickey-Fuller test for stationarity
        :param data: Time series data
        :param max_lag: Maximum lag order (default: 12*(n/100)^(1/4))
        :return: (ADF statistic, p-value, critical values)
        """
        n = len(data)
        if max_lag is None:
            max_lag = int(12 * (n / 100) ** (1/4))
            
        # Calculate first difference and lagged differences
        y_diff = np.diff(data)
        y_lagged = data[:-1]
        
        # Create design matrix with lagged level and constant
        X = np.column_stack([y_lagged[max_lag:], np.ones_like(y_lagged[max_lag:])])
        
        # Add lagged differences
        for lag in range(1, max_lag + 1):
            lagged_diff = y_diff[max_lag-lag:-lag]
            X = np.column_stack([X, lagged_diff])
            
        # Align dependent variable
        y = y_diff[max_lag:]
        
        # Remove rows with NaN
        valid = ~np.isnan(X).any(axis=1)
        X = X[valid]
        y = y[valid]
        
        # Perform regression
        try:
            theta = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(X).dot(y)
            
        # Calculate ADF statistic
        resid = y - X.dot(theta)
        sigma2 = np.sum(resid**2) / (len(y) - X.shape[1])
        adf_stat = theta[0] / np.sqrt(sigma2 * np.linalg.inv(X.T.dot(X))[0,0])
        
        # Calculate p-value using MacKinnon approximation
        p_value = 1.0 / (1.0 + np.exp(-0.00025 * (adf_stat**3) - 0.021 * (adf_stat**2) - 0.41 * adf_stat))
        
        return adf_stat, p_value, self.adf_critical_values
        
    def determine_differencing_order(self, data, max_diff=2):
        """
        Determine differencing order d
        :param data: Time series data
        :param max_diff: Maximum differencing order
        :return: Optimal differencing order d
        """
        # First check stationarity with ADF test
        adf_stat, p_value, _ = self.adf_test(data)
        
        if p_value <= 0.05:  # Stationary - no differencing needed
            return 0
            
        best_d = 0
        min_std = float('inf')
        
        for d in range(1, max_diff+1):
            diff_data = np.diff(data, n=d)
            
            # Check stationarity after differencing
            adf_stat, p_value, _ = self.adf_test(diff_data)
            if p_value <= 0.05:  # Stationary at this differencing order
                return d
                
            # Calculate standard deviation
            current_std = np.std(diff_data)
            if current_std < min_std:
                min_std = current_std
                best_d = d
                
        return best_d
        
    def difference(self, data, lag=1):
        """Differencing calculation"""
        return np.diff(data, n=lag)
        
    def inverse_difference(self, differenced, original, lag=1):
        """
        Inverse differencing operation
        :param differenced: Differenced data
        :param original: Original data
        :param lag: Differencing order
        :return: Restored data
        """
        inv_diff = differenced.copy()
        original_array = np.array(original)  # Convert to numpy array
        
        for i in range(lag):
            # Get the last lag+1 values of the original data
            last_values = original_array[-(i+1)]
            # Perform cumulative sum
            inv_diff = np.r_[last_values, inv_diff].cumsum()
        
        return inv_diff
        
    def fit(self, data):
        """
        Train ARIMA model
        :param data: Time series data (n_samples,)
        """
        self.history = data.copy()
        
        # Perform d-order differencing
        diff_data = data.copy()
        for _ in range(self.d):
            diff_data = self.difference(diff_data)
            
        # Remove mean
        self.mean = np.mean(diff_data)
        centered_data = diff_data - self.mean
        
        # Estimate AR parameters
        self.ar_params = self._estimate_ar_params(centered_data)
        
        # Calculate residuals for AR part
        residuals = centered_data.copy()
        for t in range(self.p, len(centered_data)):
            ar_pred = np.sum(self.ar_params * centered_data[t-self.p:t][::-1])
            residuals[t] = centered_data[t] - ar_pred
        
        # Estimate MA parameters
        self.ma_params = self._estimate_ma_params(residuals[self.p:])
        
        # Save final residuals
        self.resid = residuals[self.p:]
        
        # Estimate error variance
        self.sigma2 = np.var(self.resid)
    
    def predict(self, steps):
        """
        Predict future values
        :param steps: Number of prediction steps
        :return: Array of predicted values (steps,)
        """
        if self.ar_params is None or self.ma_params is None:
            raise ValueError("Model not trained yet.")
            
        # Initialize prediction results
        forecasts = np.zeros(steps)
        
        # Get latest observed and residual values for prediction
        history_array = np.array(self.history)  # Convert to numpy array
        last_values = history_array[-self.p:] if self.p > 0 else np.array([])
        last_resid = self.resid[-self.q:] if self.q > 0 else np.array([])
        
        # For each prediction step
        for t in range(steps):
            forecast = self.mean
            
            # AR component
            if self.p > 0:
                if t < self.p:
                    ar_terms = np.flip(last_values[-(self.p-t):])
                    if t > 0:
                        ar_terms = np.r_[ar_terms, forecasts[:t]]
                    else:
                        ar_terms = forecasts[t-self.p:t]
                forecast += np.sum(self.ar_params * ar_terms)
            
            # MA component
            if self.q > 0:
                if t < self.q:
                    ma_terms = last_resid[-(self.q-t):]
                    forecast += np.sum(self.ma_params[:len(ma_terms)] * ma_terms)
            
            forecasts[t] = forecast
        
        # Inverse differencing
        if self.d > 0:
            forecasts = self.inverse_difference(forecasts, self.history, self.d)
            
        # Ensure predicted results length matches requirements
        if len(forecasts) > steps:
            forecasts = forecasts[-steps:]
        elif len(forecasts) < steps:
            # If predicted results too short, use last value to fill
            padding = np.full(steps - len(forecasts), forecasts[-1])
            forecasts = np.concatenate([forecasts, padding])
        
        return forecasts
    
    def aic(self):
        """
        Calculate Akaike Information Criterion (AIC)
        """
        n = len(self.resid)
        k = self.p + self.q + 1  # Parameter count (including variance)
        aic = n * np.log(self.sigma2) + 2 * k
        return aic
    
    def bic(self):
        """
        Calculate Bayesian Information Criterion (BIC)
        """
        n = len(self.resid)
        k = self.p + self.q + 1  # Parameter count (including variance)
        bic = n * np.log(self.sigma2) + k * np.log(n)
        return bic
    
    def plot_diagnostics(self, figsize=(12, 8)):
        """
        Plot model diagnostics
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Residuals time series plot
        ax1.plot(self.resid)
        ax1.set_title('Residuals Time Series')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Residual')
        
        # Residuals histogram
        ax2.hist(self.resid, bins=30, density=True, alpha=0.5)
        x = np.linspace(self.resid.min(), self.resid.max(), 100)
        ax2.plot(x, stats.norm.pdf(x, 0, np.sqrt(self.sigma2)))
        ax2.set_title('Residuals Distribution')
        
        # ACF plot
        lags = min(40, len(self.resid)-1)
        acf = self._compute_acf(self.resid, lags)[1:]
        ax3.bar(range(1, len(acf)+1), acf)
        ax3.axhline(y=0, linestyle='-', color='black')
        ax3.axhline(y=1.96/np.sqrt(len(self.resid)), linestyle='--', color='gray')
        ax3.axhline(y=-1.96/np.sqrt(len(self.resid)), linestyle='--', color='gray')
        ax3.set_title('ACF of Residuals')
        
        # PACF plot
        pacf = self._compute_pacf(self.resid, lags)[1:]
        ax4.bar(range(1, len(pacf)+1), pacf)
        ax4.axhline(y=0, linestyle='-', color='black')
        ax4.axhline(y=1.96/np.sqrt(len(self.resid)), linestyle='--', color='gray')
        ax4.axhline(y=-1.96/np.sqrt(len(self.resid)), linestyle='--', color='gray')
        ax4.set_title('PACF of Residuals')
        
        plt.tight_layout()
        plt.show()

    def plot_acf_pacf(self, data, lag_num=40, title=""):
        """
        Plot ACF and PACF graphs
        :param data: Time series data
        :param lag_num: Maximum lag order
        :param title: Chart title
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        
        # Calculate and plot ACF
        acf = self._compute_acf(data, lag_num)[1:]  # Use new _compute_acf method
        ax[0].stem(range(1, len(acf)+1), acf)
        ax[0].axhline(y=0, color='black', linestyle='-')
        ax[0].axhline(y=1.96/np.sqrt(len(data)), color='gray', linestyle='--')
        ax[0].axhline(y=-1.96/np.sqrt(len(data)), color='gray', linestyle='--')
        ax[0].set_title('ACF')
        
        # Calculate and plot PACF
        pacf = self._compute_pacf(data, lag_num)[1:]  # Use new _compute_pacf method
        ax[1].stem(range(1, len(pacf)+1), pacf)
        ax[1].axhline(y=0, color='black', linestyle='-')
        ax[1].axhline(y=1.96/np.sqrt(len(data)), color='gray', linestyle='--')
        ax[1].axhline(y=-1.96/np.sqrt(len(data)), color='gray', linestyle='--')
        ax[1].set_title('PACF')
        
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

def grid_search_arima(data, p_range, d_range, q_range):
    """
    Grid search to find the best ARIMA model
    
    Parameters:
        data: Time series data
        p_range: AR order range
        d_range: Differencing order range
        q_range: MA order range
        
    Returns:
        Best model and its parameters
    """
    best_aic = float('inf')
    best_model = None
    best_params = None
    
    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = ARIMAModel(p=p, d=d, q=q)
                    model.fit(data)
                    aic = model.aic()
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_model = model
                        best_params = {'p': p, 'd': d, 'q': q, 'aic': aic}
                        
                except Exception as e:
                    continue
                    
    return best_model, best_params

class ProphetModel:
    def __init__(self, daily_seasonality=False, weekly_seasonality=False, 
                 yearly_seasonality=False, changepoint_range=1, 
                 changepoint_prior_scale=0.5, holidays=None, 
                 seasonality_mode='multiplicative'):
        """
        Facebook Prophet model wrapper
        """
        self.params = {
            'daily_seasonality': daily_seasonality,
            'weekly_seasonality': weekly_seasonality,
            'yearly_seasonality': yearly_seasonality,
            'changepoint_range': changepoint_range,
            'changepoint_prior_scale': changepoint_prior_scale,
            'seasonality_mode': seasonality_mode
        }
        self.holidays = holidays
        self.model = None

    def add_seasonality(self, name, period, fourier_order, mode='multiplicative', prior_scale=0.5):
        """Add custom seasonality"""
        if not hasattr(self, 'custom_seasonalities'):
            self.custom_seasonalities = []
        self.custom_seasonalities.append({
            'name': name,
            'period': period,
            'fourier_order': fourier_order,
            'mode': mode,
            'prior_scale': prior_scale
        })

    def fit(self, train_df):
        """Train the model"""
        from prophet import Prophet  # Modified here, from fbprophet to prophet
        
        self.model = Prophet(**self.params)
        
        if self.holidays is not None:
            self.model.holidays = self.holidays
            
        if hasattr(self, 'custom_seasonalities'):
            for seasonality in self.custom_seasonalities:
                self.model.add_seasonality(**seasonality)
                
        self.model.fit(train_df)

    def predict(self, periods, freq='D'):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
            
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        return self.model.predict(future)

    def plot_forecast(self, forecast, xlabel='Date', ylabel='Value'):
        """Plot forecast"""
        return self.model.plot(forecast, xlabel=xlabel, ylabel=ylabel)

    def plot_components(self, forecast):
        """Plot components"""
        return self.model.plot_components(forecast)

def prophet_modeling(result, cryptocurrency, train, test, holidays_df, 
                    period_days, fourier_order_seasonality, 
                    forecasting_period, name_model, type_data):
    """
    Complete Prophet modeling workflow
    """
    model = ProphetModel(
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
        changepoint_range=1,
        changepoint_prior_scale=0.5,
        holidays=holidays_df,
        seasonality_mode='multiplicative'
    )
    
    model.add_seasonality(
        name='seasonality',
        period=period_days,
        fourier_order=fourier_order_seasonality,
        mode='multiplicative',
        prior_scale=0.5
    )
    
    model.fit(train)
    forecast = model.predict(forecasting_period)
    
    # Visualization
    fig1 = model.plot_forecast(
        forecast, 
        ylabel=f"{name_model} for {cryptocurrency}"
    )
    fig2 = model.plot_components(forecast)
    
    # Save results
    ypred = forecast['yhat'][-forecasting_period:]
    n = len(result)
    result.loc[n, 'name_model'] = f"Prophet_{name_model}"
    result.loc[n, 'type_data'] = type_data
    result.at[n, 'params'] = [period_days, fourier_order_seasonality]
    result.at[n, 'ypred'] = ypred
    
    return result, ypred




