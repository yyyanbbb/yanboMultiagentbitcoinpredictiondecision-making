import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def check_stationarity(series, window=12, title=''):
    # 1. Calculate rolling statistics
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    # 2. Perform ADF test
    print('-'*50)
    print(f'ADF Test Results - {title}')
    result = adfuller(series.values)
    
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    
    # 3. Determine stationarity
    if (result[1] <= 0.05) and (result[0] < result[4]['5%']):
        print("\033[32mConclusion: The series is stationary\033[0m")
    else:
        print("\033[31mConclusion: The series is non-stationary\033[0m")
    
    return result



