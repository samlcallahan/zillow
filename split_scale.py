import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler
from prep import zillow

zillow_x = zillow.loc[:, ['beds','baths', 'sqft']].set_index([zillow.parcelid])
zillow_y = pd.DataFrame(zillow.loc[:, 'value']).set_index([zillow.parcelid])

seed = 43

def split_my_data(df, train_pct, random_seed):
    train, test = train_test_split(df, train_size = train_pct, random_state = random_seed)
    return train, test

def standard_scaler(train, test):
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled

def scale_inverse(scaler, df):
    unscaled = pd.DataFrame(scaler.inverse_transform(df), columns=df.columns.values).set_index([df.index.values])
    return unscaled

def uniform_scaler(train, test):
    scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=123, copy=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled

def gaussian_scaler(train, test):
    scaler = PowerTransformer(method='yeo-johnson', standardize=False, copy=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled

def min_max_scaler(train, test):
    scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled

def iqr_robust_scaler(train, test):
    scaler = RobustScaler(quantile_range=(25.0,75.0), copy=True, with_centering=True, with_scaling=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled

train_x, test_x = split_my_data(zillow_x, .8, seed)
train_y, test_y = split_my_data(zillow_y, .8, seed)

scaler, scale_train, scale_test = standard_scaler(train_x, test_x)