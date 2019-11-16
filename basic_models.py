import os
# Import basic housing data
from data_import import training_housing_data, test_housing_data
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor # RandomForestRegressor from sklearn
from sklearn.metrics import mean_absolute_error # calculating error for validation
from sklearn.model_selection import train_test_split # splitting data into tests sets

# Use the below line to peak data
# training_housing_data.head()

# Basic numeric only features
basic_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
# Features including non-numeric values
robust_features = ['LotArea', 'LotShape', 'Utilities', 'OverallQual', 'OverallCond', 
                    'YearBuilt','YearRemodAdd', 'CentralAir', 'Electrical', 'Heating', 
                    'FullBath', 'HalfBath','Fireplaces','GarageType', 'PoolArea']
# An alternative way for selecting numeric colums
numeric_features = [colname for colname in training_housing_data if training_housing_data[colname].dtype in ['int64', 'float64']]

# Remove rows with missing target (NaN y) and seperator from predictors
training_housing_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = training_housing_data.SalePrice
training_housing_data.drop(['SalePrice'], axis=1, inplace=True)

# Select numeric predictors
X = training_housing_data[numeric_features].copy()
X_test = test_housing_data[numeric_features].copy()


