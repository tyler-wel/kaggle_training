# This file shows very basic data science modeling, based on kaggle basic data science learning module

import os
# Import basic housing data
from __data_import import training_housing_data, test_housing_data
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Use the below line to peak data
# print(training_housing_data.head())

# Basic numeric only features
basic_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
# Features including non-numeric values
robust_features = ['LotArea', 'LotShape', 'Utilities', 'OverallQual', 'OverallCond', 
                    'YearBuilt','YearRemodAdd', 'CentralAir', 'Electrical', 'Heating', 
                    'FullBath', 'HalfBath','Fireplaces','GarageType', 'PoolArea']

# # Remove rows with missing target (NaN y) and seperator from predictors
training_housing_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = training_housing_data.SalePrice
training_housing_data.drop(['SalePrice'], axis=1, inplace=True)

# An alternative way for selecting numeric colums
numeric_features = [
    colname for colname in training_housing_data 
    if training_housing_data[colname].dtype in ['int64']
]

# Select numeric predictors
X = training_housing_data[numeric_features].copy()
X_test = test_housing_data[numeric_features].copy()

# Drop various NaN columns for test (note this is not a very /good/ idea)
missing_cols = [
    col for col in X_test.columns
    if X_test[col].isnull().any()
]

X.drop(missing_cols, axis = 1, inplace=True)
X_test.drop(missing_cols, axis = 1, inplace=True)

# Basic RandomForestRegressor training
from sklearn.ensemble import RandomForestRegressor # RandomForestRegressor from sklearn
from sklearn.metrics import mean_absolute_error # calculating error for validation
from sklearn.model_selection import train_test_split # splitting data into tests sets
# Split into training/test sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
model = RandomForestRegressor(n_estimators=50, criterion='mae', random_state=0)

# Fit Model
model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE: {:,.0f}".format(val_mae))

# If feeling confident about validation, setup model on all data from test set
real_predictions = model.predict(X_test) 

# Save the prediction values for submission
output = pd.DataFrame({'Id': test_housing_data.index,'SalePrice': real_predictions})
print("Mean SalePrice: " + "{0:,.2f}".format(output['SalePrice'].mean()))


# Extra: Use a scoring model to test different parameters for different models
# For example, define different models
# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]
print("Scoring various models: ")
from __model_scoring import score_basic_model
for i in range(0, len(models)):
    mae = score_basic_model(models[i], train_X, val_X, train_y, val_y)
    print("Model {} MAE: {}".format(i+1, mae))