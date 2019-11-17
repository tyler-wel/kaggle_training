# Take our intermediate training a step forward with gradient boosting
import os
# Import basic housing data
from __data_import import training_housing_data as X_full, test_housing_data as X_test_full
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Remove rows with NaN target (SalePrice), and separate target from predictor data
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

from sklearn.model_selection import train_test_split
# Split up training/validation data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [
    colname for colname in X_train_full.columns 
    if X_train_full[colname].nunique() < 10 and X_train_full[colname].dtype == "object"
]

# Select numeric columns
numeric_cols = [
    colname for colname in X_train_full.columns 
    if X_train_full[colname].dtype in ['int64', 'float64']
]

# Keep selected columns only
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# For training purpose, encode data using pandas to shorten code
# Note its better to stick to one-hot-encoding in the field
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)

# XGBoost !
from xgboost import XGBRegressor

# XGBoost (useful) params:
#   booster [default= gbtree], gbtree, gblinear, dart
#   nthread [default to max], number of CPU threads to use
#   random_state [default= 0], random number seed
#   n_esimators [default= ], number of trees (models) to fit (typically 100~1000)
#  gbtree params:
#   learning_rate [default= 0.3], learning rate used to strink feature weights to counter overfitting
#   max_depth [default= 6], maximum depth of a tree

# Define our model using params from above
# Testing should be done to figure out what a good num of n_estimators are
my_model_2 = XGBRegressor(n_estimators=5000, learning_rate=0.05, random_state=0) # Your code here

# Fit the model
my_model_2.fit(X_train, y_train) # Your code here

# Get predictions
predictions_2 = my_model_2.predict(X_valid) # Your code here

# Calculate MAE
from sklearn.metrics import mean_absolute_error
mae_2 = mean_absolute_error(predictions_2, y_valid) # Your code here

# Uncomment to print MAE
print("Mean Absolute Error:" , mae_2)

###  Extra Notes ###
#   A good way to get good training/predictions is to use a high n_estimators a long with early_stopping_rounds.
#   Early stopping rounds will automatically stop the training (before n_estimators of models is reached) when the validation score stops improving as much
#   Combine this with a good learning rate to obtain accurate predictions
# my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
# my_model.fit(X_train, y_train, 
#              early_stopping_rounds=5, 
#              eval_set=[(X_valid, y_valid)],
#              verbose=False)
#
#   You can use XGB in your pipeline!