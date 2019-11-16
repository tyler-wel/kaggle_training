# Here, we will use pipelines and cross validations to show one way to test different models and determine which is "best"
import os
# Import basic housing data
from data_import import training_housing_data, test_housing_data
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# From here, we will HotEncoding for categorical data and pipelines to help us stay organized
# Get our data and drop rows with NaN target
training_housing_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = training_housing_data.SalePrice
training_housing_data.drop(['SalePrice'], axis=1, inplace=True)

# For this test, lets select numeric columns only
numeric_cols = [colname for colname in training_housing_data.columns if training_housing_data[colname].dtype in ['int64', 'float64']]
X = training_housing_data[numeric_cols].copy()
X_test = test_housing_data[numeric_cols].copy()

from model_scoring import get_pipeline_score
# Using get_pipeline_score from model_scoring, test different values for the number of trees (n_estimators) of a random forest model.
results = {}
for i in range(50, 401, 50):
    results[i] = get_pipeline_score(i, X, y)

# plot our results to visualize the different models
import matplotlib.pyplot as plt

plt.plot(list(results.keys()), list(results.values()))
plt.show()