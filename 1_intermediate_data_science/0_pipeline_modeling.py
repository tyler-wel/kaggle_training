# This file shows more intermediate ideas and data modeling from kaggle learning module(s)
import os
# Import basic housing data
from __data_import import training_housing_data as X_full, test_housing_data as X_test_full
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# From here, we will HotEncoding for categorical data and pipelines to help us stay organized
# Get our data and drop rows with NaN target
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# Split data for train/validation
from sklearn.model_selection import train_test_split # splitting data into tests sets
X_train_full, X_validation_full, y_train, y_validation = train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [
    colname for colname in X_train_full.columns 
    if X_train_full[colname].nunique() < 10 and X_train_full[colname].dtype == "object"
]

# Select numerical columns
numerical_cols = [
    colname for colname in X_train_full.columns 
    if X_train_full[colname].dtype in ['int64', 'float64']
]

# Narrow down data to categorical + numerical
features = categorical_cols + numerical_cols
X_train = X_train_full[features].copy()
X_validation = X_validation_full[features].copy()
X_test = X_test_full[features].copy()

# print(X_train.head())

# Now that we have our data setup, it's time to setup our preprocessing pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

numerical_transformer = SimpleImputer(strategy='mean')
categorical_transform = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),     # transform the numerical_cols with the numerical transformer
        ('cat', categorical_transform, categorical_cols)    # transform the categorical_cols with the categorical transformer
    ]
)

# Define our model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Setup our pipeline
my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Preprocess of training data and fit the model
my_pipeline.fit(X_train, y_train)


# Preprocess of test data and fit the model
predictions = my_pipeline.predict(X_validation)

# Evaluate the model (basic scoring)
# score = mean_absolute_error(y_validation, predictions) 
# print("MAE: ", score)

# Evaluate the model (cross validation)
# from sklearn.model_selection import cross_val_score
# Multiply by -1 since sklearn calculates *negative* MAE
# scores = -1 * cross_val_score(my_pipeline, X_train, y_train,
#                               cv=5,
#                               scoring='neg_mean_absolute_error')
# print("Average MAE score: " + "{0:,.2f}".format(scores.mean()))

# Evaluate the model (cross validation from model_scoring)
from my_modules import __model_scoring
mean_score = __model_scoring.score_pipeline_validation(my_pipeline, X_train, y_train)
print(mean_score)
print("Average MAE score: " + "{0:,.2f}".format(mean_score))

# Make real predictions
real = my_pipeline.predict(X_test)

# Save the prediction values for submission
output = pd.DataFrame({'Id': X_test_full.index,'SalePrice': real})
print("Mean SalePrice: " + "{0:,.2f}".format(output['SalePrice'].mean()))