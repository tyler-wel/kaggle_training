import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# print(os.path.exists("datasets/"))
# print(os.listdir("./"))

if(os.path.exists("kaggle_training/datasets/housing-data/")):
    print("kaggle_training/datasets/housing-data/")
    training_file_path = "kaggle_training/datasets/housing-data/train.csv"
    test_file_path = "kaggle_training/datasets/housing-data/test.csv"
elif(os.path.exists("datasets/housing-data/")):
    print("datasets/housing-data/")
    training_file_path = "datasets/housing-data/train.csv"
    test_file_path = "datasets/housing-data/test.csv"

training_housing_data = pd.read_csv(training_file_path, index_col='Id')
test_housing_data = pd.read_csv(test_file_path, index_col='Id')
