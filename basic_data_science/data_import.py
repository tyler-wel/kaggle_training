import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# print(os.path.exists("../projects/kaggle_training/datasets/"))
# print(os.listdir("./"))

training_file_path = "kaggle_training/datasets/housing-data/train.csv"
test_file_path = "kaggle_training/datasets/housing-data/test.csv"

training_housing_data = pd.read_csv(training_file_path, index_col='Id')
test_housing_data = pd.read_csv(test_file_path, index_col='Id')