import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

training_file_path = "datasets/housing-data/train.csv"
test_file_path = "datasets/housing-data/test.csv"

training_housing_data = pd.read_csv(training_file_path)
test_housing_data = pd.read_csv(test_file_path)