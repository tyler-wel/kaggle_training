import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from termcolor import colored # colored prints

if not (os.path.exists("datasets/housing-data/")):
    print(colored("Paths aren't setup correctly", 'red'))
else:
    housing_training_path = "datasets/housing-data/train.csv"
    housing_test_path = "datasets/housing-data/test.csv"
    training_housing_data = pd.read_csv(housing_training_path, index_col='Id')
    test_housing_data = pd.read_csv(housing_test_path, index_col='Id')

def setup_seaborn():
    from termcolor import colored

    import pandas as pd
    pd.plotting.register_matplotlib_converters()

    import matplotlib.pyplot as plt
    # %matplotlib inline

    import seaborn as sns
    print(colored("Seaborn Setup Complete!", 'green'))

    import os
    filepath = ("./datasets/data-vis/fifa.csv")
    if os.path.exists(filepath):
        print(colored("File exists, ready to go!", 'green'))