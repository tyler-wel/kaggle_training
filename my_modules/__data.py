import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from termcolor import colored # colored prints

fifa_data = ""


fifa_path = ("./datasets/data-vis/fifa.csv")
if not os.path.exists(fifa_path):
    print(colored("Fifa path doesn't exist", 'red'))
fifa_data = pd.read_csv(fifa_path, index_col="Date", parse_dates=True)
print(colored('Fifa data imported', 'green'))  

def test():
    print('hey')