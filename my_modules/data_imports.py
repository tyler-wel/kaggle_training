import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from termcolor import colored # colored prints

fifa_path = ("./datasets/data-vis/fifa.csv")
if not os.path.exists(fifa_path):
    failed_import(fifa_path)

cancer_b_path = ("./datasets/data-vis/cancer_b.csv")
if not os.path.exists(cancer_b_path):
    failed_import(cancer_b_path)

cancer_m_path = ("./datasets/data-vis/cancer_m.csv")
if not os.path.exists(cancer_m_path):
    failed_import(cancer_m_path)
    
candy_path = ("./datasets/data-vis/candy.csv")
if not os.path.exists(candy_path):
    failed_import(candy_path)

flight_delays_path = ("./datasets/data-vis/flight_delays.csv")
if not os.path.exists(flight_delays_path):
    failed_import(flight_delays_path)
    
ign_scores_path = ("./datasets/data-vis/ign_scores.csv")
if not os.path.exists(ign_scores_path):
    failed_import(ign_scores_path)

insurance_path = ("./datasets/data-vis/insurance.csv")
if not os.path.exists(insurance_path):
    failed_import(insurance_path)

iris_setosa_path = ("./datasets/data-vis/iris_setosa.csv")
if not os.path.exists(iris_setosa_path):
    failed_import(iris_setosa_path)

iris_versicolor_path = ("./datasets/data-vis/iris_versicolor.csv")
if not os.path.exists(iris_versicolor_path):
    failed_import(iris_versicolor_path)

iris_virginica_path = ("./datasets/data-vis/iris_virginica.csv")
if not os.path.exists(iris_virginica_path):
    failed_import(iris_virginica_path)

iris_path = ("./datasets/data-vis/iris.csv")
if not os.path.exists(iris_path):
    failed_import(iris_path)

museum_path = ("./datasets/data-vis/museum_visitors.csv")
if not os.path.exists(museum_path):
    failed_import(museum_path)

spotify_path = ("./datasets/data-vis/spotify.csv")
if not os.path.exists(spotify_path):
    failed_import(spotify_path)

def failed_import(file_name):
    print(colored(file_name + " doesn't exist", 'red'))
    
# Note:
# - index_col param is generally the name of the first column, used to index our data
    
def import_fifa():
    fifa_data = pd.read_csv(fifa_path, index_col="Date", parse_dates=True)
    print(colored('Fifa data imported', 'green'))
    return fifa_data