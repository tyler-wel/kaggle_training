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

wine_path = ("./datasets/pandas/winemag.csv")
if not os.path.exists(wine_path):
    failed_import(wine_path)

def failed_import(file_name):
    print(colored(file_name + " doesn't exist", 'red'))
    
# Note:
# - index_col param is generally the name of the first column, used to index our data
    
def import_fifa():
    fifa_data = pd.read_csv(fifa_path, index_col="Date", parse_dates=True)
    print(colored('Fifa data imported', 'green'))
    return fifa_data

def import_spotify():
    spotify_data = pd.read_csv(spotify_path, index_col="Date", parse_dates=True)
    print(colored("Spotify data imported!", 'green'))
    return spotify_data

def import_flight_delays():
    flight_delay_data = pd.read_csv(flight_delays_path, index_col="Month")
    print(colored("Flight delay data imported!", 'green'))
    return flight_delay_data

def import_insurance_data():
    ins_data = pd.read_csv(insurance_path)
    print(colored("Insurance data imported!", 'green'))
    return ins_data

def import_flower_data():
    flw = pd.read_csv(iris_path, index_col="Id")
    print(colored("Iris flower data imported!", 'green'))
    return flw

def import_iris_set():
    iris_s = pd.read_csv(iris_setosa_path, index_col=("Id"))
    print(colored("Iris setosa data imported", 'green'))
    return iris_s
    
def import_iris_ver():
    iris_ver = pd.read_csv(iris_versicolor_path, index_col=("Id"))
    print(colored("Iris versicolo data imported", 'green'))
    return iris_ver

def import_iris_vir():
    iris_vir = pd.read_csv(iris_virginica_path, index_col=("Id"))
    print(colored("Iris virignica data imported", 'green'))
    return iris_vir  

def import_wine_data():
    wine_data = pd.read_csv(wine_path, index_col=0)
    print(colored("Wine data imported", 'green'))
    return wine_data