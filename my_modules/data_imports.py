import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from termcolor import colored # colored prints

fifa_path = ("./datasets/data-vis/fifa.csv")
cancer_b_path = ("./datasets/data-vis/cancer_b.csv")
cancer_m_path = ("./datasets/data-vis/cancer_m.csv")
candy_path = ("./datasets/data-vis/candy.csv")
flight_delays_path = ("./datasets/data-vis/flight_delays.csv")
ign_scores_path = ("./datasets/data-vis/ign_scores.csv")
insurance_path = ("./datasets/data-vis/insurance.csv")
iris_setosa_path = ("./datasets/data-vis/iris_setosa.csv")
iris_versicolor_path = ("./datasets/data-vis/iris_versicolor.csv")
iris_virginica_path = ("./datasets/data-vis/iris_virginica.csv")
iris_path = ("./datasets/data-vis/iris.csv")
museum_path = ("./datasets/data-vis/museum_visitors.csv")
spotify_path = ("./datasets/data-vis/spotify.csv")
wine_path = ("./datasets/pandas/winemag.csv")
youtube_ca_path = ("./datasets/pandas/CAvideos.csv")
youtube_gb_path = ("./datasets/pandas/GBvideos.csv")    
kickstarter_2018_path = ("./datasets/feature_engineering/ks-projects-201801.csv")
ad_clicks_path = ("./datasets/feature_engineering/train_sample.csv")

# Note:
# - index_col param is generally the name of the first column, used to index our data
    
def check_existance(file_name):
    if not os.path.exists(file_name):
        print(colored(file_name + " doesn't exist", 'red'))
        return False
    else:
        return True
    
def import_fifa():
    if not check_existance(fifa_path):
        return None
    fifa_data = pd.read_csv(fifa_path, index_col="Date", parse_dates=True)
    print(colored('Fifa data imported', 'green'))
    return fifa_data

def import_spotify():
    if not check_existance(spotify_path):
        return None
    spotify_data = pd.read_csv(spotify_path, index_col="Date", parse_dates=True)
    print(colored("Spotify data imported!", 'green'))
    return spotify_data

def import_flight_delays():
    if not check_existance(flight_delays_path):
        return None
    flight_delay_data = pd.read_csv(flight_delays_path, index_col="Month")
    print(colored("Flight delay data imported!", 'green'))
    return flight_delay_data

def import_insurance_data():
    if not check_existance(insurance_path):
        return None
    ins_data = pd.read_csv(insurance_path)
    print(colored("Insurance data imported!", 'green'))
    return ins_data

def import_flower_data():
    if not check_existance(iris_path):
        return None
    flw = pd.read_csv(iris_path, index_col="Id")
    print(colored("Iris flower data imported!", 'green'))
    return flw

def import_iris_set():
    if not check_existance(iris_setosa_path):
        return None
    iris_s = pd.read_csv(iris_setosa_path, index_col=("Id"))
    print(colored("Iris setosa data imported", 'green'))
    return iris_s
    
def import_iris_ver():
    if not check_existance(iris_versicolor_path):
        return None
    iris_ver = pd.read_csv(iris_versicolor_path, index_col=("Id"))
    print(colored("Iris versicolo data imported", 'green'))
    return iris_ver

def import_iris_vir():
    if not check_existance(iris_virginica_path):
        return None
    iris_vir = pd.read_csv(iris_virginica_path, index_col=("Id"))
    print(colored("Iris virignica data imported", 'green'))
    return iris_vir  

def import_wine_data():
    if not check_existance(wine_path):
        return None
    wine_data = pd.read_csv(wine_path, index_col=0)
    print(colored("Wine data imported", 'green'))
    return wine_data

def import_youtube_ca_data():
    if not check_existance(youtube_ca_path):
        return None
    youtube_ca = pd.read_csv(youtube_ca_path)
    print(colored("CAVideos data imported", 'green'))
    return youtube_ca

def import_youtube_gb_data():
    if not check_existance(youtube_gb_path):
        return None
    youtube_gb = pd.read_csv(youtube_gb_path)
    print(colored("GBVideos data imported", 'green'))
    return youtube_gb

def import_kickstarter_2018_data():
    if not check_existance(kickstarter_2018_path):
        return None
    kickstarter = pd.read_csv(kickstarter_2018_path, parse_dates=['deadline', 'launched'])
    print(colored("Kickstarter 2018 data imported", 'green'))
    return kickstarter

def import_ad_clicks_data():
    if not check_existance(ad_clicks_path):
        return None
    ad_clicks = pd.read_csv(ad_clicks_path, parse_dates=['click_time'])
    print(colored("Ad Click data imported", 'green'))
    return ad_clicks