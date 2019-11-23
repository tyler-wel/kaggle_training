def setup():
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