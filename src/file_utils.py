import os

import pandas as pd

FILE_CSV_5000 = "fake_csv_5000.csv"
FILE_CSV_50000 = "fake_csv_50000.csv"
FILE_DDOS_CSV = "ddos_csv_100000.csv"

def get_file():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build the absolute path of the file
    file_path = os.path.join(script_dir, '..', 'data', FILE_DDOS_CSV )
    print(file_path)
    # Check if the files exists
    if os.path.exists(file_path):
        # Read the CSV
        dataset = pd.read_csv(file_path)
        return dataset