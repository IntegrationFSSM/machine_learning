import os
import pandas as pd
from sklearn.datasets import fetch_openml

def load_data(save_path="data/titanic.csv"):
    """
    Fetches the Titanic dataset from OpenML and saves it to a CSV file.
    """
    print("Loading data from OpenML...")
    # Version 1 is the standard "Titanic" dataset
    titanic = fetch_openml('titanic', version=1, as_frame=True)
    df = titanic.frame
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"Saving data to {save_path}...")
    df.to_csv(save_path, index=False)
    print("Data saved successfully.")
    return df

if __name__ == "__main__":
    load_data()
