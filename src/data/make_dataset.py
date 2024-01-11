import pathlib
import yaml
import sys
import pandas as pd
from sklearn.model_selection import train_test_split



def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df



def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    input_file = sys.argv[1]
    data_path = home_dir.as_posix() + input_file

    data = load_data(data_path)

    return data

if __name__ == "__main__":
    main()

