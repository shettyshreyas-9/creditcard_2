import pathlib
import yaml
import sys
import pandas as pd
from sklearn.model_selection import train_test_split



def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df

def save_data(data,output_path):
    sdf= data.to_csv(output_path+'/train.csv')

def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    dvc_yaml = yaml.safe_load(open('dvc.yaml'))['stages']

    input_file= sys.argv[1]
    data_path = home_dir.as_posix() + input_file

    
    # input_file1 = dvc_yaml['make_dataset']['deps'][0]
    # input_file2 = '/data/raw/creditcard.csv'
    # data_path2 = home_dir.as_posix() + input_file2
    # data_path1 = home_dir.as_posix() + input_file1
    
    output_path = home_dir.as_posix() + '/data/processed'

    data = load_data(data_path)

    save_data(data,output_path)

    #return data

    # print(curr_dir)
    # print(home_dir)
    # print(data_path2)
    # print(input_file1)

    # print(input_file2)
    print('',data_path,'', sep='\n')
    print('',output_path,'', sep='\n' )

    # print(input_file)

if __name__ == "__main__":
    main()

