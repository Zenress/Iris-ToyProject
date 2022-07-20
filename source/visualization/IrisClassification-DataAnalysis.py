from base64 import encode
import yaml
from sklearn import preprocessing
import pandas as pd
import pandas_profiling as pp

DATASET_PATH = 'source/data/'
REPORT_PATH = 'reports/'

#Using configuration file for variables
with open("./configuration/config.yaml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

def file_report(altered_data_df):
    """
    Write profile report.
    
    Makes a profile report using pandas_profiling and Saves the report under reports folder as a html file

    Args:
        altered_data_df (pandas.DataFrame): Holds the altered dataframe
    """
    data_profile = pp.ProfileReport(altered_data_df, title=cfg["report_settings"]["title"], explorative=True)
    data_profile.to_file(REPORT_PATH + cfg["iris_analysis_report_name"])



def main():
    """
    Execute on initialization.
    
    
    
    """
    #Assigning custom column headers while reading the csv file
    data_df = pd.read_csv(DATASET_PATH + cfg["dataset_name"], header=None, names=cfg["column_names"])
    print(data_df)

    label_encoder = preprocessing.LabelEncoder()
    
    altered_df = encode_column(data_df[cfg["label_name"]], label_encoder, data_df)
    
    file_report(label_encoder, altered_df)
    
if __name__ == "__main__":
    main()