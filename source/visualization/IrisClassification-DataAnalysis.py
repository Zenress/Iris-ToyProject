import yaml
from sklearn import preprocessing
import pandas as pd
import pandas_profiling as pp

DATASET_PATH = 'source/data/'

#Using configuration file for variables
with open("./configuration/config.yaml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

#Assigning custom column headers while reading the csv file
data_df = pd.read_csv(DATASET_PATH + cfg["dataset_name"], header=None, names=cfg["column_names"])
print(data_df)

label_encoder = preprocessing.LabelEncoder()

def data_analysis():
    #Encoding the last column header to an int datatype
    data_df[cfg["label_name"]] = label_encoder.fit_transform(data_df[cfg["label_name"]])
    print(data_df)

    data_profile = pp.ProfileReport(data_df, title=cfg["report_settings"]["title"], explorative=True)
    data_profile.to_file("reports/"+cfg["iris_analysis_report_name"])

data_analysis()