import yaml
from sklearn import preprocessing
import pandas as pd
import pandas_profiling as pp
#Using configuration file for variables
with open("./configuration/config.yaml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

#Assigning custom column headers while reading the csv file
dataset_df = pd.read_csv(cfg["file_paths"]["dataset_path"], header=None, names=cfg["column_names"])
print(dataset_df)

#Labels based on the single categorical column in the dataset
labels = dataset_df.select_dtypes(include=['object'])

label_encoder = preprocessing.LabelEncoder()

#Encoding the last column header to an int datatype
iris_class_names = dataset_df[str(labels.columns.values[0])]
dataset_df[str(labels.columns.values[0])] = label_encoder.fit_transform(dataset_df[str(labels.columns.values[0])])
print(dataset_df)


data_profile = pp.ProfileReport(dataset_df, title=cfg["report_settings"]["title"], explorative=True)
data_profile.to_file(cfg["file_paths"]["iris_analysis_report_path"])
