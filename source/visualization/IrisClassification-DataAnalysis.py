import yaml
from sklearn import preprocessing
import pandas as pd
import pandas_profiling as pp
from sklearn.utils import shuffle
#Using configuration file for variables
with open("./configuration/config.yaml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

label_encoder = preprocessing.LabelEncoder()

#Column_names made to assign as headers of the dataset
column_names = [cfg["column_names"]["column_nr1"],cfg["column_names"]["column_nr2"]
                ,cfg["column_names"]["column_nr3"],cfg["column_names"]["column_nr4"]
                ,cfg["column_names"]["column_nr5"]]
#Assigning custom column headers while reading the csv file
iris_df = pd.read_csv(cfg["file_paths"]["dataset_path"], header=None, names=column_names)
print(iris_df)


#Encoding the last column header to an int datatype
iris_class_names = iris_df[cfg["column_names"]["column_nr5"]]
iris_df[cfg["column_names"]["column_nr5"]] = label_encoder.fit_transform(iris_df[cfg["column_names"]["column_nr5"]])
print(iris_df)


iris_profile = pp.ProfileReport(iris_df, title="Iris Data Profile Report", explorative=True)
iris_profile.to_file(cfg["file_paths"]["iris_analysis_report_path"])

iris_shuffled_df = shuffle(iris_df, random_state=123)

iris_shuffled_profile = pp.ProfileReport(iris_shuffled_df, title="Iris Shuffled Dataset Profile Report", explorative=True)
iris_shuffled_profile.to_file(cfg["file_paths"]["irisshuffled_analysis_report_path"])