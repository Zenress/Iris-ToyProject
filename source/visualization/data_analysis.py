"""
Data Analysis file for creating a profile report and analysing the dataset used.
"""
from pathlib import Path
import yaml
import pandas as pd
import pandas_profiling as pp


DATASET_PATH = 'source/data/'
REPORT_PATH = 'reports/'
CONFIG_PATH = 'configuration/config.yaml'

def file_report(
    altered_data_df,
    report_title,
    report_name
    ):
    """
    Write profile report.

    Makes a profile report using pandas_profiling and,
    Saves the report under reports folder as a html file.

    Args:
        altered_data_df (pandas.DataFrame): Holds the altered dataframe.
        report_title (str) string gotten from the dictionary it is in.
        report_name (str) name of the report gotten from the configuration file.
    """
    report_full_path = Path(REPORT_PATH + report_name)
    data_profile = pp.ProfileReport(altered_data_df,
                                    title=report_title,
                                    explorative=True)
    data_profile.to_file(report_full_path)

def main():
    """
    Execute on initialization.

    Creating configuration path via pathlib -> opening config.yaml file ->,
    assigning the opened file to a cfg variable ->,
    creating dataset path via pathlib ->,
    reading the dataset and assigning it to a pandas.DataFrame ->,
    running the file_report function.
    """
    config_full_path = Path(CONFIG_PATH)
    with open(config_full_path, "r", encoding='UTF-8') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    dataset_full_path = Path(DATASET_PATH + cfg["dataset_name"])

    #Assigning custom column headers while reading the csv file
    data_df = pd.read_csv(
        dataset_full_path,
        header=None,
        names=cfg["column_names"]
        )
    print(data_df.head(10))

    file_report(
        altered_data_df=data_df,
        report_title=cfg["report_settings"]["title"],
        report_name=cfg["iris_analysis_report_name"]
        )

if __name__ == "__main__":
    main()
    