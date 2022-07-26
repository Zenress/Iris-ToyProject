"""
Training file used for training the DecisionTreeClassifier.
"""
from pathlib import Path
import pickle
import argparse
import matplotlib.pyplot as plt
import yaml
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

MODEL_PATH = 'models/'
DATASET_PATH = 'source/data/'
CONFIG_PATH = 'configuration/config.yaml'


def arguments_handler():
    """
    Handle commandline arguments

    Handles arguments that are passed through during the run command.
    Available arguments are: --graphs

    Returns:
        Namespace: Arguments that were passed through command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graphs", #Does not allow explicit paremeter name=
        help="Enables Graphing",
        action="store_true"
        )
    args = parser.parse_args()
    return args


def read_dataset_and_encode(
    label_name: str,
    dataset_name: str,
    column_names: list,
    features: dict
    ):
    """
    Read data from dataset and encode the label column

    Reads the data from the dataset and assigns the header with column_names.
    Then it encodes the categorical label column into a numerical label column.

    Args:
        label_name (str): name of the label column
        dataset_name (str): name of the dataset to read from
        column_names (list): names of all the columns
        features (dict): a dictionary with all the features and value ranges

    Returns:
        pandas.DataFrane: holds all the feature column records for each feature column
        pandas.Series: holds the label column records
        sklearn.preprocessing.LabelEncoder: Encoder used for encoding the label column
    """
    label_encoder = preprocessing.LabelEncoder()
    dataset_full_path = Path(DATASET_PATH+dataset_name)

    #Assigning custom column headers while reading the csv file
    dataset_df = pd.read_csv(dataset_full_path,
                             header=None,
                             names=column_names)

    #Encoding the categorical column header to an int datatype
    dataset_df[label_name] = label_encoder.fit_transform(dataset_df[label_name])

    X = dataset_df[list(features)]
    y = dataset_df[label_name]

    return X, y, label_encoder

def train_model(
    dtc_model: DecisionTreeClassifier,
    train_index: int,
    test_index: int,
    X: pd.DataFrame,
    y: pd.Series
    ):
    """
    Train DecisionTreeClassifier Model using Iris Dataset.

    Trains a DecisionTreeClassifier using a KFolded dataset.
    Splits dataset between feature columns and labels,
    and then further splits them into a training and testing set.

    Args:
        dtc_model (sklearn.tree.DecisionTreeClassifier): An untrained DecisionTreeClassifier,
            used for classifying the KFolded Iris Dataset
        train_index (int) indices for the training side of the kfold split
        test_index (int) indices for the testing side of the kfold split
        X (pandas.DataFrame) The Feature columns of the dataset
        y (pandas.Series) The Label column of the dataset
    """
    i = 1
    X_train = X.iloc[train_index].values
    X_test = X.iloc[test_index].values
    y_train = y.iloc[train_index].values
    y_test = y.iloc[test_index].values

    dtc_model = dtc_model.fit(X_train,y_train)
    print((f"Accuracy for fold nr. {i} on test set:"
            f" {metrics.accuracy_score(y_test, dtc_model.predict(X_test))} - "
            f"Double check: {dtc_model.score(X_test,y_test)}"))

    i += 1


def graphing(
    train_index: int,
    test_index: int,
    occurance_df: pd.DataFrame,
    round_nr: int,
    y: pd.Series
    ) -> pd.DataFrame:
    """
    Plot dataset and training progress to graph.

    Uses Matplotlib to plot a graph with how the dataset was distributed
    in the different KFold splits.
    Afterwards it tells you how it distributed the label records throughout the splits.

    Args:
        train_index (int): indices for the training side of the kfold split
        test_index (int): indices for the testing side of the kfold split
        occurance_df (pd.DataFrame): For counting label occurances of each label
        round_nr (int): nr that represents which round the occurances was recorded in
        y (pd.Series): Series that holds the label for each dataset record.

    Returns:
        pd.DataFrame: Occurances of each of the labels
    """
    o_train = y.iloc[train_index].value_counts()
    o_train.name = f"train {round_nr}"
    o_test = y.iloc[test_index].value_counts()
    o_test.name = f"test {round_nr}"

    #Concatenate 2 pandas objects along a single dataframe axis
    df = pd.concat([o_train, o_test], axis=1, sort=False)
    df["|"] = "|"
    occurance_df.append(df)

    plt.scatter(x=y.iloc[train_index].index,y=y.iloc[train_index],label="train")
    plt.scatter(x=y.iloc[test_index].index,y=y.iloc[test_index],label="test")
    plt.legend()
    plt.show()

    return occurance_df


def save_file(
    model: DecisionTreeClassifier,
    encoder: preprocessing.LabelEncoder,
    model_and_encoder_name: str
    ):
    """
    Save model and encoder mappings.

    Saving the trained model and encoder mappings that was used in the model.
    A dictionary is used to save both objects and pickle them into a single file.

    Args:
        model (sklearn.tree.DecisionTreeClassifier): This is the trained model
        encoder (sklearn.preprocessing.LabelEncoder): We take the encoder mappings from this
        model_and_encoder_name (str): The name of the model and encoder file,
            that is gonna be pickled.
    """
    dtc_model_and_encoder_mapping = {
        "model": model,
        "encoder_mappings": encoder.classes_,
    }

    model_and_encoder_full_path = Path(MODEL_PATH+model_and_encoder_name)
    pickle.dump(dtc_model_and_encoder_mapping, open(model_and_encoder_full_path, 'wb'))


def main():
    """
    Execute at code runtime.

    Initiating configuration file ->,
    Labelencoder is initialized -> Dataset is read ->,
    Categorical Label Column is encoded to numerical type ->,
    X is assigned with Features and Y is assigned with Label ->,
    Model and KFold cross validation is initialized and run ->,
    arguments_handler is run to check for cmd line arguments ->,
    the model is trained with train_model ->,
    graphing is run if correct arguments are present ->,
    the finished training and encoding is saved to a file as dictionaries.

    """
    args = arguments_handler()

    config_full_path = Path(CONFIG_PATH)
    with open(config_full_path, "r", encoding='UTF-8') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)


    X, y, label_encoder = read_dataset_and_encode(
        label_name=cfg["label_name"],
        dataset_name=cfg["dataset_name"],
        column_names=cfg["column_names"],
        features=cfg["features"]
        )

    dtc_model = DecisionTreeClassifier(criterion=cfg["decisiontree_settings"]["criterion"])

    indices_kfold = StratifiedKFold(
        n_splits=cfg["kfold_settings"]["nr_splits"],
        shuffle=cfg["kfold_settings"]["shuffle"],
        random_state=cfg["kfold_settings"]["random_state"]
        )
        # Randomstate for uniform results

    round_nr = 1
    for train_index, test_index in indices_kfold.split(X,y):
        train_model(
            dtc_model=dtc_model,
            train_index=train_index,
            test_index=test_index,
            X=X,
            y=y
            )

        if args.graphs is True:
            occurance_df = []
            occurances_df = graphing(
                train_index=train_index,
                test_index=test_index,
                occurance_df=occurance_df,
                round_nr=round_nr,
                y=y
                )
            round_nr+=1
            print(pd.concat(occurances_df,axis=1, sort= False))

    save_file(
        model=dtc_model,
        encoder=label_encoder,
        model_and_encoder_name=cfg["model_and_encoder_name"]
        )

if __name__ == "__main__":
    main()
    