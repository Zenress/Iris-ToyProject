"""
Prediction file used for predicting on a DecisionTreeClassified Model.
"""
from pathlib import Path
import pickle
import numpy as np
import yaml

MODEL_PATH = "models/"
CONFIG_PATH = "configuration/config.yaml"


def value_check(feature_dict: dict) -> list:
    """
    Check conditions.

    Conditions checked for:
    Datatype: Float.
    Length: 1 number for each iteration.
    Maximum: [7.9, 4.4, 6.9, 2.5] Depending on iteration.
    Minimum: [4.3, 2, 1, 0.1] Depending on iteration.
    Along with user instructions to follow.

    Try:
        Checks for different input requirements.
    Raises:
        ValueError: Raises a valueerror when you type wrong datatype or too low / high number.

    Args:
        feature_dict (dict) dictionary of the features,
            derived from the cfg dictionary object.
    """
    features_input = []

    for count, (key, value) in enumerate(feature_dict.items(), 1):
        while len(features_input) != count:
            print(f"Write the data you want to be predicted {count}/{len(feature_dict)}:")
            print(f"Input Feature: {key}")
            print("Datatype: Float")
            print(f"Highest Possible Number: {value['max']}")
            print(f"Lowest Possible Number: {value['min']}")
            user_input = input("Write here: ")
            print("____________________________________________")

            if value['min'] <= float(user_input) <= value['max']:
                features_input.append(user_input)
            else:
                print(
                    f"Please enter a Float that's between {value['min']} and {value['max']}"
                )
                print()
                print()
    return features_input


def prediction(features_input: list, model_encoder_dictionary: dict) -> None:
    """
    Predict on Decistion Tree Model.

    Using the predict function of SKlearn DecisionTreeClassifier,
    accompanied by the corresponding encoder mapping to produce a readable prediction.

    Args:
        features (list): user input edited into a 4 item list.
        model_encoder_dictionary (dict): dictionary with the model and encoder objects.
    """
    class_value = model_encoder_dictionary["model"].predict(
        np.reshape(features_input, (1, len(features_input)))
    )[0]
    print(
        class_value, "=", model_encoder_dictionary["encoder_mappings"][class_value]
    )


def main() -> None:
    """
    Execute at runtime.

    creating the config path using pathlib ->,
    initializing configuration file ->,
    creating the model and encoder path using pathlib ->,
    unpickling the dictionary model_encoder_dictionary ->,
    Running the value_check function to make sure the prediction conditions are met ->,
    Predicting using the value_check result and the encoder mappings.
    """
    with open(CONFIG_PATH, "r", encoding="UTF-8") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    model_and_encoder_full_path = Path(MODEL_PATH, cfg["model_and_encoder_name"])
    model_encoder_dictionary = pickle.load(open(model_and_encoder_full_path, "rb"))

    features_input = value_check(feature_dict=cfg["features"])

    prediction(features_input=features_input, model_encoder_dictionary=model_encoder_dictionary)


if __name__ == "__main__":
    main()
