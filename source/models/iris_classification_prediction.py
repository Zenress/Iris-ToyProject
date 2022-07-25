"""
Prediction file used for predicting on a DecisionTreeClassified Model.
"""
import pickle
import numpy as np
import yaml

MODEL_PATH = 'models/'
CONFIG_PATH = 'configuration/config.yaml'


def value_check(feature_dict):
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
    edited_input = []
    round_nr = 0
    while len(edited_input) != len(feature_dict):
        for key, value in feature_dict.items():
            print(f"Write the data you want to be predicted {round_nr+1}/4:")
            print(f"Input Feature: {key}")
            print("Datatype: Float")
            print(f"Highest Possible Number: {value['max']}")
            print(f"Lowest Possible Number: {value['min']}")
            user_input = input("Write here: ")
            print("____________________________________________")

            try:
                if float(user_input) >= value['min'] and float(user_input) <= value['max']:
                    edited_input.append(user_input)
                else:
                    raise ValueError()

            except ValueError:
                print(f"Please enter a Float that's between {value['min']} and {value['max']}")
                break

    return edited_input

def prediction(user_input, model_encoder_dictionary):
    """
    Predict on Decistion Tree Model.

    Using the predict function of SKlearn DecisionTreeClassifier,
    accompanied by the corresponding encoder mapping to produce a readable prediction.

    Args:
        user_input (list): user input edited into a 4 item list.
        model_encoder_dictionary (dict): dictionary with the model and encoder objects.

    """
    class_value = model_encoder_dictionary["model"].predict(np.reshape(user_input,(1,4)))
    print(class_value, '=', model_encoder_dictionary['encoder_mappings'][class_value[0]])

def main():
    """
    Execute at runtime.

    initializing configuration file ->,
    loading the pickled dictionary model_encoder_dictionary ->,
    Running the value_check to make sure the prediction conditions are met ->,
    Predicting using the value_check result and the encoder mappings.

    """
    with open(CONFIG_PATH, "r", encoding='UTF-8') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    model_encoder_dictionary = pickle.load(open(MODEL_PATH + cfg["model_and_encoder_name"], 'rb'))

    user_input = value_check(cfg["features"])

    prediction(user_input, model_encoder_dictionary)

if __name__ == "__main__":
    main()
