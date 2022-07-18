import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import yaml
#Using configuration file for variables
with open("configuration/config.yaml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

loaded_dictionary = pickle.load(open("models/" + cfg["model_and_encoder_name"], 'rb'))
mappings_file = loaded_dictionary["encoder_mappings"]

def prediction(): 
    """_summary_
    Conditions checked for:
    Datatype: Float
    Length: 1 number for each iteration
    Maximum: [7.9, 4.4, 6.9, 2.5] Depending on iteration
    Minimum: [4.3, 2, 1, 0.1] Depending on iteration
    Along with user instructions to follow

    Try:
     Checks for different input requirements
    Raises:
        ValueError: Raises a valueerror when you type wrong datatype or too low / high number
    """
    edited_input = []
    while len(edited_input) != len(cfg["features"]):
        for feature_nr in range(len(cfg["features"])):
            print(f"Write the data you want to be predicted {feature_nr+1}/4:")
            print(f"Input Feature: {cfg['features'][feature_nr]}")
            print("Datatype: Float")
            print(f"Highest Possible Number: {cfg['input_max_values'][feature_nr]}")
            print(f"Lowest Possible Number: {cfg['input_min_values'][feature_nr]}")
            user_input = input("Write here: ")
            print("____________________________________________")
        
            try:            
                if float(user_input) >= cfg['input_min_values'][feature_nr] and float(user_input) <= cfg['input_max_values'][feature_nr]:  
                    edited_input.append(user_input)
                else:
                    raise ValueError()
                
            except ValueError:
                print("___________________________________________________")
                print(f"Please enter a Float that's between {cfg['input_min_values'][feature_nr]} and {cfg['input_max_values'][feature_nr]}\n \n \n")
                break
            
    class_value = loaded_dictionary["model"].predict(np.reshape(edited_input,(1,4)))
    print(class_value,'=',mappings_file[class_value[0]])

prediction()
#Follow more machine learning naming conventions
#explanation of configuration file?
#Avoid casting.
#Make code shorter using configuration
#Review what the zip function does
