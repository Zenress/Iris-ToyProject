import pickle
import numpy as np
import yaml

MODEL_PATH = 'models/'
CONFIG_PATH = 'configuration/config.yaml'

#Using configuration file for variables
with open(CONFIG_PATH, "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

model_encoder_dictionary = pickle.load(open(MODEL_PATH + cfg["model_and_encoder_name"], 'rb'))
mappings_file = model_encoder_dictionary["encoder_mappings"]

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
    round_nr = 0
    while len(edited_input) != len(cfg["features"]):
        for key, value in cfg["features"].items():
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
                print(f"Please enter a Float that's between {value['min']} and {value['max']}\n \n \n")
                break
            
    class_value = model_encoder_dictionary["model"].predict(np.reshape(edited_input,(1,4)))
    print(class_value,'=',mappings_file[class_value[0]])

prediction()
#Divide Data analysis into a function
#Argparse instead of Getopt
#Better function layout