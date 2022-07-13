import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import yaml
#Using configuration file for variables
with open("configuration/config.yaml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)


loaded_model = pickle.load(open("models/" + cfg["model_name"], 'rb'))
mappings_file = np.load("models/" + cfg["encoder_mappings"], allow_pickle=True)

def prediction(): 
    """_summary_
    Conditions checked for:
    Datatype: Float
    Length: 4 floats seperated by ,
    Maximum: [7.9, 4.4, 6.9, 2.5] Depending on iteration
    Minimum: [4.3, 2, 1, 0.1] Depending on iteration
    Along with user instructions to follow

    Try:
     Checks for different input requirements
    Raises:
        ValueError: _description_
    """
    print("Write the data you want to be predicted:")
    print("Input Format: Sepal Length, Sepal Width, Petal Length and Petal Width")
    print("Datatype: Float")
    print("Highest Possible Number: 10")
    print("Lowest Possible Number: 0.1")
    unedited_input = input("Write here: ")
    edited_input = unedited_input.split(',')
    
    i=0 
    try:
        for number in edited_input:
            if float(number) >= cfg["input_min_values"][i] and float(number) <= cfg["input_max_values"][i]:  
                i+=1
                continue
            else:
                raise ValueError()
        
        if len(edited_input) == 4:
            class_value = loaded_model.predict(np.reshape(edited_input,(1,4)))
            print(class_value,'=',mappings_file[class_value[0]])
            
        else:
            print("You have not fulfilled the format requirements")
            print("Try Again \n \n \n")
            prediction()
            
    except ValueError:
        print("Please enter a Float that's between 0.1 and 11 \n \n \n ")
        prediction()

prediction()
#Make sure the highest and lowest number match the iteration
#Iterate through the columns instead of a comma seperated list
#label is not labels as there is only one label
#label name in config, instead of in code
#Features in config instead of in code
#Follow more machine learning naming conventions
#Update readme with new file distribution
#explanation of configuration file?
#Avoid casting.
#Make code shorter using configuration
#Encoder included with model file?
#Change the way the Encoder is saved and loaded
#Review what the zip function does