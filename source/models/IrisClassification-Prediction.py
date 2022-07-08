import pickle
import numpy as np
import yaml
#Using configuration file for variables
with open("configuration/config.yaml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)


loaded_model = pickle.load(open(cfg["file_paths"]["model_path"], 'rb'))
mappings_file = pickle.load(open(cfg["file_paths"]["encoder_mappings"],"rb"))
list_mappings = list(mappings_file)
print(list_mappings)


def prediction(): 
    """_summary_
    Conditions checked for:
    Datatype: Float
    Length: 4 floats seperated by ,
    Maximum: 10
    Minimum 0.1
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
            print(class_value,'=',list_mappings[class_value[0]])
            
        else:
            print("You have not fulfilled the format requirements")
            print("Try Again \n \n \n")
            prediction()
            
    except ValueError:
        print("Please enter a Float that's between 0.1 and 11 \n \n \n ")
        prediction()

prediction()

#Check what is necessary for someone who knows their way around a terminal
#Read through the readme so it's more relevant and less info dumping when unnecessary
#input needs to disallow non-unicode if i have time