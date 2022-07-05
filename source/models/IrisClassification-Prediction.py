import pickle
import numpy as np
import yaml
#Using configuration file for variables
with open("configuration/config.yaml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

#Class names for representing the string version of the encoded ints
class_names = [cfg["class_names"]["class_nr1"],cfg["class_names"]["class_nr2"],cfg["class_names"]["class_nr3"]]
print(class_names)

loaded_model = pickle.load(open(cfg["file_paths"]["model_path"], 'rb'))


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
    
    try:
        for each in edited_input:
            x = float(each)
            if float(each) > 0.09 and float(each) < 11:
                continue
            else:
                raise ValueError()            
        
        if len(edited_input) == 4:
            class_value = loaded_model.predict(np.reshape(edited_input,(1,4)))
            print(class_value,'=',class_names[class_value[0]])
            
        else:
            print("You have not fulfilled the format requirements")
            print("Try Again \n \n \n")
            prediction()
            
    except ValueError:
        print("Please enter a Float that's between 0.1 and 11 \n \n \n ")
        prediction()

prediction()