import pickle
import numpy as np
import yaml

with open("configuration/config.yaml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

#Creating the class names in a seperate list so it can be used to tell the user which Iris it predicted
class_names = [cfg["class_names"]["class_nr1"],cfg["class_names"]["class_nr2"],cfg["class_names"]["class_nr3"]]
print(class_names)

#The name of the modelfile, if you change the filename in the training file you'll have to update it here as well
#The model is loaded using pickle and the filename that you have given

loaded_model = pickle.load(open(cfg["file_paths"]["model_path"], 'rb'))


def prediction(): 
    """_summary_

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
            if float(each) > 0.1 and float(each) < 11:
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

#make the readme file better by adding setup category, run category on how to run it, table of contents, try to style documentation after importance, lists, bold, italic characters and headers and subheaders
#Coding conventions on Comments and