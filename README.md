# **Iris ToyProject**

This project is a **Classification model** using the **Iris Dataset**. The model used is a **DecisionTree** and it is made primarily using **SKlearn** and **Pandas**

This project is meant to serve as a sort of ability guaging to see what i can do with my own skills.
The project also helped me with building a more **team oriented** mindset and understanding of how i should **structure repositories**, **the code** and this **README file**. So that other people might be able to understand it easier

I've learned a lot about **Data analysis**, **Github repositories**, **library/dependency** **management** and **User Experience**

## Table of Contents

- [**Iris ToyProject**](#iris-toyproject)
  - [Table of Contents](#table-of-contents)
  - [The libraries used are](#the-libraries-used-are)
  - [Setup](#setup)
    - [Configuration](#configuration)
    - [Installation](#installation)
    - [Removing the Conda Environment](#removing-the-conda-environment)
  - [Project usage](#project-usage)
    - [Using the project](#using-the-project)
    - [Program information](#program-information)
  - [Running the project](#running-the-project)
  - [Dataset](#dataset)
    - [Data Analysis](#data-analysis)
    - [Profile report: Examples](#profile-report-examples)
  - [Credits](#credits)
  - [Sources](#sources)

## The libraries used are

- *Pandas*
- *Pandas Profiling*
- *SKlearn (StratifiedKFold, Preprocessing, Metrics and DecisionTreeClassifier)*
- *Matplotlib Pyplot*
- *Pickle*

## Setup

### Configuration

There is an included config.yaml file under the configuration folder. This file is where you would change different variables easily. Example:

```yaml
kfold_settings:
 nr_splits: 5
 shuffle: True
 random_state: 123
```

### Installation

To run this project you should create a **Conda Environment** (<https://www.anaconda.com/products/distribution>) to run it on, this will help with making sure it can run in it's default configuration. This is easily done with the included **Conda Environment** file.
To do so, you should type the command:

```console
conda env create -f configuration/irisproject_conda_env.yaml
```

The default name of the environment file is: `irisproject_conda_env.yaml`.

When you run the command above it creates a **conda environment** which can be selected as the interpretor when you run one of the python files. It cointains all the libraries you would need to run the project.

After having created the environment you also have to activate it:

```console
conda activate Iris_Project_Conda_Environment
```

### Removing the Conda Environment

To delete the conda environment you will have to locate where the Anaconda3 folder is, by default it is under: `C:\Users\YourUserHere\Anaconda3\envs`

You then delete the folder that matches the name of the Environment files configured name, by default it's name would be: `Iris_Project_Conda_Environment`

## Project usage

This is a guide on how to use the project for it's intended purposes

### Using the project

- *You can use it to predict whether what you write is an Iris Versicolor, Iris Setosa or Iris Virginica.*
- *You can use it to do data analysis on the dataset and figure out things about the original dataset.*
- *You can use it to test your knowledge about models and do some testing with the model.*

To use the project you need to choose the file best suited to what you want to do with the project.

- *If you want the model to predict an Iris you should open the prediction file.*
- *If you want to retrain the model you should use the Training file.*
- *If you want to analyze the dataset then you should use the DataAnalysis file.*

### Program information

- The project is split into a group of 4 distinctive programs all meant to serve a different purpose in the overall project.
- There is the DataAnalysis Program which serves to show you information, correlations, maximum and minimum, distributions and so on via a profiling module.
There is also plenty to understand from reading the Console/Terminal printout that the file gives you
- There is the Prediction Program which is meant to predict on the saved training model from the Training Program, it has a specific format you should write your prediction in,
a maximum and minimum number as well as a datatype, you use it by following the instructions given in the Console/Terminal
- Then there is the Training Program, there are 2. One of them is completely void of graphs and is mostly supposed to be used when you just want to retrain the model, edit a bit of the parameters
and things like that. The Training Program that has +graphs in the name is the file that prints out graphs and information about how the KFolded Dataset functions with the DecisionTreeClassifier

## Running the project

This section is intended to give you specific commands to run that you can easily copy paste.

The commands below were run in an Anaconda prompt but it can be run in a terminal or console just fine as well.

For the Prediction file you can type these commands:

```console
python source/models/IrisClassification-Prediction.py 
```

For the Training file without graphs you can type this command:

```console
python source/models/IrisClassification-Training.py
```

For the Training file that has graphs you can type this command:

```console
python source/models/IrisClassification-Training +graphs.py
```

And for the last file named DataAnalytsis you can type this command:

```console
python source/models/IrisClassification-DataAnalysis.py
```

## Dataset

The dataset used is the Iris Dataset (<https://archive.ics.uci.edu/ml/datasets/Iris>)

- 5 columns, headers added later on
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
  - Class
    - Iris Setosa
    - Iris Versicolor
    - Iris Virginica
- 150 records

The names we're derived from the documentation under **Attribute information** gathered here: <https://archive.ics.uci.edu/ml/datasets/Iris>

### Data Analysis

I used some Data Analysis tools to learn more about the dataset as a whole. Pandas helped me understand how many records there was and how it was distributed in the dataset.

Pandas-profling gave me a detailed report about Correlation and Interactions, along with Maximum and Minimum ranges for each feature. (`reports/irisreport.html`)

### Profile report: Examples

- Correlation Matrix:

![Correlation Matrix](docs/READMEIrisDataReportCorrelation.png)

- Interactions between Sepal Length and Petal Width:

![Interactions between Sepal Length and Petal Width](docs/READMEIrisDataReportInteractionSepalLengthVSPetalWidth.png)

- Interactions between Sepal Length and Sepal Width

![Interactions between Sepal Length and Sepal Width](docs/READMEIrisDataReportInteractionSepalLengthSepalWidth.png)

## Credits

- Credit to **UCI** for making the dataset widely accessible.
- Credit to **Michele Stawowy** for **Quality Assurance and Guidance**
- Credit to **Martin Riishøj Mathiasen** for the idea to **KFold the Dataset**

## Sources

- Iris Dataset can be found here: <https://archive.ics.uci.edu/ml/datasets/Iris>
- Reference for Folder Structure Inspiration: <https://i0.wp.com/neptune.ai/wp-content/uploads/DL-project-directory.png?resize=938%2C1024&ssl=1>
- DecisionTreeClassifier Inspiration: <https://www.datacamp.com/tutorial/decision-tree-classification-python>
- Anaconda: <https://www.anaconda.com/>

---
