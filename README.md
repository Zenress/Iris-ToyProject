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
    - [Running the project](#running-the-project)
      - [Prediction](#prediction)
      - [Training](#training)
      - [Data Analysis](#data-analysis)
        - [Dataset](#dataset)
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

- *You can use it to predict whether what you write is an Iris Versicolor, Iris Setosa or Iris Virginica.*
- *You can use it to do data analysis on the dataset and figure out things about the original dataset.*
- *You can use it to test your knowledge about models and do some testing with the model.*

### Running the project

*The commands below were run in an Anaconda prompt but can be run in a terminal / console just fine.*

#### Prediction

To run the Prediction program use the console command below:

```console
python source/models/IrisClassification-Prediction.py 
```

- This program is meant to predict on the saved training model from the Training Program, it has a specific format you should write your prediction in, a maximum and minimum number as well as a datatype.

#### Training

To run the Training program use the console command below:

```console
python source/models/IrisClassification-Training.py
```

To enable graphing on the training file, you would add --graphs. It looks like this:

```console
python source/models/IrisClassification-Training.py --graphs
```

#### Data Analysis

To run the Data Analysis program use the console command below:

```console
python source/models/IrisClassification-DataAnalysis.py
```

- This program serves to show you information, correlations, maximum and minimum, distributions and so on via a profiling module.

I used some Data Analysis tools to learn more about the dataset as a whole. Pandas helped me understand how many records there was and how it was distributed in the dataset.

Pandas-profling gave me a detailed report about Correlation and Interactions, along with Maximum and Minimum ranges for each feature. (`reports/irisreport.html`)

##### Dataset

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

#### Profile report: Examples

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
