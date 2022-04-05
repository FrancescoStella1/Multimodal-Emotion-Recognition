# Affective Computing & Natural Interaction project
The goal of this project is to classify emotions based on Valence and Arousal estimates of subjects involved in watching short video clips in a lab setting. More precisely, for each subject **Electroencephalogram (EEG)** is collected and **Facial Features (FF)** are computed starting from video recordings.
Different classifiers are devised and trained on multimodal data, decision-level fusion is then performed in order to retrieve the final predictions. 
**Bayesian Hyperparameter Optimization** is performed on each model in order to try to find the best hyperparameter configuration leading to high accuracy.\
If a **GPU** with **CUDA** architecture is available, then it is automatically used whenever is possible.

## Dataset
For this classification task the DEAP dataset has been used: [DEAP dataset Homepage](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/)

## Requirements
This software has been tested on Windows and Linux with Python 3.8.\
In order to be able to execute the software and reduce the probability of conflicts/issues, please follow these steps:

- make sure *anaconda* is installed on the system
- change the **prefix** field in ```environment.yml``` to a desired path and change the **name** field as desired
- create a new environment with the following command: *"conda env create -f environment.yml"*


## Usage
Before starting to train the models, please download EEG .dat files and video recordings from the DEAP dataset and place them into two directories. Then, in ```main.py``` modify **EEG_DATA_PATH** and **VIDEO_DATA_PATH** global variables accordingly.\
Modify also the other variables related to the types of the models and preprocessing associated to EEG, Facial Features and decision-fusion (ensemble).

### 1. Preprocess EEG data and extract features
- Modify the global variable **EEG_SAVE_PATH** in ```main.py``` in order to choose a directory in which to store the preprocessed EEG data (32 files).

- Use the following command: *'python main.py --create_eeg_files True'*

### 2. Preprocess video clips and extract Facial Features
- Modify the global variables **LANDMARKS_DATA_PATH** and **FF_SAVE_PATH** in ```main.py``` in order to choose two directories in which to store respectively the extracted landmarks (~ hundres of files) and the facial features (32 files).
 
- Use the following commands:
    1. *'python main.py --create_landmark_files True'*
    2. *'python main.py --create_ff_files True'*


### 3. Train Models
In order to train both models and perform decision-level fusion, use: *'python main.py --train_ensemble True'*.\
\
Replace **--train_ensemble** with **--train_eeg** or **--train_ff** in order to independently train EEG or FF models.\
For testing the models use **--test_ensemble**, **--test_eeg** or **--test_ff**.