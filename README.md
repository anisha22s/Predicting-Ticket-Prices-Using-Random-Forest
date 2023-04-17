Introduction

This repository contains the code for building a regression tree model to set base prices for performances at a performing arts center. The dataset consists of 10 years of historical ticketing data that includes variables such as paid amount, performance date, seating type, section, row, subscription information, and performance name. The model uses the scikit-learn library in Python to perform cross-validation and hyperparameter tuning using grid search and pruning.

Installation

To run the code in this repository, you will need to have Python 3 installed on your computer.

To use this code, you will need to replace the dataset with your own data. The dataset should be in an Excel format. My. used dataset contains the following variables: paid amount, performance date, seating type, section, row, subscription information, and performance name.

After importing the necessary packages and loading the dataset, the code does the following:

Removes rows where paid amount is less than 5.
Extracts season from performance date.
One-hot encodes the categorical variables.
Splits the data into training and testing sets.
Trains a decision tree model with default hyperparameters.
Performs grid search cross-validation to find the best hyperparameters.
Prunes the decision tree.
Exports the pruned decision tree as a PNG image and displays it in the notebook.
You can modify the hyperparameters in the code to improve the performance of the model.


Acknowledgements

The data used in this analysis was provided by a performing arts center as part of the author's MSBA practicum at UC Davis. The code for building the regression tree model was adapted from the scikit-learn documentation.
