# Titanic-Survival-Prediction

This repository contains code for predicting the survival of passengers on the Titanic using various machine learning classifiers. The dataset used is the Titanic dataset from Kaggle.

## Table of Contents
- [Introduction](#introduction)
- [Libraries Used](#libraries-used)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Using Iris Dataset](#using-iris-dataset)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction
The goal of this project is to predict whether a passenger survived the Titanic disaster based on various features such as age, sex, and passenger class. We use several machine learning classifiers to build the predictive models and evaluate their performance.

## Libraries Used
- **pandas**: For data manipulation and analysis.
- **seaborn**: For data visualization.
- **matplotlib**: For plotting graphs and visualizations.
- **numpy**: For numerical computations.
- **sklearn**: For machine learning algorithms and model evaluation.
- **xgboost**: For the XGBoost classifier, an efficient implementation of gradient boosting.

## Data Preprocessing
1. **Loading Data**: The Titanic dataset is loaded into a pandas DataFrame.
2. **Handling Missing Values**: Missing values in the `Age` column are filled based on the passenger class. The `Cabin` column is dropped due to many missing values.
3. **Encoding Categorical Variables**: Categorical variables such as `Sex` and `Embarked` are converted into numerical format using one-hot encoding.
4. **Feature Selection**: Irrelevant features such as `Name`, `Ticket`, and `PassengerId` are dropped.

## Model Training
We use the following classifiers to train our models:
- **Support Vector Machine (SVM)**: A powerful classifier for both linear and non-linear data.
- **Decision Tree**: A simple and interpretable model that splits data based on feature values.
- **Random Forest**: An ensemble method that uses multiple decision trees to improve accuracy.
- **XGBoost**: A highly efficient and scalable implementation of gradient boosting.

Hyperparameter tuning is performed using `GridSearchCV` to find the best parameters for each model.

## Model Evaluation
Each model is evaluated using accuracy, precision, recall, and confusion matrix. Visualizations are provided to compare the performance of different models.

## Using Iris Dataset
In addition to the Titanic dataset, the code also demonstrates the use of the Iris dataset to train and evaluate the models.

## Installation
To install the required libraries, run:
```bash
pip install pandas seaborn matplotlib numpy scikit-learn xgboost
