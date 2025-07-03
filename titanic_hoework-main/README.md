# Titanic Survival Prediction

This project analyzes the famous Titanic dataset to predict passenger survival using machine learning algorithms.

## Overview

The Titanic dataset contains information about passengers aboard the Titanic, including demographic information, ticket details, and survival status. This project uses various machine learning models to predict whether a passenger survived the disaster.

## Features

- **Data Exploration**: Comprehensive analysis of the dataset with visualizations
- **Data Preprocessing**: Handling missing values, feature engineering, and encoding
- **Multiple Models**: Comparison of Random Forest, Logistic Regression, and SVM
- **Visualizations**: Beautiful plots showing data insights and model performance
- **Train/Test Split**: Proper evaluation using held-out test data

## Files

- `titanic_survival_prediction.py`: Main Python script
- `train.csv`: Titanic dataset
- `requirements.txt`: Required Python packages
- `titanic_analysis.png`: Dataset exploration visualizations
- `model_results.png`: Model performance comparison

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python titanic_survival_prediction.py
```

This will:
1. Load and explore the dataset
2. Preprocess the data
3. Create visualizations
4. Train multiple machine learning models
5. Evaluate and compare model performance
6. Save visualization files

## Models Used

1. **Random Forest**: Ensemble method with feature importance analysis
2. **Logistic Regression**: Linear model for binary classification
3. **Support Vector Machine**: SVM with RBF kernel

## Results

The script provides:
- Dataset exploration with survival rates by various factors
- Feature correlation analysis
- Model accuracy comparison
- Confusion matrices
- Feature importance rankings

## Dataset Features

- **PassengerId**: Unique identifier
- **Survived**: Target variable (0 = No, 1 = Yes)
- **Pclass**: Passenger class (1st, 2nd, 3rd)
- **Name**: Passenger name
- **Sex**: Gender
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Passenger fare
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Engineered Features

- **FamilySize**: Total family members aboard
- **IsAlone**: Whether passenger traveled alone
- **Title**: Extracted title from name (Mr, Mrs, Miss, etc.)

## Author

Created for Software Engineering Bootcamp 2025
