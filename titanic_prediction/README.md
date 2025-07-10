# titanic_homework
day1 homework
# Titanic Survival Prediction

This project demonstrates a basic machine learning workflow to predict passenger survival on the Titanic using Python and scikit-learn.

## Functionality

- **Data Loading:** Loads Titanic passenger data from `data/train.csv`.
- **Preprocessing:** 
  - Removes irrelevant features (`PassengerId`, `Name`, `Ticket`, `Cabin`).
  - Handles missing values by dropping rows with NaNs.
  - Converts categorical variables to numerical using one-hot encoding.
- **Model Training:** 
  - Splits the data into training and test sets.
  - Trains three different classifiers: Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Random Forest.
- **Evaluation:** 
  - Evaluates each model's accuracy and prints classification reports.
  - Visualizes confusion matrices for each model.
  - Compares model accuracies in a bar chart.

## How to Use

1. Place the Titanic dataset (`train.csv`) in the `data/` directory.
2. Run `code_to_start_with.py` to train models and view evaluation results.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- matplotlib
- seaborn

Install dependencies with:
```sh
pip install pandas scikit-learn matplotlib seaborn