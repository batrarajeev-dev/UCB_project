# Practical Application III: Comparing Classifiers

## Overview
This project analyzes a bank marketing dataset from the UCI Machine Learning Repository and compares the performance of four classifiers:
- K Nearest Neighbors
- Logistic Regression
- Decision Trees
- Support Vector Machines

The goal is to predict whether a client will subscribe to a term deposit (`y = yes/no`). The notebook uses the full `bank-additional-full.csv` dataset and evaluates classifiers on training and test splits.

## Dataset
- Source: UCI Machine Learning Repository - Bank Marketing dataset
- File used: `data/bank-additional-full.csv`
- Records: 41,188 examples
- Features include demographic, financial, and campaign-related information such as age, job, marital status, education, credit default, housing loan, and contact type.
- Target: `y` indicating subscription to a term deposit.

## What the notebook does
1. Loads and inspects the dataset
2. Examines feature distributions and missing values
3. Preprocesses categorical and numeric data as needed
4. Splits the dataset into training and test sets
5. Trains and evaluates the following models:
   - KNN
   - Logistic Regression
   - Decision Tree
   - SVM
6. Performs hyperparameter tuning for the classifiers
7. Compares model performance using test accuracy and training time
8. Visualizes results with a combined bar/line chart

## Key Findings
- **Best baseline model**: Logistic Regression gave the highest test accuracy among the default models, reaching around `0.9104`.
- **Best tuned model**: After hyperparameter tuning, a Decision Tree with `max_depth = 5` achieved slightly better accuracy at `0.9151` while maintaining low training time.
- **Data concern**: The `default` feature contains a large number of `unknown` values, and the `default` column is heavily skewed toward `no`. This may introduce bias into the model and should be investigated further.

## Recommendations / Next steps
- Investigate how `unknown` values in `default` and `loan` should be handled to reduce bias.
- Explore additional classifiers or ensemble methods to improve test accuracy while keeping training time reasonable.
- Consider feature engineering or resampling approaches for imbalanced label distributions.

## How to run
1. Open `prompt_III.ipynb` in Jupyter Notebook or JupyterLab.
2. Ensure the dataset file exists at `data/bank-additional-full.csv`.
3. Execute cells sequentially from top to bottom.

## Notes
- The notebook also includes dataset documentation references and questions to guide analysis.
- The analysis is designed as a practical comparison of classifier behavior on the same dataset.
