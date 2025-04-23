# Project MINDD – Hotel Booking Cancellation Analysis

## Team Members
- **Mateusz Janowski**
- **Zuzanna Rybok**
- **Gonçalo Teixeira**

## Overview
This project leverages advanced machine learning techniques to predict whether a hotel booking will be canceled. The main notebook `Hotel_Booking_Cancelation_Analysis.ipynb` walks through the full pipeline—from data cleaning and exploratory analysis to modeling and evaluation.

We focus on real-world hotel booking data, identifying trends and building interpretable models that can support operational decision-making.

## Table of Contents
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Environment Setup](#environment-setup)
- [Running the Project](#running-the-project)
- [Methodology](#methodology)
  - [Preprocessing](#preprocessing)
  - [Feature Engineering](#feature-engineering)
  - [Models](#models)
  - [Evaluation Metrics](#evaluation-metrics)
- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [License](#license)

## Objectives
- Predict booking cancellations based on reservation metadata.
- Handle missing and noisy data realistically.
- Engineer domain-relevant features to improve predictions.
- Compare a wide range of classifiers including boosting and ensemble methods.

## Dataset
- File: `hotel_booking.csv`
- Observations: ~119,000
- Columns: 32 attributes per booking
- Target: `is_canceled` (binary classification)

### Sample Features:
- **Customer behavior**: lead time, special requests, repeated guest
- **Booking logistics**: arrival date, reserved vs assigned room
- **Channel & financials**: distribution type, deposit, ADR

## Environment Setup

```bash
pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib xgboost lightgbm missingno
```

## Running the Project

```bash
git clone https://github.com/[your-username]/[repo-name].git
cd [repo-name]
jupyter notebook Hotel_Booking_Cancelation_Analysis.ipynb
```

## Methodology

### Preprocessing
- Dropped columns: `company`, `agent` (too many missing values)
- Combined columns: `babies` + `children`
- Encoded categorical features using `LabelEncoder`
- Imputed missing values and scaled numeric data (`StandardScaler`, `MinMaxScaler`)

### Feature Engineering
- Removed identifiers like name/email
- Created meaningful aggregates (e.g., total_children)

### Models
- **Linear**: Logistic Regression
- **Tree-based**: Decision Trees, Random Forest
- **Boosted**: XGBoost, LightGBM, AdaBoost, Gradient Boosting
- **Others**: KNN, Naive Bayes, SVM, Bagging

### Evaluation Metrics
- Accuracy
- Confusion Matrix
- Classification Report (precision, recall, F1-score)
- K-Fold Cross-Validation
- GridSearchCV for tuning

## Key Results
- Best performance: Gradient Boosting and XGBoost (after tuning)
- Reduced features improved training time while maintaining accuracy
- Confusion matrix and feature importances plotted and analyzed

## Repository Structure

```
├── Hotel_Booking_Cancelation_Analysis.ipynb   # Full EDA and modeling
├── hotel_booking.csv                          # Dataset
├── README.md                                  # Documentation
├── LICENSE.txt                                # MIT License
```

## License

This project is licensed under the MIT License.
