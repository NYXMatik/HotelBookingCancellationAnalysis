# Project MINDD – Hotel Booking Cancellation Analysis

## Team Members
- **Mateusz Janowski**
- **Zuzanna Rybok**
- **Gonçalo Teixeira**

## Overview
Project MINDD focuses on applying advanced machine learning techniques to predict hotel booking cancellations using a real-world dataset. The goal is to identify patterns and key factors influencing customer decisions, thereby enabling better decision-making for hotel management systems.

## Table of Contents
1. [Project Objectives](#project-objectives)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Methods & Techniques](#methods--techniques)
6. [Data Preprocessing](#data-preprocessing)
7. [Model Training & Evaluation](#model-training--evaluation)
8. [Results](#results)
9. [Project Structure](#project-structure)
10. [License](#license)

## Project Objectives
- Predict whether a hotel reservation will be canceled
- Explore and clean customer reservation data
- Perform feature engineering and reduction
- Evaluate multiple classifiers, including ensemble models
- Balance the dataset and handle missing values effectively

## Dataset
- Source: `hotel_booking.csv`
- Rows: ~119,000
- Features: 32
- Target: `is_canceled` (binary classification)

### Key Features:
- Booking info: lead time, arrival dates, distribution channel, etc.
- Customer profile: repeated guest, customer type, country
- Room information: reserved vs assigned type
- Financials: ADR, deposit type, special requests

## Installation

Install required dependencies:

```bash
pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib xgboost lightgbm missingno
```

## Usage

To run the notebook:

```bash
git clone [repository-url]
cd [project-directory]
jupyter notebook Project_MINDD.ipynb
```

## Methods & Techniques

### Preprocessing
- Imputation using `SimpleImputer`
- Encoding categorical data (`LabelEncoder`)
- Handling missing values:
  - Dropped high-missing-value columns (`company`, `agent`)
  - Merged child and baby columns into one
- Feature scaling: `StandardScaler` and `MinMaxScaler`

### Feature Engineering
- Combined features: `children + babies`
- Dropped identifiers to reduce data leakage

### Machine Learning Models
- Logistic Regression
- Random Forest
- K-Nearest Neighbors
- SVM (Support Vector Machine)
- Naive Bayes
- Ensemble Models:
  - Gradient Boosting (GB)
  - AdaBoost
  - Bagging
  - XGBoost
  - LightGBM

### Evaluation Metrics
- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)
- Cross-Validation

## Data Preprocessing

Steps taken:
1. Dropped identifier/irrelevant columns
2. Managed missing values (dropping/imputing)
3. Normalized numeric features
4. Encoded categorical variables
5. Balanced dataset using `RandomUnderSampler` and `NearMiss`

## Model Training & Evaluation

- Used `train_test_split` with stratification
- Hyperparameter tuning using `GridSearchCV`
- Compared models across consistent folds using `KFold`

## Results

- Best performance achieved with `GradientBoostingClassifier` and `XGBoost` (subject to tuning)
- Effective feature reduction improved speed without significant accuracy loss
- Visualizations (confusion matrix, feature importance) are included in the notebook

## Project Structure

```
├── Project_MINDD.ipynb       # Main notebook
├── hotel_booking.csv         # Dataset
├── README.md                 # Project documentation
```

## License

This project is licensed under the MIT License.
