import pandas as pd
import numpy as np
import logging as log
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from scipy.stats import skew

def data_cleaning_pipeline(data):
    # Drop columns with missing values
    print("Checking for missing values...")
    print(data.isnull().sum())
    print("\n")

    print('Dropping columns with missing values...')
    data = data.dropna(axis=1)
    print('Columns dropped:', data.columns)
    print("\n")

    # Drop duplicate rows
    print('Dropping duplicate rows...')
    data = data.drop_duplicates()
    print('Number of duplicate rows dropped:', data.duplicated().sum())
    print("\n")

    print("Columns", data.columns)
    return data

def data_splitting_regression(data):
    X = data.drop(columns=['total_water_requirement_m3','water_requirement_mm_day'])
    y = data['total_water_requirement_m3']

    categorical_features = X.select_dtypes(include=[object]).columns
    categorical_pipeline = Pipeline([
        ('encoder', OneHotEncoder())
    ])

    # Combine the preprocessing pipelines
    preprocessor = ColumnTransformer([
        ('categorical_preprocessor', categorical_pipeline, categorical_features)
    ])

    X_preprocessed = preprocessor.fit_transform(X)

    y_skew = skew(y)
    print(f"Skewness: {y_skew:.2f}")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, preprocessor, categorical_features
