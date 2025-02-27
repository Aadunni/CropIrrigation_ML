from model import train_keras_model,train_random_forest

from data_prep import data_cleaning_pipeline,data_splitting_regression

from load_data import fetch_irrigation_data
import seaborn as sns
import os
import pickle

def main():
    print("\033[1;34m=======================================\033[0m")
    print("\033[1;32mStarting Crop Model Training and Evaluation...\033[0m")
    print("\033[1;34m=======================================\033[0m")

    # Load the data
    print("\n\033[1;33mLoading irrigation data...\033[0m")
    data = fetch_irrigation_data()
    print("\033[1;32mData loaded successfully.\033[0m")

    # Data cleaning pipeline
    print("\n\033[1;33mCleaning data...\033[0m")
    data = data_cleaning_pipeline(data)
    print("\033[1;32mData cleaning complete.\033[0m")

    # Split the data for regression
    print("\n\033[1;33mSplitting data for regression...\033[0m")
    X_train_regression, X_test_regression, y_train_regression, y_test_regression, preprocessor, categorical_features = data_splitting_regression(data)
    print("\033[1;32mData splitting complete.\033[0m")

    # Ensure artifacts directory exists
    if not os.path.exists("artifacts"):
        os.makedirs("artifacts")
        print("\n\033[1;33mCreated 'artifacts' directory.\033[0m")

    # Save preprocessor
    with open('artifacts/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    print("\033[1;32mPreprocessor saved to artifacts/preprocessor.pkl\033[0m")

    # Get categorical feature names
    # categorical_features = X_train_regression.select_dtypes(include=[object]).columns.tolist()

    # Get feature names after one-hot encoding
    feature_columns = preprocessor.named_transformers_['categorical_preprocessor']['encoder'].get_feature_names_out(categorical_features)

    # Save feature columns
    with open('artifacts/feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns.tolist(), f)
    print("\033[1;32mFeature columns saved to artifacts/feature_columns.pkl\033[0m")

    # Train and evaluate the regression models and save model artifacts
    print("\n\033[1;33mTraining Keras Regression Model...\033[0m")
    keras_model = train_keras_model(X_train_regression, y_train_regression, X_test_regression, y_test_regression)
    keras_model.save("artifacts/keras_model.h5")
    print("\033[1;32mKeras model training complete and saved to artifacts/keras_model.h5\033[0m")

    print("\n\033[1;33mTraining Random Forest Regression Model...\033[0m")
    rf_model = train_random_forest(X_train_regression, y_train_regression, X_test_regression, y_test_regression)
    with open("artifacts/rf_model.pkl", "wb") as f:
         pickle.dump(rf_model, f)
    print("\033[1;32mRandom Forest model training complete and saved to artifacts/rf_model.pkl\033[0m")

    print("\n\033[1;34m=======================================\033[0m")
    print("\033[1;32mCrop Model Training and Evaluation Completed.\033[0m")
    print("\033[1;34m=======================================\033[0m")


if __name__ == "__main__":
    main()
