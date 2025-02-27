# 🌱 Crop Irrigation Data Engineering Project

## Overview
This project implements a machine learning pipeline for irrigation data analysis and water requirement prediction. The pipeline includes stages for data loading, data cleaning, data splitting, and model training using both a Keras regression model and a Random Forest model. The goal is to predict the total water requirement and optimize irrigation strategies based on soil type, altitude, water requirement per day, and other factors.

## Features
- **Data Loading:** Loads irrigation data from CSV files (and databases if needed).
- **Data Cleaning:** Cleans and preprocesses raw data to handle missing values and inconsistencies.
- **Data Splitting:** Divides the dataset for regression tasks.
- **Model Training:**
  - **Keras Regression Model:** A deep learning model built using Keras.
  - **Random Forest Regression:** A traditional machine learning model implemented for comparison.
- **Visualization:** Uses Seaborn for exploratory data analysis and visualization.

## Project Structure
- **docker-compose.yaml:** Docker setup for containerized deployment.
- **irrigation_strategy_with_soil_type.csv:** Primary data source containing irrigation strategies and related parameters.
- **requirements.txt:** List of Python dependencies.
- **data/**: Directory to store raw and processed data files.
- **db/**:
  - **csv_import.sh:** Script to import CSV data into a database.
  - **init.sql:** SQL scripts for database initialization.
- **init.sql/**: Additional SQL initialization scripts.
- **models/**:
  - **regression_model.py:** Contains the implementations of machine learning models.
- **src/**:
  - **build.py:** Main script that orchestrates the data pipeline and model training.
  - **data_prep.py:** Contains functions for data cleaning and splitting.
  - **load_data.py:** Module to fetch and load irrigation data.
  - **model.py:** Contains definitions and training functions for both the Keras and Random Forest models.

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Change to the project directory:
   ```
   cd /Users/admin/Documents/crop_ml
   ```
3. Create a virtual environment:
   ```
   python3 -m venv venv
   ```
4. Activate the virtual environment:
   ```
   source venv/bin/activate
   ```
5. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
To run the complete pipeline, execute the following command:
   ```
   sh run_app.sh
   ```
This script will:
- Load the irrigation data.
- Clean and process the data.
- Split the data for regression training.
- Train and evaluate both the Keras regression model and the Random Forest model.
- Print performance metrics such as MSE, RMSE, MAE, MAPE, and R² score.

## Model Performance
After running the pipeline, the output includes:
- **Keras Model Performance:**  
  Metrics include Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and R² Score.
- **Random Forest Performance:**  
  Currently reports a perfect score (R² = 1.0) which may indicate overfitting or an evaluation anomaly. Further investigation is recommended.

## Future Improvements
- **Investigate Random Forest Evaluation:** Check for potential overfitting or data leakage.
- **Enhanced EDA & Feature Engineering:** Deepen the analysis of feature interactions and transformations.
- **Model Optimization:** Experiment with alternative architectures, hyperparameters, and regularization techniques.
- **Cross-Validation:** Implement cross-validation for more robust performance evaluation.
- **Diagnostic Analysis:** Perform detailed error analysis to identify areas for improvement.

## License
This project is released under the MIT License.

## Acknowledgements
Appreciation goes to all contributors who helped in data preparation, model development, and overall project improvements.
