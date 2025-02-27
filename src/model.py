from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np

# Keras Regression Model
def build_regression_model(input_shape):
    model = Sequential([
        Dense(64, input_shape=(input_shape,), activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')  # Regression output
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model

# Train and evaluate Keras model
def train_keras_model(X_train, y_train, X_test, y_test):
    """
    Trains and evaluates a Keras Sequential model for regression.

    Args:
        X_train (np.ndarray): Training data features.
        y_train (np.ndarray): Training data target.
        X_test (np.ndarray): Testing data features.
        y_test (np.ndarray): Testing data target.

    Returns:
        Sequential: Trained Keras Sequential model.
    """
    model = build_regression_model(X_train.shape[1])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    
    model.fit(X_train, y_train, validation_data=(X_test, y_test), 
              epochs=100, batch_size=32, callbacks=[early_stopping, reduce_lr], verbose=1)
    
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, "Keras Model")
    
    return model

# Train and evaluate Random Forest
def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Trains and evaluates a Random Forest Regressor model.

    Args:
        X_train (np.ndarray): Training data features.
        y_train (np.ndarray): Training data target.
        X_test (np.ndarray): Testing data features.
        y_test (np.ndarray): Testing data target.

    Returns:
        RandomForestRegressor: Trained Random Forest Regressor model.
    """
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    evaluate_model(y_test, y_pred, "Random Forest")
    
    return rf_model

# Common function to evaluate models
def evaluate_model(y_test, y_pred, model_name):
    y_test = np.array(y_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- {model_name} Performance ---")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"R^2 Score: {r2}")

    # Print some sample predictions
    print("\nSample Predictions (Actual vs Predicted):")
    for i in range(5):  # Print 5 samples
        print(f"Actual: {y_test[i]}, Predicted: {y_pred[i]}")