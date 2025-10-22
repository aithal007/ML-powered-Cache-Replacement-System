import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
import sys

# TensorFlow/Keras for LSTM
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

print("Starting model training for LightGBM, Random Forest, and LSTM...")

# --- 1. Load Data ---
try:
    df = pd.read_csv('features.csv')
except FileNotFoundError:
    print("Error: features.csv not found. Please run 1_feature_engineering.py first.")
    sys.exit(1)

print(f"Loaded {len(df)} rows from features.csv")

# --- 2. Prepare Data for Training ---
# Use all available features
available_features = [col for col in df.columns if col != 'reuse_distance']
print(f"Using features: {available_features}")

X = df[available_features]
y = df['reuse_distance']

# Split into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# --- 3. Train the Enhanced LightGBM Model ---
print("\n=== Training LightGBM Model ===")
lgbm_model = lgb.LGBMRegressor(
    objective='regression',
    n_estimators=500,  # More trees
    learning_rate=0.05,  # Lower learning rate
    max_depth=8,  # Deeper trees
    num_leaves=64,  # More leaves
    min_child_samples=10,  # Prevent overfitting
    subsample=0.8,  # Feature bagging
    colsample_bytree=0.8,
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=0.1,  # L2 regularization
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgbm_model.fit(X_train, y_train)
print("LightGBM training complete.")

# Evaluate LightGBM
y_pred_lgbm = lgbm_model.predict(X_test)
rmse_lgbm = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
print(f"LightGBM RMSE: {rmse_lgbm:.4f}")

# --- 4. Train the Random Forest Model ---
print("\n=== Training Random Forest Model ===")
rf_model = RandomForestRegressor(
    n_estimators=500,  # Same number of trees
    max_depth=8,  # Same depth
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',  # Use sqrt of features
    random_state=42,
    n_jobs=-1,
    verbose=0
)

rf_model.fit(X_train, y_train)
print("Random Forest training complete.")

# Evaluate Random Forest
y_pred_rf = rf_model.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f"Random Forest RMSE: {rmse_rf:.4f}")

# --- 5. Train LSTM Model ---
print("\n=== Training LSTM Model ===")

# Scale the data for LSTM (neural networks work better with normalized data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape data for LSTM: (samples, timesteps, features)
# For cache prediction, we treat each sample as a single timestep
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Build LSTM model
lstm_model = Sequential([
    LSTM(128, activation='relu', input_shape=(1, X_train_scaled.shape[1]), return_sequences=True),
    Dropout(0.2),
    LSTM(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression
])

lstm_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)

print("LSTM Architecture:")
lstm_model.summary()

# Train with early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = lstm_model.fit(
    X_train_lstm, y_train,
    validation_data=(X_test_lstm, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

print("LSTM training complete.")

# Evaluate LSTM
y_pred_lstm = lstm_model.predict(X_test_lstm, verbose=0).flatten()
rmse_lstm = np.sqrt(mean_squared_error(y_test, y_pred_lstm))
print(f"LSTM RMSE: {rmse_lstm:.4f}")

# --- 6. Compare All Models ---
print("\n=== Model Comparison ===")
print(f"LightGBM RMSE:      {rmse_lgbm:.4f}")
print(f"Random Forest RMSE: {rmse_rf:.4f}")
print(f"LSTM RMSE:          {rmse_lstm:.4f}")

# Find the best model
rmse_dict = {
    'LightGBM': rmse_lgbm,
    'Random Forest': rmse_rf,
    'LSTM': rmse_lstm
}
best_name = min(rmse_dict, key=rmse_dict.get)
best_rmse = rmse_dict[best_name]

print(f"\nWinner: {best_name} (RMSE: {best_rmse:.4f})")

# --- 7. Save All Models ---
joblib.dump(lgbm_model, 'cache_model_lgbm.pkl')
print(f"\nLightGBM model saved to cache_model_lgbm.pkl")

joblib.dump(rf_model, 'cache_model_rf.pkl')
print(f"Random Forest model saved to cache_model_rf.pkl")

# Save LSTM model and scaler
lstm_model.save('cache_model_lstm.h5')
joblib.dump(scaler, 'cache_model_lstm_scaler.pkl')
print(f"LSTM model saved to cache_model_lstm.h5")
print(f"LSTM scaler saved to cache_model_lstm_scaler.pkl")

# Save the best model info
if best_name == 'LSTM':
    # For LSTM, we need to note this separately since it's a different format
    with open('best_model_info.txt', 'w') as f:
        f.write('LSTM')
    print(f"\nBest model: LSTM (saved as cache_model_lstm.h5)")
elif best_name == 'LightGBM':
    joblib.dump(lgbm_model, 'cache_model.pkl')
    with open('best_model_info.txt', 'w') as f:
        f.write('LightGBM')
    print(f"\nBest model: LightGBM (saved as cache_model.pkl)")
else:
    joblib.dump(rf_model, 'cache_model.pkl')
    with open('best_model_info.txt', 'w') as f:
        f.write('Random Forest')
    print(f"\nBest model: Random Forest (saved as cache_model.pkl)")
