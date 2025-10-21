import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
import sys

print("Starting model training...")

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
print("Training enhanced LightGBM model...")
model = lgb.LGBMRegressor(
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

model.fit(X_train, y_train)
print("Model training complete.")

# --- 4. Evaluate the Model ---
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Model RMSE on validation set: {rmse:.4f}")

# --- 5. Save the Trained Model ---
joblib.dump(model, 'cache_model.pkl')
print(f"Model saved successfully to cache_model.pkl")
