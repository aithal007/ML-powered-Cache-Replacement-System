import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
import sys

print("Starting model training for both LightGBM and Random Forest...")

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

# --- 5. Compare Models ---
print("\n=== Model Comparison ===")
print(f"LightGBM RMSE:     {rmse_lgbm:.4f}")
print(f"Random Forest RMSE: {rmse_rf:.4f}")

if rmse_lgbm < rmse_rf:
    print(f"Winner: LightGBM (better by {rmse_rf - rmse_lgbm:.4f})")
    best_model = lgbm_model
    best_name = "LightGBM"
else:
    print(f"Winner: Random Forest (better by {rmse_lgbm - rmse_rf:.4f})")
    best_model = rf_model
    best_name = "Random Forest"

# --- 6. Save Both Models ---
joblib.dump(lgbm_model, 'cache_model_lgbm.pkl')
print(f"\nLightGBM model saved to cache_model_lgbm.pkl")

joblib.dump(rf_model, 'cache_model_rf.pkl')
print(f"Random Forest model saved to cache_model_rf.pkl")

# Save the better model as default
joblib.dump(best_model, 'cache_model.pkl')
print(f"\nBest model ({best_name}) saved as cache_model.pkl")
