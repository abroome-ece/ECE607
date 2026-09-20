import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the datasets
train_df = pd.read_csv("dataset/Linear_Regression/train.csv")
test_df = pd.read_csv("dataset/Linear_Regression/test.csv")

# drop missing data
train_df = train_df.dropna()
test_df = test_df.dropna()

# 2. features and targer
X_train = train_df[['x']]
y_train = train_df['y']

X_test = test_df[['x']]
y_test = test_df['y']

# 3. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Print learned parameters
print(f"Intercept (beta_0): {model.intercept_:.4f}")
print(f"Slope (beta_1): {model.coef_[0]:.4f}")

# 4. Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 5. Performance Metrics
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)

test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)

print("\n--- Training Performance ---")
print(f"MSE:  {train_mse:.4f}")
print(f"RMSE: {train_rmse:.4f}")
print(f"R2:   {train_r2:.4f}")

print("\n--- Testing Performance ---")
print(f"MSE:  {test_mse:.4f}")
print(f"RMSE: {test_rmse:.4f}")
print(f"R2:   {test_r2:.4f}")