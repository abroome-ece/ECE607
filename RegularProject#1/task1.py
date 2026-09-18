import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the Dataset
# (Make sure 'Salary_Data.csv' is in your working directory or provide the full path)
df = pd.read_csv("dataset/salary/Salary_dataset.csv")

# Extracting features (X) and target (Y)
X = df[["YearsExperience"]]  # Independent variable
Y = df["Salary"]  # Dependent variable

# 2. Design and Train the Linear Regression Model
model = LinearRegression()
model.fit(X, Y)

# 3. Predict on Training Data
Y_pred = model.predict(X)

# --- a) Report Training Performance ---
mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y, Y_pred)

print("======================REPORT======================")
print(f"Intercept (Beta 0):        {model.intercept_:.2f}")
print(f"Coefficient/Slope (Beta 1): {model.coef_[0]:.2f}")
print(f"Mean Squared Error (MSE):   {mse:.2f}")
print(f"Root Mean Squared Error:    {rmse:.2f}")
print(f"R-squared (R2 Score):       {r2:.4f}")
print("=================================================")

# --- b) Plot Dataset Along with Predictions ---
plt.figure(figsize=(8, 5))
plt.scatter(X, Y, color="blue", alpha=0.7, label="Actual Data")
plt.plot(
    X,
    Y_pred,
    color="red",
    linewidth=2,
    label=f"Predicted Line (R2 = {r2:.2f})",
)
plt.title("Salary vs. Years of Experience (Training Data)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()