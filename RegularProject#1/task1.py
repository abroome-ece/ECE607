from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def analyze_salary_trends(file_path: str | Path, show_plot: bool = True) -> dict[str, float]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path.resolve()}")
        
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV file: {e}")
        
    required_cols = {"YearsExperience", "Salary"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Dataset is missing required columns. Expected: {required_cols}, Found: {list(df.columns)}"
        )
        
    X = df[["YearsExperience"]]
    y = df["Salary"]
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    
    metrics = {
        "intercept": float(model.intercept_),
        "coefficient": float(model.coef_[0]),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
    }
    
    if show_plot:
        plt.figure(figsize=(8, 5))
        plt.scatter(X, y, color="blue", alpha=0.7, label="Actual Data")
        plt.plot(X, y_pred, color="red", linewidth=2, label=f"R2 = {r2:.2f}")
        plt.title("Salary vs. Years of Experience")
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.show()
        
    return metrics

if __name__ == "__main__":
    try:
        results = analyze_salary_trends("dataset/salary/Salary_dataset.csv")
        print("Analysis completed successfully:")
        for metric, value in results.items():
            print(f"  {metric.upper()}: {value:.4f}")
    except (FileNotFoundError, ValueError, RuntimeError) as err:
        print(f"Execution failed: {err}")