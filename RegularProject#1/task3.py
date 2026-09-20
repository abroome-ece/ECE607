import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. Load the dataset
df = pd.read_csv("dataset/wine_quality/winequality-red.csv")

# quality is target
X = df.drop('quality', axis=1)
y_raw = df['quality']

# 2. quality > 5 is 1, otherwise 0
y = (y_raw > 5).astype(int)

# 3. make model
model = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))

# 4. 3-fold cross-validation
kf = KFold(n_splits=3, shuffle=True, random_state=0)

# metrics to score with
scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']
cv_results = cross_validate(model, X, y, cv=kf, scoring=scoring_metrics)

# report fold results
for i in range(len(cv_results['test_accuracy'])):
    print(f"--- Fold {i + 1} ---")
    print(f"Accuracy:  {cv_results['test_accuracy'][i]:.4f}")
    print(f"Precision: {cv_results['test_precision'][i]:.4f}")
    print(f"Recall:    {cv_results['test_recall'][i]:.4f}")
    print(f"F1-Score:  {cv_results['test_f1'][i]:.4f}\n")

# report mean results 

print(f"--- Mean ---")
print(f"Mean Accuracy:  {cv_results['test_accuracy'].mean():.4f}")
print(f"Mean Precision: {cv_results['test_precision'].mean():.4f}")
print(f"Mean Recall:    {cv_results['test_recall'].mean():.4f}")
print(f"Mean F1-Score:  {cv_results['test_f1'].mean():.4f}")