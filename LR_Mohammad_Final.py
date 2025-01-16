# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# Load dataset
X_test = pd.read_csv('X_test_scaled.csv')
X_train = pd.read_csv('X_train_scaled_balanced.csv')
y_test = pd.read_csv('y_test.csv').values.ravel()  # Flatten y_test to 1D array
y_train = pd.read_csv('y_train_balanced.csv').values.ravel()  # Flatten y_train to 1D array

# Create a pipeline with StandardScaler and LogisticRegression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=5000))
])

# Define parameter grid for GridSearchCV
param_grid = {
    'logreg__penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'logreg__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'logreg__solver': ['saga'],  # 'saga' solver supports elasticnet and l1 regularization
    'logreg__l1_ratio': [0, 0.25, 0.5, 0.75, 1]  # Only used with elasticnet penalty
}

# Initialize GridSearchCV with Logistic Regression and parameter grid
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the model
grid_search.fit(X_train, y_train)

# Output the best parameters and accuracy score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Evaluate the model on the test data
y_pred = grid_search.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Get the best model
best_model = grid_search.best_estimator_

# --- Feature Importance ---
if 'logreg__penalty' in grid_search.best_params_ and grid_search.best_params_['logreg__penalty'] != 'none':
    coef = best_model.named_steps['logreg'].coef_[0]
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': coef
    }).sort_values(by='Importance', key=abs, ascending=False)
    print("\nFeature Importance:\n", feature_importance)

# --- Confusion Matrix ---
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# --- ROC Curve ---
y_prob = grid_search.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
plt.show()
