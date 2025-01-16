# Import required libraries
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Load dataset
X_test = pd.read_csv('X_test_scaled.csv')
X_train = pd.read_csv('X_train_scaled_balanced.csv')
y_test = pd.read_csv('y_test.csv')
y_train = pd.read_csv('y_train_balanced.csv')

# Create a pipeline with StandardScaler and LogisticRegression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression())
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
