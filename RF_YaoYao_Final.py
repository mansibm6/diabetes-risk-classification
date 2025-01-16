import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt


X_train = np.load("/Users/yaoyao/Downloads/X_train_scaled_balanced.npy")
X_test = np.load("/Users/yaoyao/Downloads/X_test_scaled.npy")
y_train = np.load("/Users/yaoyao/Downloads/y_train_balanced.npy")
y_test = np.load("/Users/yaoyao/Downloads/y_test.npy")


rf = RandomForestClassifier(random_state=42)


rf.fit(X_train, y_train)


y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(y_test, y_pred)
print("Test Set Accuracy:", accuracy)

f1 = f1_score(y_test, y_pred, average='weighted')
print("Test Set F1 Score:", f1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()


conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=rf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()


feature_importances = pd.DataFrame({
    'Feature': [f"Feature {i}" for i in range(X_train.shape[1])],
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importances)



param_distributions = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}


rf = RandomForestClassifier(random_state=42)


random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=50,  # Number of parameter settings sampled
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    verbose=2,
    n_jobs=-1,
    random_state=42
)


random_search.fit(X_train, y_train)


best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]


print("\nBest Parameters:", random_search.best_params_)

accuracy = accuracy_score(y_test, y_pred)
print("Test Set Accuracy:", accuracy)

f1 = f1_score(y_test, y_pred, average='weighted')
print("Test Set F1 Score:", f1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()


feature_importances = pd.DataFrame({
    'Feature': [f"Feature {i}" for i in range(X_train.shape[1])],
    'Importance': best_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importances)


conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=best_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()