"""Run the original model experiment with explicit local inputs."""

import argparse
from pathlib import Path
from data_io import load_splits, logistic_grid
import pandas as pd
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
    roc_auc_score,
)


def run(data_dir, format="csv"):
    X_train, X_test, y_train, y_test = load_splits(data_dir, format)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=5000, random_state=42)),
        ]
    )

    param_grid = logistic_grid()

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)

    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)

    print("Best Cross-Validation Accuracy:", grid_search.best_score_)

    y_pred = grid_search.predict(X_test)

    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    best_model = grid_search.best_estimator_

    if True:  # Coefficients exist for all candidate logistic models.
        coef = best_model.named_steps["logreg"].coef_[0]
        feature_importance = pd.DataFrame(
            {"Feature": X_train.columns, "Importance": coef}
        ).sort_values(by="Importance", key=abs, ascending=False)
        print("\nFeature Importance:\n", feature_importance)

    conf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Class 0", "Class 1"],
        yticklabels=["Class 0", "Class 1"],
    )

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted Labels")

    plt.ylabel("True Labels")

    plt.show()

    y_prob = grid_search.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    roc_auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(8, 6))

    plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")

    plt.title("Receiver Operating Characteristic (ROC) Curve")

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.legend(loc="lower right")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--format", choices=["csv", "npy"], default="csv")
    args = parser.parse_args()
    run(args.data_dir, args.format)


if __name__ == "__main__":
    main()
