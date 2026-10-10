"""Run the original model experiment with explicit local inputs."""

import argparse
from pathlib import Path
from data_io import load_splits, logistic_grid
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--format", choices=["csv", "npy"], default="csv")
    args = parser.parse_args()
    run(args.data_dir, args.format)


if __name__ == "__main__":
    main()
