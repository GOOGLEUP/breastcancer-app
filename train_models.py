"""
Model training and persistence for BITS Pilani ML Assignment 2.

Run:
    python train_models.py

The script trains the five model families required by the assignment and
stores the fitted estimators in model/.
"""

from pathlib import Path
import pickle

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).parent
MODEL_FOLDER = PROJECT_ROOT / "model"
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


def get_data():
    """Return the dataset with readable binary class labels."""
    source = load_breast_cancer(as_frame=True)

    predictors = source.data.copy()
    response = pd.Series(source.target, name="diagnosis").replace(
        {0: "malignant", 1: "benign"}
    )

    return predictors, response


def build_estimators():
    """Create the five classifiers required for the assignment."""
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, random_state=RANDOM_SEED)
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=RANDOM_SEED
        ),
        "knn": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=7)
        ),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
    }


def save_pickle(estimator, destination):
    """Persist a fitted estimator."""
    with destination.open("wb") as handle:
        pickle.dump(estimator, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    X, y = get_data()

    X_fit, X_holdout, y_fit, y_holdout = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y
    )

    # Keep the holdout set separate so that the evaluation data is not used
    # during fitting.
    holdout = X_holdout.copy()
    holdout["diagnosis"] = y_holdout.to_numpy()
    holdout.to_csv(PROJECT_ROOT / "test_data.csv", index=False)

    for name, estimator in build_estimators().items():
        estimator.fit(X_fit, y_fit)
        save_pickle(estimator, MODEL_FOLDER / f"{name}.pkl")
        print(f"Saved {name}.pkl")

    print(
        f"\nTraining complete: {len(X_fit)} training rows and "
        f"{len(X_holdout)} holdout rows."
    )


if __name__ == "__main__":
    main()
