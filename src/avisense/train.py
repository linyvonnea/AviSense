from pathlib import Path
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


def get_models(random_state=42):
    """
    Define baseline classical machine learning models.
    """
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=random_state,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
    }

    return models


def train_and_evaluate_models(X_train, y_train, cv_folds=5, random_state=42):
    """
    Train and cross-validate all baseline models.
    """
    models = get_models(random_state=random_state)

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    rows = []

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        row = {
            "model": model_name,
            "accuracy_mean": scores["test_accuracy"].mean(),
            "accuracy_std": scores["test_accuracy"].std(),
            "precision_macro_mean": scores["test_precision_macro"].mean(),
            "precision_macro_std": scores["test_precision_macro"].std(),
            "recall_macro_mean": scores["test_recall_macro"].mean(),
            "recall_macro_std": scores["test_recall_macro"].std(),
            "f1_macro_mean": scores["test_f1_macro"].mean(),
            "f1_macro_std": scores["test_f1_macro"].std(),
        }

        rows.append(row)

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values("f1_macro_mean", ascending=False)

    return results_df


def fit_best_model(model_name, X_train, y_train, random_state=42):
    """
    Fit selected best model on the full training set.
    """
    models = get_models(random_state=random_state)

    if model_name not in models:
        raise ValueError(f"Unknown model name: {model_name}")

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", models[model_name]),
        ]
    )

    pipeline.fit(X_train, y_train)

    return pipeline


def save_model(model, output_path):
    """
    Save model as joblib file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path)


def load_model(model_path):
    """
    Load joblib model.
    """
    return joblib.load(model_path)