from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC

from avisense.config import load_config, ensure_directories
from avisense.dataset import (
    load_features,
    prepare_xy,
    encode_labels,
    make_train_test_split,
)


def tune_xgboost(X_train, y_train, cv, random_state):
    from xgboost import XGBClassifier

    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
    }

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=2,
    )

    search.fit(X_train, y_train)
    return search


def tune_random_forest(X_train, y_train, cv, random_state):
    model = RandomForestClassifier(
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    param_grid = {
        "n_estimators": [300, 500, 700],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=2,
    )

    search.fit(X_train, y_train)
    return search


def tune_extra_trees(X_train, y_train, cv, random_state):
    model = ExtraTreesClassifier(
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    param_grid = {
        "n_estimators": [300, 500, 700],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=2,
    )

    search.fit(X_train, y_train)
    return search


def tune_svm(X_train, y_train, cv, random_state):
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    probability=True,
                    random_state=random_state,
                ),
            ),
        ]
    )

    param_grid = {
        "model__C": [0.1, 1, 10, 50, 100],
        "model__gamma": ["scale", 0.001, 0.01, 0.1],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=2,
    )

    search.fit(X_train, y_train)
    return search


def main():
    config = load_config()
    ensure_directories(config)

    random_state = config["training"]["random_state"]

    features_path = Path(config["paths"]["features_dir"]) / "recording_features.csv"
    reports_dir = Path(config["paths"]["reports_dir"])
    models_dir = Path(config["paths"]["models_dir"])

    df = load_features(features_path)

    X, y = prepare_xy(df, label_column="species")
    y_encoded, label_encoder = encode_labels(y)

    X_train, X_test, y_train, y_test = make_train_test_split(
        X,
        y_encoded,
        test_size=config["training"]["test_size"],
        random_state=random_state,
    )

    cv = StratifiedKFold(
        n_splits=config["training"]["cv_folds"],
        shuffle=True,
        random_state=random_state,
    )

    searches = {}

    print("\nTuning XGBoost...")
    searches["xgboost_tuned"] = tune_xgboost(X_train, y_train, cv, random_state)

    print("\nTuning Random Forest...")
    searches["random_forest_tuned"] = tune_random_forest(
        X_train, y_train, cv, random_state
    )

    print("\nTuning Extra Trees...")
    searches["extra_trees_tuned"] = tune_extra_trees(
        X_train, y_train, cv, random_state
    )

    print("\nTuning SVM...")
    searches["svm_rbf_tuned"] = tune_svm(X_train, y_train, cv, random_state)

    rows = []

    for name, search in searches.items():
        rows.append(
            {
                "model": name,
                "best_cv_f1_macro": search.best_score_,
                "best_params": search.best_params_,
            }
        )

        joblib.dump(search.best_estimator_, models_dir / f"{name}.joblib")

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values("best_cv_f1_macro", ascending=False)

    results_path = reports_dir / "tuned_model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    best_name = results_df.iloc[0]["model"]
    best_model = searches[best_name].best_estimator_

    joblib.dump(best_model, models_dir / "best_tuned_model.joblib")
    joblib.dump(label_encoder, models_dir / "label_encoder.joblib")

    joblib.dump(
        {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "test_indices": X_test.index.tolist(),
            "train_indices": X_train.index.tolist(),
            "feature_columns": X.columns.tolist(),
            "class_names": label_encoder.classes_.tolist(),
            "best_model_name": best_name,
        },
        models_dir / "tuned_train_test_split.joblib",
    )

    print("\nTuned model comparison:")
    print(results_df)

    print(f"\nBest tuned model: {best_name}")
    print(f"Saved tuned results to: {results_path}")
    print(f"Saved best tuned model to: {models_dir / 'best_tuned_model.joblib'}")


if __name__ == "__main__":
    main()