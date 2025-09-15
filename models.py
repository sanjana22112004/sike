from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
import numpy as np


def select_and_train_models(task_type: str):
    if task_type == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        }
    else:
        return {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=300, random_state=42),
        }


def evaluate_model(task_type: str, model, X_test, y_test):
    preds = model.predict(X_test)
    if task_type == "classification":
        acc = accuracy_score(y_test, preds)
        # Probabilities for AUC if available
        try:
            proba = model.predict_proba(X_test)
            if proba.shape[1] == 2:
                auc = roc_auc_score(y_test, proba[:, 1])
            else:
                auc = np.nan
        except Exception:
            auc = np.nan
        f1 = f1_score(y_test, preds, average="weighted")
        return {"accuracy": acc, "f1": f1, "auc": auc}
    else:
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        return {"MSE": mse, "RMSE": rmse, "R2": r2}


def cross_validate_model(model, X_train, y_train, scoring: str):
    try:
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring, n_jobs=-1)
        return float(np.mean(scores)), float(np.std(scores))
    except Exception:
        return np.nan, np.nan


def train_and_evaluate_automl(task_type, preprocess, X_train, X_test, y_train, y_test):
    from sklearn.pipeline import Pipeline

    models = select_and_train_models(task_type)
    leaderboard = {}
    best_model = None
    best_score = -np.inf if task_type == "classification" else np.inf

    for name, base_model in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocess), ("model", base_model)])
        # Cross-val scoring choice
        scoring = "accuracy" if task_type == "classification" else "neg_root_mean_squared_error"
        cv_mean, cv_std = cross_validate_model(pipe, X_train, y_train, scoring)

        pipe.fit(X_train, y_train)
        metrics = evaluate_model(task_type, pipe, X_test, y_test)
        leaderboard[name] = {"cv_mean": cv_mean, "cv_std": cv_std, **metrics}

        # Determine best: maximize accuracy, minimize RMSE
        primary = metrics["accuracy"] if task_type == "classification" else -metrics["RMSE"]
        if primary > best_score:
            best_score = primary
            best_model = (name, pipe)

    # Feature influence via permutation importance
    try:
        perm = permutation_importance(best_model[1], X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
        feature_names = []
        # Attempt to get transformed feature names
        try:
            feature_names = best_model[1].named_steps["preprocess"].get_feature_names_out()
        except Exception:
            feature_names = [f"f{i}" for i in range(len(perm.importances_mean))]
        importances = sorted(zip(feature_names, perm.importances_mean), key=lambda x: x[1], reverse=True)
    except Exception:
        importances = []

    reasoning = build_model_reasoning(task_type, leaderboard)

    return {
        "leaderboard": leaderboard,
        "best_model_name": best_model[0] if best_model else None,
        "best_model": best_model[1] if best_model else None,
        "feature_importance": importances,
        "reasoning": reasoning,
    }


def build_model_reasoning(task_type: str, leaderboard: dict) -> str:
    # Simple human-readable rationale
    if not leaderboard:
        return "No models evaluated."
    if task_type == "classification":
        # pick max accuracy
        best = max(leaderboard.items(), key=lambda kv: kv[1].get("accuracy", float("-inf")))
        name, metrics = best
        return (
            f"Selected {name} because it achieved the highest validation accuracy "
            f"({metrics.get('accuracy', 'n/a'):.3f}). Logistic Regression is strong for linearly-separable data, "
            f"while Random Forest captures nonlinearities and interactions without heavy tuning."
        )
    else:
        best = min(leaderboard.items(), key=lambda kv: kv[1].get("RMSE", float("inf")))
        name, metrics = best
        return (
            f"Selected {name} because it achieved the lowest error (RMSE={metrics.get('RMSE', float('nan')):.3f}). "
            f"Linear/Ridge models are interpretable baselines; Random Forest handles nonlinear patterns and mixed dtypes."
        )
