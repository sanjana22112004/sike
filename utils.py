import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_correlation_matrix(df: pd.DataFrame):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=False, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation heatmap (numeric features)")
    fig.tight_layout()
    return fig


def plot_target_distribution(y: pd.Series, task_type: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    if task_type == "classification":
        y.value_counts().plot(kind="bar", ax=ax, color="#4C78A8")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title("Target class distribution")
    else:
        sns.histplot(y, kde=True, ax=ax, color="#72B7B2")
        ax.set_xlabel("Target value")
        ax.set_title("Target distribution")
    fig.tight_layout()
    return fig


def plot_top_feature_distributions(df: pd.DataFrame, top_n: int = 6):
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()[:top_n]
    if not numeric_cols:
        return None
    n = len(numeric_cols)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = axes.flatten() if isinstance(axes, (list, tuple, pd.Series)) else axes.ravel()
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="#E45756")
        axes[i].set_title(f"{col}")
    for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j])
    fig.suptitle("Top numeric feature distributions", y=1.02)
    fig.tight_layout()
    return fig


def eda_summary(df: pd.DataFrame):
    info = {
        "rows": len(df),
        "columns": df.shape[1],
        "numeric_cols": len(df.select_dtypes(include=["number"]).columns),
        "categorical_cols": len(df.select_dtypes(exclude=["number"]).columns),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_pct": float((df.isna().sum().sum() / (df.shape[0] * max(df.shape[1], 1))) * 100.0) if df.size else 0.0,
    }
    missing_by_col = df.isna().mean().sort_values(ascending=False)
    return info, missing_by_col
