import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from typing import Union, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score
)


def load_data(
        expr_data: str, 
        meta_data: str, 
        extra_strat: bool = False
        ) -> Union[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series, pd.Series]]:

    """Load meta and data, drop nas and return X and y."""
    expr_df = pd.read_csv(expr_data, sep='\t', header=0, index_col=0).T
    meta_df = pd.read_csv(meta_data, sep='\t', header=0, index_col=0)

    # align meta just in case input is not aligned
    meta_df = meta_df.loc[expr_df.index]

    y = meta_df['ClusterK4_kmeans']
    mask = y.notna()

    if not extra_strat:
        X = expr_df.loc[mask]
        y = y[mask]
        return X, y
    
    else:
        race = meta_df['race']
        X = expr_df.loc[mask]
        y = y[mask]
        race = race[mask]
        return X, y, race


def plot_stats(
        y_true: pd.Series,
        y_pred: pd.Series,
        fig_file: Path,
        sample_name: str,
        ) -> None:
    """Function to plot confusion matrix for all classses"""
    labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        cbar_kws={'label': 'fraction of true class'},
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion matrix ({sample_name} individuals)')
    plt.tight_layout()
    plt.savefig(fig_file, dpi=150, bbox_inches='tight')
    plt.close()


def make_pipeline(C: float) -> Pipeline:

    """
    Combine all processing in one pipeline to avoid leakage.
    Log-transform, scale, L2 reg
    """
    pipe = Pipeline([
        ('log', FunctionTransformer(np.log1p)),
        ('scale', StandardScaler()),
        ('classifier', LogisticRegression(
            penalty='l2',
            #multi_class='multinomial',
            class_weight='balanced',
            max_iter=1000,
            C=C
        ))
    ])

    return pipe


def ev_cv(
        pipe: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int,
        ) -> pd.Series:
    """
    Stratified cross validation. Plotting Balanced accuracy, F1 annd ROC AUR
    """
    # Cluster are inbalanced, so each fold will preserve class distribution.
    skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=314)

    # cross val
    y_proba = cross_val_predict(pipe, X, y, cv=skfold, method='predict_proba')

    # get labels from classes
    # retain the same order of labels!!
    classes = np.unique(y)
    y_pred = pd.Series(classes[y_proba.argmax(axis=1)], index=y.index)

    # get metrics for imbalanced classes
    balanced_acc = balanced_accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average='macro')
    auc = roc_auc_score(y, y_proba, multi_class='ovr', average='macro')

    print('=========Stats================')
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nPer-class:")
    print(classification_report(y, y_pred))
    print(f"Macro ROC AUC: {auc:.3f}")

    return y_pred