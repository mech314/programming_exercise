import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Args to train classifier')

    parser.add_argument(
        '-meta', 
        required=True, 
        type=str, 
        help='Path to metadata')
    parser.add_argument(
        '-expr', 
        required=True, 
        type=str, 
        help='Path to expression table)')
    parser.add_argument(
        '-model', 
        required=True,
        type=str,
        help='Path to pretrained model')
    parser.add_argument(
        '-out_path', 
        type=str, 
        default='out', 
        help='Folder to save model')
    parser.add_argument(
        '-fig_path', 
        type=str, 
        default='figs', 
        help='Folder to save figures')
    parser.add_argument(
        '-n_splits', 
        type=int, 
        default=5, 
        help='CV folds')
    parser.add_argument(
        '-C', 
        type=float, 
        default=1.0, 
        help='Reg strength')
    parser.add_argument(
        '-sample',
        required=True,
        type=str,
        help="Sample name")


    return parser.parse_args()


def load_data(expr_data: str, meta_data: str) -> tuple[pd.DataFrame, pd.Series]:

    """Load meta and data, drop nas and return X and y."""
    expr_df = pd.read_csv(expr_data, sep='\t', header=0, index_col=0).T
    meta_df = pd.read_csv(meta_data, sep='\t', header=0, index_col=0)

    # align meta just in case input is not aligned
    meta_df = meta_df.loc[expr_df.index]

    y = meta_df['ClusterK4_kmeans']
    mask = y.notna()
    X = expr_df.loc[mask]
    y = y[mask]

    return X, y


def load_model(model_file: str) -> Pipeline:
    """Load the pretrained model."""
    with open(model_file, 'rb') as f:
        return pickle.load(f)


def plot_stats(
        y_true: pd.Series,
        y_pred: pd.Series,
        fig_file: Path,
        sample_name: str
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


def main() -> None:
    
    args = get_args()

    # make sure paths exists
    out_path = Path(args.out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    fig_path = Path(args.fig_path)
    fig_path.mkdir(parents=True, exist_ok=True)

    # Getting data
    X, y = load_data(args.expr, args.meta)

    # building pipeline
    pipe = load_model(args.model)

    # run cross val
    y_pred = pd.Series(pipe.predict(X), index=y.index) 

    print(f"Balanced accuracy: {balanced_accuracy_score(y, y_pred):.4f}")
    print(f"Macro F1:          {f1_score(y, y_pred, average='macro'):.4f}")
    print(classification_report(y, y_pred))

    y_proba = pipe.predict_proba(X)
    print(f"Macro ROC AUC: {roc_auc_score(y, y_proba, multi_class='ovr', average='macro'):.4f}")

    # plot stats
    plot_stats(y, y_pred, fig_path / f'{args.sample}_individuals_confMatrix.png', 'black')


if __name__ == "__main__":
    main()