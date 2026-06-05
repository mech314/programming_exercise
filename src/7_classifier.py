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


    return parser.parse_args()


def load_data(expr_data: str, meta_data: str) -> tuple[pd.DataFrame, pd.Series]:
    expr_df = pd.read_csv(expr_data, sep='\t', header=0, index_col=0).T
    meta_df = pd.read_csv(meta_data, sep='\t', header=0, index_col=0)

    # align meta just in case input is not aligned
    meta_df = meta_df.loc[expr_df.index]

    y = meta_df['ClusterK4_kmeans']
    mask = y.notna()
    X = expr_df.loc[mask]
    y = y[mask]

    return X, y


def make_pipeline(C: float) -> Pipeline:
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
    
    skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=314)

    y_pred = cross_val_predict(pipe, X, y, cv=skfold)
    y_pred = pd.Series(y_pred, index=y.index)

    balanced_acc = balanced_accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average='macro')

    print('=========Stats================')
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")
    print("\nPer-class:")
    print(classification_report(y, y_pred))

    return y_pred


def plot_stats(
        y_true: pd.Series,
        y_pred: pd.Series,
        fig_file: Path,
        ) -> None:
    """confusion matrix."""
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
    plt.title('Confusion matrix (white, cross-validated)')
    plt.tight_layout()
    plt.savefig(fig_file, dpi=150, bbox_inches='tight')
    plt.close()


def train_final(
        pipe: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        model_file: Path, 
    ) -> Pipeline:
    """Fit final model"""
    pipe.fit(X, y)
    with open(model_file, 'wb') as f:
        pickle.dump(pipe, f)

    return pipe


def main() -> None:
    
    args = get_args()

    # make sure paths exists
    out_path = Path(args.out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    fig_path = Path(args.fig_path)
    fig_path.mkdir(parents=True, exist_ok=True)

    X, y = load_data(args.expr, args.meta)

    pipe = make_pipeline(args.C)

    y_pred = ev_cv(pipe, X, y, args.n_splits)

    plot_stats(y, y_pred, fig_path / 'white_individuals_conf.png')

    train_final(pipe, X, y, out_path / 'white_logreg.pkl')


if __name__ == "__main__":
    main()