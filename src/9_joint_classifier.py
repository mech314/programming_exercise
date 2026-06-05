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
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
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
        '-out_path', 
        type=str, 
        default='models', 
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
    parser.add_argument(
        '-test_size', 
        type=float, 
        default=0.2, 
        help='Holdout size')

    return parser.parse_args()


def load_data(
        expr_data: str, 
        meta_data: str
        ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:

    """Load meta and data, drop nas and return X and y."""
    # if you use output from previous tasks we don't need to transpose.
    expr_df = pd.read_csv(expr_data, sep='\t', header=0, index_col=0) 
    meta_df = pd.read_csv(meta_data, sep='\t', header=0, index_col=0)

    # align meta just in case input is not aligned
    meta_df = meta_df.loc[expr_df.index]

    y = meta_df['ClusterK4_kmeans']
    race = meta_df['race']
    mask = y.notna()
    X = expr_df.loc[mask]
    y = y[mask]
    race = race[mask]

    return X, y, race


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
    Stratify by calss.
    """
    # Cluster are inbalanced, so each fold will preserve class distribution.
    skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=314)

    # cross val
    y_pred = cross_val_predict(pipe, X, y, cv=skfold)
    y_pred = pd.Series(y_pred, index=y.index)

    # get metrics for imbalanced classes
    balanced_acc = balanced_accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average='macro')

    print('=========Stats================')
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nPer-class:")
    print(classification_report(y, y_pred))

    y_probabilities = cross_val_predict(pipe, X, y, cv=skfold, method='predict_proba')
    auc = roc_auc_score(y, y_probabilities, multi_class='ovr', average='macro')
    print(f"Macro ROC AUC: {auc:.3f}")

    return y_pred


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


def train_final(
        pipe: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        model_file: Path, 
    ) -> Pipeline:
    """Fit final model"""

    pipe.fit(X, y)
    # save as pickle to use later
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

    # Getting data
    X, y, race = load_data(args.expr, args.meta)

    # It seems that we need to stratify here by both race and subtype
    strat = y.astype(str) + "_" + race.astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=strat,
        random_state=314,
    )

    # building pipeline
    pipe = make_pipeline(args.C)

    # run cross val
    y_cv_pred = ev_cv(pipe, X_train, y_train, args.n_splits)

    # don't really need to plot CV resuls since we are evaluating holdout
    # # plot stats
    # plot_stats(y_train, y_cv_pred, fig_path / f'{args.sample}_individuals_confMatrix.png', 'white')

    # train and save model
    pipe = train_final(pipe, X_train, y_train, out_path / f'{args.sample}_logreg.pkl')

    # test holdout dataset
    y_test_pred = pd.Series(pipe.predict(X_test), index=y_test.index)
    y_test_proba =pipe.predict_proba(X_test)

    print("=== Holdout (20%) evaluation ===")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Macro F1:          {f1_score(y_test, y_test_pred, average='macro'):.4f}")
    print(classification_report(y_test, y_test_pred))
    print(f"Macro ROC AUC: {roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='macro'):.4f}")

    plot_stats(y_test, y_test_pred, fig_path / f'{args.sample}_holdout_confMatrix.png', f'{args.sample}_holdout')


if __name__ == "__main__":
    main()