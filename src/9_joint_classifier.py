import argparse
import pickle
import pandas as pd

from pathlib import Path
from data_utils import load_data, plot_stats, make_pipeline, ev_cv

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
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
    parser.add_argument(
        '-cmap',
        default='mako_r',
        type=str,
        help='Color pallet fpr confusion matrix'
    )

    return parser.parse_args()


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
    X, y, race = load_data(args.expr, args.meta, extra_strat=True)

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
    y_test_proba = pipe.predict_proba(X_test)

    print("===Holdout (20%) evaluation===")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Macro F1:          {f1_score(y_test, y_test_pred, average='macro'):.4f}")
    print(classification_report(y_test, y_test_pred))
    print(f"Macro ROC AUC: {roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='macro'):.4f}")

    plot_stats(
        y_test, 
        y_test_pred, 
        fig_path / f'9_{args.sample}_confMatrix.png', 
        f'{args.sample}',
        cmap=args.cmap
        )


if __name__ == "__main__":
    main()