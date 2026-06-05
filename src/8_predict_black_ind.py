import argparse
import pickle
import pandas as pd

from pathlib import Path
from data_utils import load_data, plot_stats

from sklearn.pipeline import Pipeline
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
        '-sample',
        required=True,
        type=str,
        help="Sample name")
    parser.add_argument(
        '-cmap',
        default='YlOrBr',
        type=str,
        help='Color pallet fpr confusion matrix'
    )


    return parser.parse_args()


def load_model(model_file: str) -> Pipeline:
    """Load the pretrained model."""
    with open(model_file, 'rb') as f:
        return pickle.load(f)


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

    # predict
    y_pred = pd.Series(pipe.predict(X), index=y.index) 

    print(f"Balanced accuracy: {balanced_accuracy_score(y, y_pred):.4f}")
    print(f"Macro F1:          {f1_score(y, y_pred, average='macro'):.4f}")
    print(classification_report(y, y_pred))

    y_proba = pipe.predict_proba(X)
    print(f"Macro ROC AUC: {roc_auc_score(y, y_proba, multi_class='ovr', average='macro'):.4f}")

    # plot stats
    plot_stats(y, y_pred, fig_path / f'8_{args.sample}_individuals_confMatrix.png', args.sample, cmap=args.cmap)


if __name__ == "__main__":
    main()