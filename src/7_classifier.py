import argparse
import pickle
import pandas as pd

from pathlib import Path
from sklearn.pipeline import Pipeline
from data_utils import load_data, plot_stats, make_pipeline, ev_cv

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
        '-cmap',
        default='YlGnBu',
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
    X, y = load_data(args.expr, args.meta)

    # building pipeline
    pipe = make_pipeline(args.C)

    # run cross val
    y_pred = ev_cv(pipe, X, y, args.n_splits)

    # plot stats
    plot_stats(y, y_pred, fig_path / f'7_{args.sample}_individuals_confMatrix.png', args.sample, cmap=args.cmap)

    # train and save model
    train_final(pipe, X, y, out_path / f'{args.sample}_logreg.pkl')


if __name__ == "__main__":
    main()