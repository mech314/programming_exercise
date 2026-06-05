import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Compare subtype proportions between races')

    parser.add_argument(
        '-meta', 
        required=True, 
        type=str, 
        help='Path to merged black/white metadata')
    parser.add_argument(
        '-out_path', 
        type=str, 
        default='out', 
        help='Folder to save tables')
    parser.add_argument(
        '-fig_path', 
        type=str, 
        default='figs', 
        help='Folder to save figures')
    parser.add_argument(
        '-cluster_col', 
        type=str, 
        default='ClusterK4_kmeans',
        help='Cluster column to compare (named subtypes by default)')

    return parser.parse_args()


def subtype_proportions(
        meta_df: pd.DataFrame,
        cluster_col: str,
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calclate per-race raw counts and proportions of each subtype."""
    counts = meta_df.groupby('race')[cluster_col].value_counts().unstack()
    prop = meta_df.groupby('race')[cluster_col].value_counts(normalize=True).unstack()
    return counts, prop


def plot_proportions(
        prop: pd.DataFrame,
        fig_file: Path,
        ) -> None:
    """Grouped bar plot of subtype proportions by race."""
    ax = prop.T.plot(kind='bar', figsize=(7, 4))
    ax.set_ylabel('Proportion within race')
    ax.set_xlabel('HGSC subtype')
    ax.set_title('Subtype proportions by race')
    ax.legend(title='race')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(fig_file, dpi=150, bbox_inches='tight')


def main() -> None:

    args = get_args()

    # make sure paths exists
    out_path = Path(args.out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    fig_path = Path(args.fig_path)
    fig_path.mkdir(parents=True, exist_ok=True)

    meta_df = pd.read_csv(args.meta, sep='\t', index_col=0)

    counts, prop = subtype_proportions(meta_df, args.cluster_col)
    print('Raw counts:')
    print(counts, '\n')
    print('Proportions within race:')
    print(prop)

    plot_proportions(prop, fig_path / 'subtype_proportions_by_race.png')


if __name__ == '__main__':
    main()