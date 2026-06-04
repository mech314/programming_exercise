import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Provide paths')

    parser.add_argument(
        '-black_expr', 
        required=True, 
        type=str, 
        help='Expression table for black ind')
    parser.add_argument(
        '-black_meta', 
        required=True, 
        type=str, 
        help='Metadata for black data')
    parser.add_argument(
        '-white_expr', 
        required=True, 
        type=str, 
        help='Expression table for white ind')
    parser.add_argument(
        '-white_meta', 
        required=True, 
        type=str, 
        elp='Metadata for white data')
    parser.add_argument(
        '-fig_path', 
        type=str, 
        default='figs', 
        help='Folder to save figures')
    parser.add_argument(
        '-num_components', 
        default=3, 
        type=int, 
        help='Number of PCs')

    return parser.parse_args()


def run_pca(
        merged_df: pd.DataFrame, 
        components: int
        ) -> tuple[pd.DataFrame, dict]:

    # log-transform to tame skewed range, then standardize per gene
    X = np.log1p(merged_df)
    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=components)
    pcs = pca.fit_transform(X)

    pca_df = pd.DataFrame(pcs, columns=['PC1', 'PC2', 'PC3'], index=merged_df.index)

    var = pca.explained_variance_ratio_[:3] * 100
    var = {'PC1': var[0], 'PC2': var[1], 'PC3': var[2]}
    print(f"Variance: PC1: {var['PC1']:.1f}%, PC2: {var['PC2']:.1f}%, PC3: {var['PC3']:.1f}%")

    return pca_df, var


def plot_pca(
        df: pd.DataFrame,
        pc_var: dict,
        color_col: str,
        title: str,
        out_path: str
        ) -> None:

    pc_pairs = [('PC1', 'PC2'), ('PC2', 'PC3'), ('PC1', 'PC3')]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    groups = df[color_col].astype('category')
    cats = groups.cat.categories
    colors = plt.cm.tab10(range(len(cats)))

    for ax, (x, y) in zip(axes, pc_pairs):
        for cat, c in zip(cats, colors):
            mask = groups == cat
            ax.scatter(df.loc[mask, x], df.loc[mask, y],
                       s=18, alpha=0.7, color=c, label=str(cat))
        ax.set_xlabel(f"{x} ({pc_var[x]:.1f}%)")
        ax.set_ylabel(f"{y} ({pc_var[y]:.1f}%)")

    axes[0].legend(title=color_col, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()

    fig.savefig(f'{out_path}/PCA_{color_col}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():

    args = get_args()
    Path(args.fig_path).mkdir(parents=True, exist_ok=True)

    # load and merge expression data
    black_df = pd.read_csv(args.black_expr, sep='\t', header=0, index_col=0)
    white_df = pd.read_csv(args.white_expr, sep='\t', header=0, index_col=0)

    # merge expression on shared genes, then transpose to samples x genes
    merged_df = pd.concat([black_df, white_df], axis=1, join='inner').T

    # load and merge metadata
    black_meta_df = pd.read_csv(args.black_meta, sep='\t', header=0)
    white_meta_df = pd.read_csv(args.white_meta, sep='\t', header=0)

    # add race label to metadata
    black_meta_df['race'] = 'black'
    white_meta_df['race'] = 'white'

    merged_meta_df = pd.concat([black_meta_df, white_meta_df], axis=0)
    # align metadata to expression sample order for correct coloring
    merged_meta_df = merged_meta_df.set_index('ID').loc[merged_df.index]

    pca_df, var = run_pca(merged_df, args.num_components)
    pca_df = pca_df.join(merged_meta_df[['ClusterK4_kmeans', 'race']])

    plot_pca(pca_df, var, 'ClusterK4_kmeans', 
             'PCA colored by ClusterK4_kmeans subtype', args.fig_path)
    plot_pca(pca_df, var, 'race', 'PCA colored by race', args.fig_path)


if __name__ == "__main__":
    main()