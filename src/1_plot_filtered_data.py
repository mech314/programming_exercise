import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from matplotlib.patches import Patch

TAN = '#D4B95E'        # black individuals
TURQUOISE = '#5BC0BE'  # white individuals


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Plot gene counts before/after filtering')

    parser.add_argument('-summary', required=True, type=str, help='Path to filter_summary.tsv')
    parser.add_argument('-fig_path', type=str, default='figs', help='Folder to save figure')

    return parser.parse_args()


def plot_gene_counts(summary: pd.DataFrame, fig_file: Path) -> None:
    """Grouped by race; each race has before (full) and after (translucent) bars."""
    races = ['black', 'white']
    colors = {'black': TAN, 'white': TURQUOISE}
    x = np.arange(len(races))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))

    for i, race in enumerate(races):
        c = colors[race]
        ax.bar(x[i] - width/2, summary.loc[race, 'genes_before'], width, color=c)
        ax.bar(x[i] + width/2, summary.loc[race, 'genes_after'], width, color=c, alpha=0.6)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels(['Black', 'White'])
    ax.set_ylabel('Number of genes')
    ax.set_title('Genes count before and after filtering')

    # neutral legend: opacity encodes filter status, color on bars encodes race
    legend_elements = [
        Patch(facecolor='grey', label='Before'),
        Patch(facecolor='grey', alpha=0.6, label='After'),
    ]
    ax.legend(handles=legend_elements, fontsize=8)

    for bars in ax.containers:
        ax.bar_label(bars, fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_file, dpi=150, bbox_inches='tight')
    plt.close()


def main() -> None:
    args = get_args()

    fig_path = Path(args.fig_path)
    fig_path.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.summary, sep='\t', index_col=0)
    plot_gene_counts(summary, fig_path / '1_gene_filter_barplot.png')


if __name__ == '__main__':
    main()