import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

# Columns to use in analysis
COLUMNS_TO_USE = [
    'ID',
    'ran_in_way_pipeline',
    'ClusterK2_kmeans',
    'ClusterK3_kmeans',
    'ClusterK4_kmeans',
    'ClusterK2_NMF',
    'ClusterK3_NMF',
    'ClusterK4_NMF',
    'ClusterK4_kmeans_TCGA_names',
    'external_HGSCsubtype_estimate',
]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Provide arguments for filtering expr')

    parser.add_argument(
        '-meta', 
        required=True, 
        type=str, 
        help='Path to metadata')
    parser.add_argument(
        '-expr', 
        required=True, 
        type=str, 
        help='Path to expr table')
    parser.add_argument(
        '-genes', 
        required=True, 
        type=str, 
        help='Path to file with gene list')
    parser.add_argument(
        '-out_path', 
        type=str, 
        default='out', 
        help='Folder to save data')
    parser.add_argument(
        '-fig_path', 
        type=str, 
        default='figs', 
        help='Folder to save figures')
    parser.add_argument(
        '-sample', 
        required=True, 
        type=str, 
        help='Sample name')
    parser.add_argument(
        '-gene_list_name', 
        required=True, 
        type=str, 
        help='Gene list name')

    return parser.parse_args()


def load_data(
        expr_path: str,
        genes_path: str,
        meta_path: str,
        ) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Load expression table, gene list, and metadata."""
    expr_df = pd.read_csv(expr_path, sep='\t', index_col=0)
    gene_list = pd.read_csv(genes_path).iloc[:, 0].dropna().unique()
    meta_df = pd.read_csv(meta_path, sep='\t', usecols=COLUMNS_TO_USE)
    meta_df['ID'] = 'Sample_' + meta_df['ID'].astype(str)
    return expr_df, gene_list, meta_df


def filter_expression(
        expr_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        gene_list: np.ndarray,
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filter to pipeline samples and to the provided gene list."""
    # keep only samples that passed the pipeline
    meta_df = meta_df[meta_df['ran_in_way_pipeline'] == True]
    common = meta_df['ID'][meta_df['ID'].isin(expr_df.columns)]

    # subset expression to those samples and align metadata to expr column order
    expr_samples = expr_df[common]
    meta_df = meta_df.set_index('ID').loc[expr_samples.columns]

    # subset to the provided gene list
    expr_filtered = expr_samples[expr_samples.index.isin(gene_list)]

    print(f'Samples: {expr_df.shape[1]} -> {expr_samples.shape[1]}')
    print(f'Genes:   {expr_samples.shape[0]} -> {expr_filtered.shape[0]}')
    return expr_samples, expr_filtered, meta_df


def plot_gene_counts(
        before: int,
        after: int,
        gene_list_name: str,
        fig_file: Path,
        ) -> None:
    """Bar plot of gene count before and after filtering."""
    bars = plt.bar(['Before', 'After'], [before, after], color=['#4C72B0', '#DD8452'])
    plt.bar_label(bars)
    plt.ylabel('Genes')
    plt.title(f'Genes before and after {gene_list_name} filtering')
    plt.savefig(fig_file, dpi=150, bbox_inches='tight')
    plt.close()


def main() -> None:
    args = get_args()

    out_path = Path(args.out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    fig_path = Path(args.fig_path)
    fig_path.mkdir(parents=True, exist_ok=True)

    expr_df, gene_list, meta_df = load_data(args.expr, args.genes, args.meta)
    expr_samples, expr_filtered, meta_df = filter_expression(expr_df, meta_df, gene_list)

    # save filtered, aligned outputs (gene names / sample IDs kept in the index)
    meta_df.to_csv(out_path / f'{args.sample}_filtered_metadata.tsv', sep='\t')
    expr_filtered.to_csv(out_path / f'{args.sample}_{args.gene_list_name}_filtered.tsv', sep='\t')

    plot_gene_counts(
        expr_samples.shape[0],
        expr_filtered.shape[0],
        args.gene_list_name,
        fig_path / f'{args.sample}_individuals_{args.gene_list_name}.png',
    )


if __name__ == '__main__':
    main()