import argparse
import numpy as np
import pandas as pd

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
        ) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """Filter to pipeline samples and to the provided gene list."""
    # keep only samples that passed the pipeline filter
    meta_df = meta_df[meta_df['ran_in_way_pipeline'] == True]
    common = meta_df['ID'][meta_df['ID'].isin(expr_df.columns)]

    # subset expression to those samples and align metadata to expr column order
    expr_samples = expr_df[common]
    meta_df = meta_df.set_index('ID').loc[expr_samples.columns]

    # subset to the provided gene list
    expr_filtered = expr_samples[expr_samples.index.isin(gene_list)]

    genes_before = expr_samples.shape[0]
    genes_after = expr_filtered.shape[0]

    print(f'Genes: {genes_before} -> {genes_after}')
    return expr_filtered, meta_df, genes_before, genes_after


def write_summary(
        genes_before: int, 
        genes_after: int, 
        sample: str, 
        summary_file: Path
        ) -> None:
    """Append a row to a shared gene-count summary."""
    row = pd.DataFrame(
        {'genes_before': [genes_before], 'genes_after': [genes_after]},
        index=[sample],
    )
    # append if file exists, else create with header
    if summary_file.exists():
        existing = pd.read_csv(summary_file, sep='\t', index_col=0)
        out = pd.concat([existing, row])
        out = out[~out.index.duplicated(keep='last')]  # rerun of same sample overwrites its row
    else:
        out = row
    out.to_csv(summary_file, sep='\t')


def main() -> None:
    args = get_args()

    # make sure paths exists
    out_path = Path(args.out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    expr_df, gene_list, meta_df = load_data(args.expr, args.genes, args.meta)
    expr_filtered, meta_df, genes_before, genes_after = filter_expression(expr_df, meta_df, gene_list)

    write_summary(genes_before, genes_after, args.sample, out_path / 'filter_summary.tsv')

    # save filtered, aligned outputs
    meta_df.to_csv(out_path / f'{args.sample}_filtered_metadata.tsv', sep='\t')
    expr_filtered.to_csv(out_path / f'{args.sample}_{args.gene_list_name}_filtered.tsv', sep='\t')


if __name__ == '__main__':
    main()