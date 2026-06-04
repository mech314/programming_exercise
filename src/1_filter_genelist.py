import argparse
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
    'external_HGSCsubtype_estimate'
    ]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Provide arguments for filtering expr')

    parser.add_argument('-meta', required=True, type=str, help='Path to metadata')
    parser.add_argument('-expr', required=True, type=str, help='Path to expr table')
    parser.add_argument('-genes', required=True, type=str, help='Path to file with gene list')
    parser.add_argument('-out_path', type=str, default='out', help='Folder to save data')
    parser.add_argument('-fig_path', type=str, default='figs', help='Folder to save data')
    parser.add_argument('-sample', required=True, type=str, help='Sample name')
    parser.add_argument('-gene_list_name', required=True, type=str, help='Gene list name')

    return parser.parse_args()


def main():

    # get arguments
    args = get_args()

    # make sure out folders exist
    out_path = Path(args.out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    fig_path = Path(args.fig_path)
    fig_path.mkdir(parents=True, exist_ok=True)

    # loading data and gene list
    expr_df = pd.read_csv(args.expr, sep='\t', header=0, index_col=0)
    genes_df = pd.read_csv(args.genes, sep=',', header=0)

    # load metadata, take only required columns and format ID to match expression table format
    meta_df = pd.read_csv(args.meta, sep='\t', header=0, usecols=COLUMNS_TO_USE)
    meta_df['ID'] = 'Sample_' + meta_df['ID'].astype(str)

    # first filter expression to the same sample in metadata
    print('Filtering samples by "Ran_in_way_pipeline" flag in metadata')

    meta_df = meta_df[meta_df['ran_in_way_pipeline'] == True]
    common = meta_df["ID"][meta_df['ID'].isin(expr_df.columns)]

    print(f'Saving filtered metadata...')
    meta_df.to_csv(
        f'{args.out_path}/{args.sample}_filtered_metadata.tsv', 
        sep='\t', 
        header=True, 
        index=False)

    # filter expr table to the samples in metadata
    expr_meta_df = expr_df[common]
    meta_df = meta_df.set_index('ID').loc[expr_meta_df.columns]
    print(f'Filtered out {len(expr_df.columns) - len(expr_meta_df.columns)} samples')

    # filter expr table to the gene list provided
    gene_list = genes_df.iloc[:, 0].dropna().unique() # need gene list as series
    expr_meta_df_filtered = expr_meta_df[expr_meta_df.index.isin(gene_list)]
    print(f'Total Number of filtered genes: {len(expr_meta_df) - len(expr_meta_df_filtered)}')

    print(f'Saving filtered expression table...')
    expr_meta_df_filtered.to_csv(
        f'{args.out_path}/{args.sample}_{args.gene_list_name}_filtered.tsv', 
        sep='\t', 
        header=True, 
        index=True)

    vals = [expr_meta_df.shape[0], expr_meta_df_filtered.shape[0]]
    bars = plt.bar(["Before", "After"], vals, color=["#4C72B0", "#DD8452"])
    plt.bar_label(bars)
    plt.ylabel("Genes")
    plt.title("Genes before and after GlobalMAD filtering")

    plt.savefig(
        f'{args.fig_path}/{args.sample}_individuals_{args.gene_list_name}.png', 
        dpi=150,
        bbox_inches='tight')

if __name__ == '__main__':
    main()