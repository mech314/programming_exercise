import argparse
import pandas as pd

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Provide gene lists')

    parser.add_argument('-list1', required=True, type=str, help='List 1')
    parser.add_argument('-list2', required=True, type=str, help='List 2')

    return parser.parse_args()


def main():

    args = get_args()

    list1_df = pd.read_csv("data/supp_table_1_GlobalMAD_genelist.csv").iloc[:, 0]
    list2_df = pd.read_csv("data/supp_table_2_CommonGenes_genelist.csv").iloc[:, 0]

    shared = set(list1_df) & set(list2_df)

    print(f'Shared genes: {len(shared)}')

if __name__ == "__main__":
    main()