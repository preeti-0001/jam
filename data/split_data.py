import pandas as pd

mode = 'avg'  # or 'cross'

df_query = pd.read_csv(f'data/zenodo/{mode}_query.tsv', sep='\t')
df_user = pd.read_csv(f'data/zenodo/{mode}_user.tsv', sep='\t')
df_item = pd.read_csv(f'data/zenodo/{mode}_item.tsv', sep='\t')

import numpy as np

def parse_ids(s):
    return list(map(int, s.strip('[]').split()))

df_query['item_idxs'] = df_query['item_id'].apply(parse_ids)

out_dir = 'data/zenodo/processed'
import os; os.makedirs(out_dir, exist_ok=True)

for split in ['train', 'val', 'test']:
    subset = df_query[df_query['set'] == split][['query_id', 'aug_query', 'user_id', 'item_idxs']].copy()
    subset.columns = ['query_idx', 'text', 'user_idx', 'item_idxs']
    subset.to_csv(f'{out_dir}/{split}_split.tsv', sep='\t', index=False)

df_user[['user_id']].rename(columns={'user_id': 'user_idx'}).reset_index().to_csv(f'{out_dir}/user_idxs.tsv', sep='\t', index=False)
df_item[['item_id']].rename(columns={'item_id': 'item_idx'}).reset_index().to_csv(f'{out_dir}/item_idxs.tsv', sep='\t', index=False)

print("Done")
