import pandas as pd
import numpy as np
import ast
import os

out_dir = 'data/zenodo/processed'
os.makedirs(out_dir, exist_ok=True)

def parse_embed(s):
    return np.fromstring(s.strip('[]'), sep=',' if ',' in s else ' ')

# --- avg model features ---
df_user = pd.read_csv('data/zenodo/avg_user.tsv', sep='\t')
df_item = pd.read_csv('data/zenodo/avg_item.tsv', sep='\t')

# user_cf_features.npy  (shape: n_users x embed_dim)
user_embeds = np.stack(df_user['user_encoder.out_embed'].apply(parse_embed).values)
np.save(f'{out_dir}/user_cf_features.npy', user_embeds)

# item_audio_features.npy, item_lyrics_features.npy, item_cl_features.npy
for col, name in [
    ('item_encoder.audio.out_embed',  'audio'),
    ('item_encoder.lyrics.out_embed', 'lyrics'),
    ('item_encoder.cl.out_embed',     'cl'),
]:
    embeds = np.stack(df_item[col].apply(parse_embed).values)
    np.save(f'{out_dir}/item_{name}_features.npy', embeds)

print("Done. Shapes:")
print("  user_cf:", user_embeds.shape)
for name in ['audio', 'lyrics', 'cl']:
    print(f"  item_{name}:", np.load(f'{out_dir}/item_{name}_features.npy').shape)
