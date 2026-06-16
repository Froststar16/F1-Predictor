import pandas as pd
import os

df = pd.read_csv('data/processed/race_history.csv')
have_profiles = {f.replace('.yaml', '') for f in os.listdir('circuits') if f.endswith('.yaml') and not f.startswith('_')}
df = df[df['circuit_id'].isin(have_profiles)]
df.to_csv('data/processed/race_history.csv', index=False)
print(df['circuit_id'].value_counts())