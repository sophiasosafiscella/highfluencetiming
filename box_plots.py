import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

dir: str = "./results/pol_calibrated/820_band_meerguard_pazr/0.0_sigma/"
features = pd.read_pickle(dir + "features.pkl")

bin_edges = np.histogram_bin_edges(features['Amp'], bins=15)
features.loc[:, 'Amplitude'] = pd.cut(x=features['Amp'].to_numpy(), bins=bin_edges, precision=3)
features.dropna(axis=0, how='any', subset='Amplitude', inplace=True)   # Drop bin ranges that ar NaNs
features.sort_values('Amplitude', ascending=True, inplace=True)
features.loc[:, 'Amplitude'] = features.astype({'Amplitude':'str'})

features = features[features['Pos'] > 244]
features = features[features['Pos'] < 277]

features = features[features['Amplitude'] != '(-0.012, -0.00327]']
features = features[features['Amplitude'] != '(-0.00327, 0.00548]']
features = features[features['Amplitude'] != '(0.11, 0.119]']

fig = px.box(features, x="Pos", y="Amplitude")
fig.show()
fig.write_image("./plots/pos_amp_correlation.pdf")