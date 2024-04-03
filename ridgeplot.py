import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# Obtain the data
dir: str = "./results/pol_calibrated/820_band_meerguard_pazr/0.0_sigma/"
features = pd.read_pickle(dir + "features.pkl")

bin_edges = np.histogram_bin_edges(features['Amp'], bins=12)
features.loc[:, 'Amplitude'] = pd.cut(x=features['Amp'].to_numpy(), bins=bin_edges, precision=3)
features.dropna(axis=0, how='any', subset='Amplitude', inplace=True)   # Drop bin ranges that ar NaNs
features.sort_values('Amplitude', ascending=False, inplace=True)
features.loc[:, 'Amplitude'] = features.astype({'Amplitude':'str'})

features = features[features['Pos'] > 230]
features = features[features['Pos'] < 270]

features = features[features['Amplitude'] != '(-0.012, -0.00108]']
features = features[features['Amplitude'] != '(-0.00108, 0.00985]']
#features = features[features['Amplitude'] != '(0.11, 0.119]']

df = features[['Pos', 'Amplitude']]


# Initialize the FacetGrid object
pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
g = sns.FacetGrid(df, row="Amplitude", hue="Amplitude", aspect=15, height=.5, palette=pal)

# Draw the densities in a few steps
g.map(sns.kdeplot, "Pos",
      bw_adjust=.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, "Pos", clip_on=False, color="w", lw=2, bw_adjust=.5)

# passing color=None to refline() uses the hue mapping
g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)


# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)


g.map(label, "Pos")

# Set the subplots to overlap
g.figure.subplots_adjust(hspace=-.25)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)
plt.show()
