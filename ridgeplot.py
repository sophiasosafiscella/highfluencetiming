import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.express as px
import ast
import sys
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# Obtain the data
dir: str = "./results/pol_calibrated/820_band_meerguard_pazr/0.0_sigma/"
features = pd.read_pickle(dir + "features.pkl")
features = features.rename({'Pos': 'Phase Bins', 'b': 'Y'}, axis='columns')

bin_edges = np.histogram_bin_edges(features['Amp'], bins=12)
features.loc[:, 'Amplitude'] = pd.cut(x=features['Amp'].to_numpy(), bins=bin_edges, precision=1)
features.dropna(axis=0, how='any', subset='Amplitude', inplace=True)   # Drop bin ranges that ar NaNs
features.sort_values('Amplitude', ascending=False, inplace=True)
features.loc[:, 'Amplitude'] = features.astype({'Amplitude':'str'})

features = features[features['Phase Bins'] > 240]
features = features[features['Phase Bins'] < 265]

# Calculate the mean position for each interval
means = [np.median(features[features['Amplitude'] == interval]['Phase Bins'].to_numpy()) for interval in features['Amplitude'].unique()]
print(features['Amplitude'].unique())
print(means)


features = features[features['Amplitude'] != '(-0.012, -0.0011]']
features = features[features['Amplitude'] != '(-0.0011, 0.0099]']
#features = features[features['Amplitude'] != '(0.11, 0.119]']

df = features[['Phase Bins', 'Amplitude']]


# Initialize the FacetGrid object
#pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
#pal = [px.colors.qualitative.Plotly[i] for i in [3, 0, 5, 2, 9, 4, 1, 8, 7, 6, 5, 4, 3, 2, 1]]

scientific_publication_rainbow_colors = [
    "#FF6666",  # Light Red
#    "#FFC0CB",  # Light Pink
    "#FFCCCC",   # Light Coral
    "#FFB266",  # Light Orange
    "#FFE066",  # Light Yellow
    "#9AC46C",  # Darker Green
    "#B6DF9C",  # Light Green
    "#AED9E0",  # Light Cyan
    "#99CCFF",  # Light Blue
    "#E0B0FF",  # Light Lavender
    "#B0A3E6"  # Light Indigo
]

pal = scientific_publication_rainbow_colors
pal2 = [pal[i] for i in [9, 8, 7, 5, 6, 4, 3, 2, 0, 1]]

grid = sns.FacetGrid(df, row="Amplitude", hue="Amplitude", aspect=15, height=.5, palette=pal)

# Draw the densities in a few steps
grid.map(sns.kdeplot, "Phase Bins",
         bw_adjust=.5, clip_on=False,
         hue=features['Amplitude'], palette=pal2,
         fill=True, alpha=1, linewidth=1.5)
grid.map(sns.kdeplot, "Phase Bins", clip_on=False, color="w", lw=2, bw_adjust=.5)

# passing color=None to refline() uses the hue mapping
#grid.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

for i, ax in enumerate(grid.axes.flat):

    ax.axhline(y=0, color=pal[i], linewidth=2, linestyle="-", clip_on=False)
    ax.axvline(x=means[i], ymax=0.5, color='r', linewidth=2, linestyle=":", clip_on=False)


# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()

    table = str.maketrans({'[': '(', ']': ')'})
    left, right = np.round(ast.literal_eval(label.translate(table)),2)
    left = "%.2f" % left
    right = "%.2f" % right

    ax.text(0, .2, f"({left}, {right}]", fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)


grid.map(label, "Phase Bins")

# Set the subplots to overlap
grid.figure.subplots_adjust(hspace=-.45)

# Remove axes details that don't play well with overlap
grid.set_titles("")
grid.set(yticks=[], ylabel="")
grid.despine(bottom=True, left=True)
plt.savefig("./plots/ridgeplot.pdf", bbox_inches="tight")
plt.show()
