import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

dir: str = "./results/820_band/0.0_sigma/"
n_clusters = np.load(dir + 'MeanShift/n_clusters.npy', allow_pickle=True)
results = np.load(dir + 'MeanShift/results.npy', allow_pickle=True)

results = pd.read_pickle(dir + 'MeanShift/meanshift_clusters.pkl')
#results = pd.read_pickle(dir + 'Kmeans/11_kmeans_clusters.pkl')

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11']

# Create a figure and a 3D Axes
fig = plt.figure()
#ax = Axes3D(fig)
ax = fig.add_subplot(111,projection='3d')

# Create an init function and the animate functions.
# Both are explained in the tutorial. Since we are changing
# the the elevation and azimuth and no objects are really
# changed on the plot we don't have to return anything from
# the init and animate function. (return value is explained
# in the tutorial.
def init():
#    ax.scatter(results['Width'], results['Energy'], results['Amp], marker='o', s=20, c="goldenrod", alpha=0.6)
    for i in range(len(results)):
        x, y, z = results.iloc[i]['Width'], results.iloc[i]['Energy'], results.iloc[i]['Amp']
        ax.scatter(x, y, z, c=colors[int(results.iloc[i]['Cluster'])])
    return fig,

def animate(i):
    ax.view_init(elev=10., azim=i)
    return fig,

# Animate
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=360, interval=20, blit=True)
# Save
anim.save('meanshift.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
