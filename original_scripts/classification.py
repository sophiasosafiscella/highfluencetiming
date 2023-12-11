import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]

    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=35, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=2, linewidths=12,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, resolution=50, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    print("plotting!")
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy, ww, zz = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution),
                         np.linspace(mins[2], maxs[2], resolution),
                         np.linspace(mins[3], maxs[3], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel(), ww.ravel(), zz.ravel()])
    Z = Z.reshape(xx.shape)
    print(np.shape(Z))

    plt.contourf(Z[:, :, 0, 0], extent=(mins[0], maxs[0], mins[1], maxs[1]),
                 cmap="Pastel2")
    plt.contour(Z[:, :, 0, 0], extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)

    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("Pos", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("Width", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)

def DBSCAN_cluster(file, windows, org_features, r, samples):

    features = StandardScaler().fit_transform(org_features)

#    window_left = windows[0, 0]
#    window_right = windows[0, 1]

#    pulses_data = np.load(file)
#    window_data = pulses_data[:, window_left:window_right]

#    avg_wind_data = np.mean(window_data, axis=0)
#    avg_pulse_data = np.mean(pulses_data, axis=0)

    for radius in r:
        print('Clustering....\n')
        kmeans = DBSCAN(eps=radius, min_samples=samples).fit(features)
        labels = kmeans.labels_
        print('Successful....\n')
        print("Unique labels = " + str(np.unique(labels).size))

        dir = "./clusters_toas/" + str(radius, ) + '_' + str(samples) + '/'
        if not os.path.isdir(dir):
            os.mkdir(dir)

        np.save(dir + file[10:-6] + '_clusters.npy', np.concatenate((org_features, labels.reshape(-1, 1)), axis=1))

#        for i, j in enumerate(np.unique(labels)):
#            print(i, j)
#            sys.exit()
#            cluster_labels = labels[np.where(labels == j)]
#            avg_wind_data = np.vstack((avg_wind_data, np.mean(window_data[np.where(labels == j)], axis=0)))
#            avg_pulse_data = np.vstack((avg_pulse_data, np.mean(window_data[np.where(labels == j)], axis=0)))


#            if j != 0 and j != -1:
#                cluster_data = np.concatenate((window_data[np.where(labels == j)], cluster_labels.reshape(-1, 1)), axis=1)

#                if j == 1:
#                    data = cluster_data
#                else:
#                    data = np.concatenate((data, cluster_data), axis=0)

#        np.save(dir + 'sequences', data)
#        np.save(dir + 'avg', avg_wind_data)
#        np.save(dir + 'avg_whole', avg_pulse_data)


def kmeans_classifier(org_features, k, plot=False):

    print("Classifying using K-Means")

    features = StandardScaler().fit_transform(org_features)

    # the number of clusters has to be set beforehand
    # Nevertheless is quite unstable, and depends on the random seed.
    # To surpass this, the algorithm runs on different random seeds (set by n_init),
    # and uses the best final value, where best is measured as the inertia:
    # The mean squared distance of each instance to its closest centroid.
    kmeans = KMeans(n_clusters=k, n_init=3, random_state=42)
    kmeans.fit(features)

    # The labels assigned to each training instance is stored in the attribute label_
#    print(kmeans.labels_)

    # And assignation to labels of new samples is done through the predict method
    y_pred = kmeans.predict(features)
    (y_pred == kmeans.labels_).all()
    print("Classification ready!")

    # Let's plot the regions, as well as the centroids:
    if plot:
        plt.figure(figsize=(8, 4))
        plot_decision_boundaries(kmeans, features)
        plt.show()

    # Save the data to a Pandas dataframe
    org_features['Cluster'] = y_pred.astype(int)
    org_features['Cluster'] = org_features['Cluster'].astype(str)

#    np.save(file[10:-6] + '_kmeans_clusters.npy', np.concatenate((org_features, y_pred.reshape(-1, 1)), axis=1))

    return org_features

