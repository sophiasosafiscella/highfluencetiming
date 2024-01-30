import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import pypulse as pyp
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation, KMeans, MeanShift, OPTICS

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

#    pulses_data = np.load(fits_file)
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

#    np.save(fits_file[10:-6] + '_kmeans_clusters.npy', np.concatenate((org_features, y_pred.reshape(-1, 1)), axis=1))

    return org_features

def meanshift_classifier(org_features):

    print("Classifying using Mean Shift")

    features = StandardScaler().fit_transform(org_features)

    clustering = MeanShift(cluster_all=True, n_jobs=-1)  # set up the classifier
    clustering.fit(features)                              # perform the classification
    labels = clustering.labels_                          # labels of each point

    # Find the number of clusters
    labels_unique = np.unique(labels)
    n_clusters: int = len(labels_unique)

    # Save the data to a Pandas dataframe
    org_features['Cluster'] = labels.astype(int)
    org_features['Cluster'] = org_features['Cluster'].astype(str)

    return org_features, n_clusters

def OPTICS_classifier(org_features, max_eps):

    print("Classifying using OPTICS")

    features = StandardScaler().fit_transform(org_features)

    clustering = OPTICS(cluster_method='xi', max_eps=max_eps)
    clustering.fit(features)                              # perform the classification
    labels = clustering.labels_                          # labels of each point

    # Find the number of clusters
    labels_unique = np.unique(labels)
    n_clusters: int = len(labels_unique)

    # Save the data to a Pandas dataframe
    org_features['Cluster'] = labels.astype(int)
    org_features['Cluster'] = org_features['Cluster'].astype(str)

    return org_features, n_clusters


def clean_artifacts(cluster_average_pulse, clean_window, delta: int = 20):

    interpolating_bins = np.r_[clean_window[0] - delta: clean_window[0],
                         clean_window[1] + 1: clean_window[1] + 1 + delta]

    cluster_average_pulse[clean_window[0]:clean_window[1] + 1] = np.interp(
        x=np.arange(clean_window[0], clean_window[1] + 1, 1), xp=interpolating_bins,
        fp=cluster_average_pulse[interpolating_bins])

    return cluster_average_pulse

def time_clusters(n_clusters, results_dir_2, clustered_data, unnormalized_data, bin_to_musec, first_file,
                  plot_clusters=True):

    clusters_toas = pd.DataFrame(columns=['TOA', 'sigma_TOA', '1/sigma^2'], index=list(range(n_clusters)))

    ar = pyp.Archive(first_file, verbose=False)
    ar.fscrunch()
    ar.tscrunch()

    print(n_clusters)
    sys.exit()
    for cluster_index in range(n_clusters):

        # Isolate the single pulses in the cluster
        cluster_sp_times = clustered_data[clustered_data['Cluster'] == str(cluster_index)].index.to_numpy()
        cluster_pulses = unnormalized_data.loc[cluster_sp_times]

        # Calculate the cluster average pulse
        cluster_average_pulse = np.average(cluster_pulses.to_numpy(), axis=0)

        # Fix for weird artifacts
        cluster_average_pulse = clean_artifacts(cluster_average_pulse, [224, 227])
        print("Cluster average pulse:")
        print(cluster_average_pulse)

        # Copy to an Archive object
        ar.data = np.copy(cluster_average_pulse)

        # Create a template for this cluster by smoothing the cluster average pulse
        cluster_avg_sp = pyp.SinglePulse(cluster_average_pulse, opw=np.arange(0, 100))
        cluster_avg_sp.remove_baseline(save=True)
        smoothed_cluster = cluster_avg_sp.component_fitting()

        # If the cluster average pulse is all negative, then component_fitting will return a float
        # instead of an array. In that case, we assign this cluster a weight of zero because it's not useful.
        if isinstance(smoothed_cluster, float):
            clusters_toas.loc[cluster_index] = np.asarray([0.0, 0.0, 0.0])

        else:

            results_dir_3: str = results_dir_2 + "/cluster_" + str(cluster_index)

            # Make plots of some the single pulses in the cluster
            if plot_clusters and not os.path.isdir(results_dir_3):

                os.makedirs(results_dir_3)

                bins = np.arange(cluster_pulses.shape[1])
                plt.xlabel("Bins")
                plt.ylabel("Intensity")

                # Make plots of the first 10 single pulses in the cluster
                for n, time in enumerate(cluster_sp_times[0:10]):
                    plt.plot(bins, cluster_pulses.loc[time], color='#e94196')
                    plt.title("Pulse at t = " + str(time))
                    plt.tight_layout()
                    plt.savefig(results_dir_3 + "/" + str(n) + ".png")
                    plt.close()

                # Make a plot of the average pulse of the cluster
                plt.plot(bins, cluster_average_pulse, color='#e94196')
                plt.title("Integrated pulse profile for cluster " + str(cluster_index))
                plt.tight_layout()
                plt.savefig(results_dir_3 + "/average.png")
                plt.close()

            # Dump the template into a SinglePulse object
            smoothed_cluster_sp = pyp.SinglePulse(smoothed_cluster, opw=np.arange(0, 100))

            # Calculate the cluster average TOA and TOA error
            print("Smoothed cluster pulse:")
            print(smoothed_cluster)

            clusters_toas.loc[cluster_index, "TOA":"sigma_TOA"] = (
                    ar.fitPulses(smoothed_cluster_sp, nums=[1, 3]) * bin_to_musec)

            # Calculate the weight associated to each TOA error
            clusters_toas.loc[cluster_index, "1/sigma^2"] = (clusters_toas.loc[cluster_index, "sigma_TOA"]) ** (-2)

#                    print("- TOA from template matching procedure = " + str(clusters_toas.loc[cluster_index, "TOA[us]"]))
#                    print("- error on TOA = " + str(clusters_toas.loc[cluster_index, "sigma_TOA[us]"]))

    return clusters_toas


def plot_sigma_vs_k(results, results_dir_3, classifier):

    plt.close()
    plt.rcParams['text.usetex'] = True
    ax = plt.figure().gca()
    ax.set_xlabel("Number of Clusters")
    plt.ylabel(r'$\sigma_{\mathrm{TOA}}$ [\textmu s]')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.axhline(y=results.loc[1, "sigma_TOA"], color='r', linestyle='--', label="No Clustering")
    ax.plot(results.index.values[1:],  # All the row names, except the first one (corresponding to one cluster)
            results.iloc[1:, [1]],  # All the errors, except the first one (corresponding to one cluster)
            linestyle='-', marker='s')
    plt.legend()
#    plt.tight_layout()
#    plt.savefig(results_dir_3 + classifier + "_results.png")

    return

def AffinityPropagation_classifier(org_features):

    print("Classifying using Affinity Propagation Clustering")

    features = StandardScaler().fit_transform(org_features)

    aff_clustering = AffinityPropagation(random_state=5)
    aff_clustering.fit(features)

    labels = aff_clustering.labels_                          # labels of each point

    # Find the number of clusters
    labels_unique = np.unique(labels)
    n_clusters: int = len(labels_unique)

    # Save the data to a Pandas dataframe
    org_features['Cluster'] = labels.astype(int)
    org_features['Cluster'] = org_features['Cluster'].astype(str)

    return org_features, n_clusters

