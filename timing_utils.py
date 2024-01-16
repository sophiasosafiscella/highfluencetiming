import numpy as np
import pypulse as pyp
from statistics import harmonic_mean
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sns.set_theme(style="darkgrid")

def time_single_pulses(files, results_dir, k, cluster_index, cluster_pulses, smoothed_cluster_sp):

    # Load any observation
    ar = pyp.Archive(files[0], verbose=False)
    ar.fscrunch()
    ar.tscrunch()

    bin_to_musec = (ar.getPeriod() / ar.getNbin()) * 10 ** 6

    # Calculate the residual for each single pulse in the cluster_sp_times
    # using this smoothed pulse as the template
    # Keep in mind that the row label is the time in seconds

    cluster_sp_toas_file: str = results_dir + str(k) + "_clusters/cluster_" + str(cluster_index) + "/toas.npy"
    if len(glob.glob(cluster_sp_toas_file)) == 0:
        toas = np.empty((len(cluster_pulses.index), 3))  # The columns are time, TOA, and TOA error

        for i, sp_t in tqdm(enumerate(cluster_pulses.index)):
            ar.data = np.copy(cluster_pulses.loc[sp_t].to_numpy())  # Replace the data with the single pulse

            if abs(ar.fitPulses(smoothed_cluster_sp, nums=[1]) * bin_to_musec) < 20.0:
                toas[i, 0] = sp_t
                toas[i, 1:] = ar.fitPulses(smoothed_cluster_sp, nums=[1, 3]) * bin_to_musec
            else:
                toas[i, 0] = 0.0
                toas[i, 1:] = 0.0

        np.save(cluster_sp_toas_file, toas)

    else:
        toas = np.load(cluster_sp_toas_file)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.errorbar(toas[:, 0], toas[:, 1], yerr=toas[:, 2], fmt=".")
    ax.axhline(y=np.average(toas[:, 1]), c="C1", ls="--", label="Average")

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, "Average = " + str(round(np.average(toas[:, 1]), 2)) + " $\mu$s \n $\sigma = $" + str(
        round(np.std(toas[:, 1]), 2)) + "$\mu$s", transform=ax.transAxes, fontsize=14, verticalalignment='top',
            bbox=props)

    ax.set_xlabel("time [sec]")
    ax.set_ylabel("Residuals [$\mu$s]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir + str(k) + "_clusters/sp_residuals_" + str(k) + ".png")
    plt.show()


'''
Return weighted sample mean and std
http://en.wikipedia.org/wiki/Weighted_mean#Weighted_sample_variance
'''
def weighted_moments(series, weights, unbiased, harmonic=False):

    if len(series) == 1:
        return series[0], 1.0/np.sqrt(weights)

    series = np.array(series)
    weights = np.array(weights)
    weightsum = np.sum(weights)

#    weights = weights/weightsum

    weightedmean = np.sum(weights*series)/weightsum
    weightedvariance = np.sum(weights * np.power(series - weightedmean, 2))

    if harmonic:
#        return weightedmean, harmonic_mean(1.0/weights)
        return weightedmean, np.sqrt(1.0/np.sum(weights))  # 1/sigma_TOT^2 = np.sum(weights)

    elif unbiased:
        weightsquaredsum = np.sum(np.power(weights, 2))
        return weightedmean, np.sqrt(weightedvariance * weightsum / (weightsum**2 - weightsquaredsum))

    else:
        return weightedmean, np.sqrt(weightedvariance / weightsum)