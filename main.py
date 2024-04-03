
# -------------------
# High Fluence Timing
# by Sophia Valentina Sosa Fiscella
# -------------------
import numpy as np
import pandas as pd
import glob
import pypulse as pyp
import matplotlib.pyplot as plt
from observations_utils import to_binary_and_calculate_rms, create_ds, merge
import sp_utils
import classification
from timing_utils import time_single_pulses, weighted_moments
from RFI_utils import remove_RFIs, meerguard
import os
import subprocess

# IMPORTANT: we're assuming that the observations has already been processed
#            with 512 phase bins and 128 single pulses per fits_file
# data shape = (128, 512) = 128 rows x 512 columns = 128 pulses x 512 bins

if __name__ == '__main__':

    #   0) Get the fits_file names
    band: str = "820_band"
    classifier: str = "DBSCAN"        # Options: "Kmeans", "OPTICS", "MeanShift", or "AffinityPropagation"
    results_dir: str = "./results/pol_calibrated/" + band + "_meerguard_pazr/"  # Directory with the results

#    pulses_dir: str = "./data/pol_calibrated/" + band + "/"
    pulses_dir: str = "/minish/svs00006/J2145_observations/" + band + "/folded/pol_calibrated/"

    if band == "L_band":
        files = sorted(glob.glob(pulses_dir + "GUPPI*calibP"))[:1714]  # Files containing the observations
    elif band == "820_band":
        files = sorted(glob.glob(pulses_dir + "GUPPI*calibP"))[:1693]

    low_res_file = glob.glob(pulses_dir + "low_res/low*pF*")[0]  # Low-resolution fits_file to create the dynamic spectrum
    template_file = glob.glob(pulses_dir +"*sm")[0]  # Files containing the template
    plot_clusters: bool = True  # Plot the single pulses in the cluster_sp_times
    time_sp: bool = False

    meerguard_ok: bool = True     # Clean using MeerGuard?
    clfd_ok: bool = False          # Clean using clfd?
    mask_RFI_ok: bool = False      # Clean using mask_RFI?
    zap_minmax_ok: bool = False    # Clean using zap_minmax?
    chisq_filter_ok: bool = False  # Clean using chisq_filter?
    opw_peaks_ok: bool = False     # Clean using opw_peaks?

    binary_out_dir: str = pulses_dir + "binary/"
#    bandpass = [64, 32]                             # How many channels we're removing from the upper and lower edges
    bandpass = None
    times_file: str = binary_out_dir + "times_data.npy"
    channels_file: str = binary_out_dir + "channels_data.npy"

    ds_data_file: str = results_dir + "dynamic_spectrum.pkl"
    windows_data_file: str = results_dir + "window_data.npy"
    sp_total_file: str = results_dir + "n_sp.npy"
    rms_data_file: str = results_dir + "rms.npy"
    basic_weights_file: str = results_dir + "basic_weights.npy"
    weights_file: str = results_dir + "weights.npy"

    #   1) Create the dynamic spectrum
    if len(glob.glob(ds_data_file)) == 0:
        print("Creating the dynamic spectrum...")
        ds = create_ds(template_file, low_res_file, band)
        ds.to_pickle(ds_data_file)
    else:
        ds = pd.read_pickle(ds_data_file)

    #   2) Create the pulse windows
    if len(glob.glob(windows_data_file)) == 0:
        print("Creating the pulse windows...")
        windows_data = sp_utils.find_windows(template_file=template_file, pulses_directory=pulses_dir,
                                             results_dir=results_dir, files=files,
                                             window_percentage=12.5, windows_factor=2.4, plot=True)
        np.save(windows_data_file, windows_data)
    else:
        windows_data = np.load(glob.glob(windows_data_file)[0])

    #   3) Count the number of single pulses
    if len(glob.glob(sp_total_file)) == 0:
        print("Counting single pulses...")
        sp_total, N_bin = sp_utils.count_sp(files)
        np.save(sp_total_file, np.asarray([sp_total, N_bin]))
    else:
        sp_total, N_bin = np.load(sp_total_file)

    #   3) Clean the observations using MeerGuard
    if meerguard_ok:

        # If we haven't cleaned using MeerGuard yet, do it
        if not os.path.isdir(pulses_dir + "cleaned/"):
            os.makedirs(pulses_dir + "cleaned/")

        if len(glob.glob(pulses_dir + "cleaned/*_cleaned.ar")) < len(files):
            files = meerguard(files, pulses_dir, band, template_file)
            subprocess.run("mv " + pulses_dir + "*cleaned.ar " + pulses_dir + "cleaned/", shell=True)

        # If we have cleaned already, load the cleaned files
        if band == "L_band":
            files = sorted(glob.glob(pulses_dir + "cleaned/*_cleaned.ar"))  # Files containing the observations
        elif band == "820_band":
            files = sorted(glob.glob(pulses_dir + "cleaned/*_cleaned.ar"))

    #   4) Convert the observations to binary and weight them according to the off-pulse noise RMS
    if len(glob.glob(binary_out_dir + "*J2145*npy")) < len(files) or len(glob.glob(basic_weights_file)) == 0:
        print("Converting the observation to binary files...")
        times_data, channels_data, rms_array, basic_weights = to_binary_and_calculate_rms(files, binary_out_dir, sp_total, bandpass)
        np.save(times_file, times_data)
        np.save(channels_file, channels_data)
        np.save(rms_data_file, rms_array)
        np.save(basic_weights_file, basic_weights)
    else:
        times_data = np.load(times_file)
        channels_data = np.load(channels_file)
        rms_array = np.load(rms_data_file)
        basic_weights = np.load(basic_weights_file)

    # Find the binary files
    binary_files = glob.glob(binary_out_dir + "*J2145*npy")

    #   6) Flags RFIs and create the weights
    if len(glob.glob(weights_file)) == 0:
        print("Removing RFIs...")
        weights = remove_RFIs(files, binary_files, windows_data, basic_weights, rms_array, template_file,
                              clfd_ok, mask_RFI_ok, zap_minmax_ok, chisq_filter_ok, opw_peaks_ok)
        np.save(weights_file, weights)
    else:
        weights = np.load(weights_file)

    # Inject different levels of noise
    for noise_factor in [0.0]:

        # Create a folder to dump the results of this amount of noise
        results_dir_2 = results_dir + str(noise_factor) + "_sigma/"
        if not os.path.isdir(results_dir_2):
            os.makedirs(results_dir_2)

        merged_file: str = results_dir_2 + "unnormalized_data.pkl"
        merged_normalized_file: str = results_dir_2 + "normalized_data.pkl"
        features_file: str = results_dir_2 + "features.pkl"

        #   7) merge and normalize the data
        if len(glob.glob(merged_normalized_file)) == 0:
            print("Merging and normalizing the data...")
            normalized_data, unnormalized_data = merge(ds=ds, binary_files=binary_files, times_data=times_data,
                                                            channels_data=channels_data, full_weights=weights,
                                                            N_bin=N_bin, sp_total=sp_total,
                                                            noise_rms=rms_array, noise_factor=noise_factor)
            normalized_data.to_pickle(merged_normalized_file)
            unnormalized_data.to_pickle(merged_file)

        else:
            normalized_data = pd.read_pickle(merged_normalized_file)
            unnormalized_data = pd.read_pickle(merged_file)

        # Load the template
        template = pyp.Archive(template_file)
#        template.bscrunch(factor=4)
        template_data = template.getData()

        # Dump the average pulse into an Archive object
        merged_average_pulse = np.average(unnormalized_data.to_numpy(), axis=0)

        plt.close()
        plt.title(str(noise_factor) + " $\sigma_{\mathrm{med}}$")
        plt.plot(merged_average_pulse, color="#e94196")
#        plt.savefig(results_dir_2 + "average_" + str(noise_factor) + ".png")
        
        ar = pyp.Archive(files[0], verbose=False)
        ar.fscrunch()
        ar.tscrunch()
        ar.data = np.copy(merged_average_pulse)

        # Conversion factor to go from pulsar phase bin units to microseconds
        bin_to_musec = (ar.getPeriod() / ar.getNbin()) * 1.0e6

        # Calculate the TOA and TOA error for the whole observation and save to the first row of the results dataframe
        non_clustered_res = np.asarray(ar.fitPulses(template_data, nums=[1, 3]) * bin_to_musec)

        #   8) Create the features for each single pulse
        if len(glob.glob(features_file)) == 0:
            print("Features not found. I'll create them...")
            org_features = sp_utils.get_params(merged_normalized_file, windows_data, results_dir=results_dir_2, plot=False)
            org_features.to_pickle(features_file)
        else:
            org_features = pd.read_pickle(features_file)

        # Create a folder to dump the results of this classifier
        results_dir_3 = results_dir_2 + classifier + "/"
        results_file: str = results_dir_3 + "results.pkl"
        if not os.path.isdir(results_dir_3):
            os.makedirs(results_dir_3)

        if classifier == "Kmeans":

            k_values = np.arange(1, 18, 1, dtype=int)  # Number of clusters for the classifier
            results = pd.DataFrame(index=k_values, columns=['TOA', 'sigma_TOA'])
            results.loc[1, 'TOA':'sigma_TOA'] = non_clustered_res

            #   Iterate over the values of k, except the first one because that one is using only one cluster
            for k in k_values[1:]:

                print("Analyzing k = " + str(k))

                #   8) Perform the classification
                clusters_file: str = results_dir_3 + str(k) + "_kmeans_clusters.pkl"
                if len(glob.glob(clusters_file)) == 0:
                    clustered_data = classification.kmeans_classifier(org_features, k=k, plot=False)
                    clustered_data.to_pickle(clusters_file)
                else:
                    clustered_data = pd.read_pickle(clusters_file)

                # Create a folder to dump the results of this value of k
                results_dir_4: str = results_dir_3 + str(k) + "_clusters"
                if not os.path.isdir(results_dir_4):
                    os.makedirs(results_dir_4)

                #   9) Calculate the TOAs and sigma_TOAs for the different clusters
                clusters_toas = classification.time_clusters(cluster_indexes=np.arange(k), results_dir=results_dir_4,
                                                             clustered_data=clustered_data,
                                                             unnormalized_data=unnormalized_data,
                                                             bin_to_musec=bin_to_musec, first_file=files[0],
                                                             plot_clusters=False)

                # Save the results for this number of cluster to an output fits_file
                k_clusters_results: str = results_dir_3 + str(k) + "_clusters/" + str(k) + "_clusters_results.plk"
                clusters_toas.to_pickle(k_clusters_results)


                # Save the results to the general results dataframe
                results.loc[k, 'TOA':'sigma_TOA'] = np.asarray(
                    weighted_moments(series=clusters_toas["TOA"].to_numpy(),
                                         weights=clusters_toas["1/sigma^2"].to_numpy(),
                                         unbiased=False, harmonic=True))

            # Save the final results
            results.to_pickle(results_file)

            # Make a plot of the results
            classification.plot_sigma_vs_k(results, results_dir_3, classifier)


        elif classifier == "MeanShift":

            #   8) Perform the classification
            clusters_file: str = results_dir_3 + "meanshift_features.pkl"
            n_clusters_file: str = results_dir_3 + "n_clusters.npy"
            if len(glob.glob(clusters_file)) == 0:
                clustered_data, n_clusters = classification.meanshift_classifier(org_features)
                clustered_data.to_pickle(clusters_file)
                np.save(n_clusters_file, n_clusters)
            else:
                clustered_data = pd.read_pickle(clusters_file)
                n_clusters = np.load(n_clusters_file)

            #   9) Calculate the TOAs and sigma_TOAs for the different clusters
            clusters_toas = classification.time_clusters(n_clusters, results_dir_3, clustered_data, unnormalized_data,
                                                         bin_to_musec, files[0])

            np.save(results_dir_3 + "results.npy",
                    np.stack((np.asarray(non_clustered_res),
                    np.asarray(weighted_moments(series=clusters_toas["TOA"].to_numpy(),
                                                weights=clusters_toas["1/sigma^2"].to_numpy(),
                                                unbiased=False, harmonic=True)))))


        elif classifier == "OPTICS":

            #  Iterate over the values of max_eps
#            max_eps_values = np.arange(start=0.0355, stop=0.0431, step=0.0001, dtype=float)
            max_eps_values = np.round(np.arange(start=0.08, stop=0.68, step=0.01, dtype=float), 2)

            for min_cluster_size in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
                results = pd.DataFrame(index=np.concatenate((np.asarray([0]), max_eps_values)), columns=['n_clusters', 'TOA', 'sigma_TOA'])

                results_dir_3 = results_dir_2 + classifier + "_min_cluster_size_" + str(min_cluster_size) + "_old/"
                if not os.path.isdir(results_dir_3):
                    os.makedirs(results_dir_3)

                # The first row is the results when no clustering is used
                results.loc[0, 'n_clusters'] = np.asarray(1)
                results.loc[0, 'TOA':'sigma_TOA'] = np.asarray(non_clustered_res)

                for max_eps in max_eps_values:

                    # Create a folder to dump the results of this value of max_eps
                    results_dir_4: str = results_dir_3 + str(max_eps) + "_maxeps"
                    if not os.path.isdir(results_dir_4):
                        os.makedirs(results_dir_4)

        #            8) Perform the classification
                    clustered_data, cluster_indexes = classification.OPTICS_classifier(org_features=org_features, max_eps=max_eps, min_cluster_size=min_cluster_size)

                    # Save the number of clusters
                    results.loc[max_eps, 'n_clusters'] = len(cluster_indexes)

                    # Save the features
                    clusters_file: str = results_dir_4 + "/" + str(max_eps) + "_OPTICS_features.pkl"
                    clustered_data.to_pickle(clusters_file)

                    #   9) Calculate the TOAs and sigma_TOAs for the different clusters
                    clusters_toas = classification.time_clusters(cluster_indexes, results_dir_4, clustered_data,
                                                                 unnormalized_data, bin_to_musec, files[0])

                    # Save the results for this number of cluster to an output file
                    max_eps_results: str = results_dir_3 + str(max_eps) + "_maxeps/" + str(max_eps) + "_maxeps_results.pkl"
                    clusters_toas.to_pickle(max_eps_results)

                    # Save the results to the general results dataframe
                    results.loc[max_eps, 'TOA':'sigma_TOA'] = np.asarray(weighted_moments(series=clusters_toas["TOA"].to_numpy(),
                                                                                    weights=clusters_toas[
                                                                                        "1/sigma^2"].to_numpy(),
                                                                                    unbiased=False, harmonic=True))

                # Save the final results
                results.to_pickle(results_dir_3 + "results.pkl")

        elif classifier == "DBSCAN":

            #  Iterate over the cluster size. A float between 0 and 1 indicates the fraction of the number of samples.
            eps_values = np.round(np.arange(start=0.51, stop=1.08, step=0.01, dtype=float), 2)

            for min_samples_fraction in [0.01]:
                min_samples: int = int(round(org_features.shape[0] * min_samples_fraction, 0))
                print(f'Processing min_samples={min_samples}    ')

#                eps_values = np.round(np.arange(start=0.28, stop=0.58, step=0.01, dtype=float), 2)
                results = pd.DataFrame(index=np.concatenate((np.asarray([0]), eps_values)),
                                       columns=['n_clusters', 'TOA', 'sigma_TOA'])

                # Create folder to save the results
                results_dir_3 = results_dir_2 + classifier + "_min_samples_" + str(min_samples) + "_old/"
                print(f"Results dir = {results_dir_3}")
                if not os.path.isdir(results_dir_3):
                    os.makedirs(results_dir_3)

                # The first row is the results when no clustering is used
                results.loc[0, 'n_clusters'] = np.asarray(1)
                results.loc[0, 'TOA':'sigma_TOA'] = np.asarray(non_clustered_res)

                for eps in eps_values:

                    print(f"Calculating eps={eps}")

                    # Create a folder to dump the results of this value of eps
                    results_dir_4: str = results_dir_3 + str(eps) + "_eps"
                    if not os.path.isdir(results_dir_4):
                        os.makedirs(results_dir_4)

                    # 8) Perform the classification
                    clusters_file: str = results_dir_4 + "/" + str(eps) + "_DBSCAN_features.pkl"
                    print(f"Clusters file = {clusters_file}")
                    if not os.path.isdir(clusters_file):
                        print("Clusters not created. I'll perform the classification...")
                        clustered_data = classification.DBSCAN_classifier(org_features=org_features,
                                                                                           eps=eps,
                                                                                           min_samples=min_samples)
                        # Save the features
                        clustered_data.to_pickle(clusters_file)

                    else:
                        print("Clusters already created")
                        clustered_data = pd.read_pickle(clusters_file)

                    # Save the number of clusters
                    results.loc[eps, 'n_clusters'] = len(clustered_data.Cluster.unique())

                    #   9) Calculate the TOAs and sigma_TOAs for the different clusters
                    eps_results: str = results_dir_3 + str(eps) + "_eps/" + str(eps) + "_eps_results.pkl"
                    if not os.path.isdir(eps_results):
                        clusters_toas = classification.time_clusters(cluster_indexes, results_dir_4, clustered_data,
                                                                 unnormalized_data, bin_to_musec, files[0])

                        # Save the results for this number of cluster to an output file
                        clusters_toas.to_pickle(eps_results)

                    else:
                        clusters_toas = pd.read_pickle(eps_results)


                    # Save the results to the general results dataframe
                    results.loc[eps, 'TOA':'sigma_TOA'] = np.asarray(
                        weighted_moments(series=clusters_toas["TOA"].to_numpy(),
                                         weights=clusters_toas[
                                             "1/sigma^2"].to_numpy(),
                                         unbiased=False, harmonic=True))

                # Save the final results
                results.to_pickle(results_dir_3 + "results.pkl")


        elif classifier == "AffinityPropagation":

            #   8) Perform the classification
            clusters_file: str = results_dir_3 + "AffinityPropagation_features.pkl"
            n_clusters_file: str = results_dir_3 + "n_clusters.npy"
            if len(glob.glob(clusters_file)) == 0:
                clustered_data, n_clusters = classification.AffinityPropagation_classifier(org_features)
                clustered_data.to_pickle(clusters_file)
                np.save(n_clusters_file, n_clusters)
            else:
                clustered_data = pd.read_pickle(clusters_file)
                n_clusters = np.load(n_clusters_file)

            #   9) Calculate the TOAs and sigma_TOAs for the different clusters
            clusters_toas = classification.time_clusters(n_clusters, results_dir_3, clustered_data, unnormalized_data,
                                                         bin_to_musec, files[0])

            np.save(results_dir_3 + "results.npy", np.stack((results.loc[1, 'TOA':'sigma_TOA'].to_numpy(), np.asarray(
                weighted_moments(series=clusters_toas["TOA"].to_numpy(), weights=clusters_toas["1/sigma^2"].to_numpy(),
                                 unbiased=False, harmonic=True)))))

