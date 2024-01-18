# -------------------
# High Fluence Timing
# -------------------
import numpy as np
import pandas as pd
import glob
import sys
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
    classifier: str = "Kmeans"        # Options: "Kmeans", "MeanShift", or "AffinityPropagation"
    results_dir: str = "./results/pol_calibrated/" + band + "_pazr/"  # Directory with the results
#    pulses_dir: str = "/minish/svs00006/J2145_observations/L-band/rficleaned/"
#    pulses_dir: str = "/minish/svs00006/J2145_observations/820-band/rficleaned/"
#    pulses_dir: str = "/minish/svs00006/J2145_observations/820-band/using_paz/"
#    pulses_dir: str = "./data/" + band + "/"
    pulses_dir: str = "/minish/svs00006/J2145_observations/820-band/pol_cal/using_pazr/"

    if band == "L_band":
        files = sorted(glob.glob(pulses_dir + "GUPPI*ar"))[:1714]  # Files containing the observations
    elif band == "820_band":
        files = sorted(glob.glob(pulses_dir + "GUPPI*ar"))[:1693]

    low_res_file = glob.glob(pulses_dir + "low_res/low*pF*")[0]  # Low-resolution fits_file to create the dynamic spectrum
    template_file = glob.glob(pulses_dir +"*sm")[0]  # Files containing the template
    plot_clusters: bool = True  # Plot the single pulses in the cluster_sp_times
    time_sp: bool = False

    meerguard_ok: bool = False     # Clean using MeerGuard?
    clfd_ok: bool = False          # Clean using clfd?
    mask_RFI_ok: bool = False      # Clean using mask_RFI?
    zap_minmax_ok: bool = False    # Clean using zap_minmax?
    chisq_filter_ok: bool = False   # Clean using chisq_filter?
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

    k_values = np.arange(1, 18, 1, dtype=int)  # Number of clusters for the classifier
    results = pd.DataFrame(index=k_values, columns=['TOA', 'sigma_TOA'])

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
    if len(glob.glob(binary_out_dir + "*J2145*npy")) < len(files) and len(glob.glob(basic_weights_file)) == 0:
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
        results_file: str = results_dir_2 + "results.pkl"

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
        plt.savefig(results_dir_2 + "average_" + str(noise_factor) + ".png")
        
        ar = pyp.Archive(files[0], verbose=False)
        ar.fscrunch()
        ar.tscrunch()
        ar.data = np.copy(merged_average_pulse)

        # Conversion factor to go from pulsar phase bin units to microseconds
        bin_to_musec = (ar.getPeriod() / ar.getNbin()) * 1.0e6

        # Calculate the TOA and TOA error for the whole observation and save to the first row of the results dataframe
        results.loc[1, 'TOA':'sigma_TOA'] = np.asarray(ar.fitPulses(template_data, nums=[1, 3]) * bin_to_musec)

        #   8) Create the features for each single pulse
        if len(glob.glob(features_file)) == 0:
            print("Features not found. I'll create them...")
            org_features = sp_utils.get_params(merged_normalized_file, windows_data, results_dir=results_dir_2, plot=False)
            org_features.to_pickle(features_file)
        else:
            org_features = pd.read_pickle(features_file)

        # Create a folder to dump the results of this classifier
        results_dir_3 = results_dir_2 + classifier + "/"
        if not os.path.isdir(results_dir_3):
            os.makedirs(results_dir_3)

        if classifier == "Kmeans":

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
                if not os.path.isdir(results_dir_3 + str(k) + "_clusters"):
                    os.makedirs(results_dir_3 + str(k) + "_clusters")

                #   9) Calculate the TOAs and sigma_TOAs for the different clusters
                clusters_toas = classification.time_clusters(k, results_dir_3 + str(k) + "_clusters", clustered_data,
                                                                 unnormalized_data, bin_to_musec, files[0])

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
                    np.stack((results.loc[1, 'TOA':'sigma_TOA'].to_numpy(),
                    np.asarray(weighted_moments(series=clusters_toas["TOA"].to_numpy(),
                                                weights=clusters_toas["1/sigma^2"].to_numpy(),
                                                unbiased=False, harmonic=True)))))

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

        elif classifier == "OPTICS":

            #   8) Perform the classification
            clusters_file: str = results_dir_3 + "OPTICS.pkl"
            n_clusters_file: str = results_dir_3 + "n_clusters.npy"
            if len(glob.glob(clusters_file)) == 0:
                clustered_data, n_clusters = classification.OPTICS_classifier(org_features)
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
