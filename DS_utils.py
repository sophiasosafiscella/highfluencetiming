import matplotlib.pyplot as plt
import pypulse as pyp
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

from RFI_utils import zap_minmax, mask_RFI, chisq_filter, opw_peaks


def create_ds(template, low_res_file, band):
    # Load the low-res file
    low_res_object = pyp.Archive(low_res_file, verbose=False)

    # Load the template
    template_object = pyp.Archive(template, verbose=False)
    if template_object.getNbin() != low_res_object.getNbin():
        template_object.bscrunch(factor=int(template_object.getNbin() / low_res_object.getNbin()))

    # tscrunch the low resolution file
    if band == "L_band":
        low_res_object.tscrunch(factor=8)
    elif band == "820_band":
        low_res_object.tscrunch(factor=4)

    # We'll use these to create the lower edges of the frequency bins
    nchan = low_res_object.getNchan()
    fc = low_res_object.getCenterFrequency()
    bw = low_res_object.getBandwidth()
    df = np.abs(bw) / nchan

    ds = pd.DataFrame(np.absolute(np.asarray(low_res_object.fitPulses(template_object.getData(), nums=[2]))[0].T),
                      index=np.array((np.arange(nchan + 1) - (nchan + 1) / 2.0 + 0.5) * df + fc)[1:],
                      # By using edges=True we not get the centers for each subintegration/channel
                      columns=low_res_object.getAxis(flag="T", edges=True)[
                              1:])  # but rather return the edges, including 0 and the upper edge of the last bin.
    ds = ds.reindex(sorted(ds.columns), axis=1)

    return ds


def old_create_ds(template, files):
    # Load the template
    template_object = pyp.Archive(template, verbose=False)
    if template_object.getNbin() != 512:
        template_object.bscrunch(factor=4)
    template_data = template_object.getData()

    n_sp, n_chan, n_bins = pyp.Archive(files[0], verbose=False).shape()  # Number of single pulses per file
    #    files_per_group = ceil((60.0/ template_object.getPeriod()) / n_sp)  # Number of files per group
    #    number_groups = ceil(len(files) / files_per_group)

    # Allocate space for the array containing the dynamic spectrum
    scale_factor = np.zeros((n_chan, len(files)))

    # Allocate space for the vector containing the times
    times = np.zeros(len(files))
    time: float = 0.0

    # Loop over the PSRFITS files
    #    for group_n in range(number_groups):
    for n, file in tqdm(enumerate(files)):
        #        group_pulses = np.zeros((n_chan, files_per_group, n_bins))
        #        group_weights = np.zeros((n_chan, files_per_group))

        #        for n, file in enumerate(files[group_n: group_n + files_per_group]):
        ar = pyp.Archive(file, verbose=False)  # Open the file using PyPulse

        #       if n == files_per_group-1:
        time += ar.getDuration()  # take the time at the end of the observation
        times[n] = time  # and add it to the array of times

        # Get rid of frequency channels at fixed time with all zeros
        weights = ar.getWeights()  # Extract the weights
        mask = (ar.getData() == 0.0).all(axis=2)  # Create a mask where True means a single pulse that is al zeros
        weights[mask] = 0.0  # Assign all null single pulses a weight of zero
        ar.setWeights(weights)  # Replace the old weights with the new filtered weights

        # Average in time -> I(ν, ϕ)
        ar.tscrunch()

        # Add this time-averaged, 32-channels, single pulse, to our matrix
        #            group_pulses[:, group_n, :] = ar.getData()
        #            group_weights[:, group_n] = ar.getWeights()

        #  Create the dynamic spectrum. nums=[2] means scale factor (bhat) https://mtlam.github.io/PyPulse/singlepulse.html#fitPulse
        scale_factor[:, n] = np.asarray(ar.fitPulses(template_data, nums=[2]))[0]

    # Save the dynamic spectrum
    ds = pd.DataFrame(scale_factor, index=ar.getAxis(flag="F", edges=True), columns=times)
    ds = ds.reindex(sorted(ds.columns), axis=1)

    return ds


def merge_and_normalize(ds, files, window_data, sp_total, noise_rms):
    first_file = pyp.Archive(files[0], verbose=False)  # let's assume n_bins = 512
    total_times = np.zeros(sp_total)
    unnormalized_data = np.zeros((sp_total, first_file.getNbin()))
    normalized_data = np.zeros((sp_total, first_file.getNbin()))  # rows: single pulses, columns: phase bins

    ds_channels = np.asarray(list(ds.index))  # Upper edges of the frequency bins
    ds_times = np.asarray(ds.columns.values.tolist())  # Upper edges of the time bins
    ds_arr = ds.to_numpy()

    # get the off-pulse window
    offpulsewindow = np.linspace(window_data[0, 0], window_data[0, 1],
                                 num=window_data[0, 1] - window_data[0, 0] + 1).astype(int)

    time: float = 0.0
    last_index: int = 0

    # Iterate over the files
    for k, file in tqdm(enumerate(files)):

        ar = pyp.Archive(file, prepare=True, verbose=False)
        new_index = ar.getNsubint() + last_index

        data_channels = ar.getAxis(flag="F", edges=True)  # Frequency channels
        data_times = ar.getTimes() + time  # We add the time at the end of the previous observation

        total_times[last_index:new_index] = data_times

        # When we fscrunch, a small offset is introduced in the upper edges of the frequency channels, so we correct it
        #        ds_channels += abs(data_channels[0] - ds_channels[0])

        # Remove RFI
        mask_RFI(ar, window_data)  # Account for individual RFIs and null single pulses
        zap_minmax(ar)  # Zap noisy frequency channels
        #        chisq_filter(ar, template_file=template_file)   # Filter RFIs by the chisq from fitting the SPs to the template
        #        opw_peaks(ar, window_data)                # Filter single pulses with sharp peaks in the off-window region

        # Extract the sp_data form the observation
        sp_data = ar.getData()  # (128 single pulses x 128 frequency channels x 2048 phase bins) -> in this order
        sp_data_normalized = np.empty_like(sp_data)
        weights = ar.getWeights()

        # Iterate over the single pulses
        for i, t in enumerate(data_times):

            # since we have the upper edges of the time bins, we look for all the DS edges that are bigger than the
            # given value of t by doing ds_times[ds_times >= t] and we keep the smaller one using np.amin()
            sp_index = np.argwhere(ds_times == np.amin(ds_times[ds_times >= t]))[0][0]  # time index

            # If the weights across all frequency channels are zero, we drop this sub-integration by marking the
            # frequency-averaged single pulse as all NaNs
            if np.all(ar.getWeights()[i, :].astype(int) == 0):

                normalized_data[last_index + i, :] = np.nan
                unnormalized_data[last_index + i, :] = np.nan
                print("One sub-integration was dropped...")

            # Otherwise, average in frequency
            else:

                # Iterate over the frequency channels
                for j, freq in enumerate(data_channels):

                    chan_index = np.argwhere(ds_channels == np.amin(ds_channels[ds_channels >= freq]))[0][
                        0]  # frequency index

                    # Assign a weight equal to 0 to all the null single pulses
                    if int(weights[i, j]) != 0:
                        ar.setWeights(val=1.0 / (np.std(sp_data[i, j, offpulsewindow]) ** 2), t=i, f=j)

                    # Add noise
                    sp_data[i, j, :] += np.random.normal(0, noise_rms, ar.getNbin())

                    # Divide by the dynamic spectrum
                    sp_data_normalized[i, j, :] = np.divide(sp_data[i, j, :],
                                                 ds_arr[chan_index, sp_index])  # Divide the single pulse by the DS

                # Now we need to average in frequency
                normalized_data[last_index + i, :] = np.average(sp_data_normalized[i, :, :], axis=0,
                                                                weights=ar.getWeights()[i, :])
                unnormalized_data[last_index + i, :] = np.average(sp_data[i, :, :], axis=0,
                                                                weights=ar.getWeights()[i, :])

            #            # Make sure that no NaN were created
            #            if np.isnan(normalized_data[last_index + i, :]).any():
            #                print("NaN created at k= " + str(k))
            #                print("i = " + str(i))
            #                print(file)
            #                print(normalized_data[last_index + i, :])
            #                sys.exit()

        time += ar.getDuration()
        last_index = new_index

    # Before returning the dataframe, we make sure to remove any rows with NaNs
    normalized_df = pd.DataFrame(data=normalized_data, index=total_times, columns=np.arange(ar.getNbin())).dropna(axis=0,
                                                                                                         how="any")
    unnormalized_df = pd.DataFrame(data=unnormalized_data, index=total_times, columns=np.arange(ar.getNbin())).dropna(axis=0,
                                                                                                         how="any")
    return normalized_df, unnormalized_df


def merge_no_filter(files):
    first_file = pyp.Archive(files[0], verbose=False)  # let's assume n_bins = 512
    integrated_pulses = np.zeros((len(files), first_file.getNbin()))  # rows: single pulses, columns: phase bins

    # Iterate over the files
    for k, file in tqdm(enumerate(files)):
        ar = pyp.Archive(file, prepare=True, verbose=False)

        ar.pscrunch()
        ar.fscrunch()
        ar.tscrunch()
        integrated_pulses[k, :] = ar.getData()

    # Before returning the dataframe, we make sure to remove any rows with NaNs
    return np.average(integrated_pulses, axis=0)


def merge(files, factor=4):
    time: float = 0.0
    first_file = pyp.Archive(files[0])
    first_file.bscrunch(factor=factor)
    n_sp = len(first_file.getTimes())
    n_bins = first_file.getNbin()
    total_times = np.zeros(len(files) * n_sp)

    # rows: single pulses, columns: phase bins
    total_data = np.zeros((len(files) * n_sp, first_file.getNbin()))

    # Iterate over the files
    for k, file in enumerate(files):
        ar = pyp.Archive(file)
        ar.bscrunch(factor=factor)
        ar.fscrunch()
        final_time = ar.getDuration()

        data_times = ar.getTimes() + time  # We add the time at the end of the previous observation
        total_times[k * n_sp: (k + 1) * n_sp] = data_times  # Array with the times of all the single pulses

        total_data[k * n_sp: (k + 1) * n_sp, :] = ar.getData()  # (128 single pulses x 512 phase bins) -> in this order

        time += final_time

    return pd.DataFrame(data=total_data, index=total_times, columns=np.arange(n_bins))


def new_merge(files, window_data, noise_rms, sp_total):
    first_file = pyp.Archive(files[0], verbose=False)  # let's assume n_bins = 512
    total_times = np.zeros(sp_total)
    total_data = np.zeros((sp_total, first_file.getNbin()))  # rows: single pulses, columns: phase bins

    # get the off-pulse window
    offpulsewindow = np.linspace(window_data[0, 0], window_data[0, 1],
                                 num=window_data[0, 1] - window_data[0, 0] + 1).astype(int)

    time: float = 0.0
    last_index: int = 0

    # Iterate over the files
    for k, file in tqdm(enumerate(files)):

        ar = pyp.Archive(file, prepare=True, verbose=False)
        new_index = ar.getNsubint() + last_index

        data_channels = ar.getAxis(flag="F", edges=True)
        data_times = ar.getTimes() + time  # We add the time at the end of the previous observation

        total_times[last_index:new_index] = data_times

        # Remove RFI
        mask_RFI(ar, window_data)  # Account for individual RFIs and null single pulses
        zap_minmax(ar)  # Zap noisy frequency channels

        # Extract the data form the observation
        data = ar.getData()  # (128 single pulses x 128 frequency channels x 2048 phase bins) -> in this order
        weights = ar.getWeights()

        # Iterate over the single pulses
        for i, t in enumerate(data_times):

            # If the weights across all frequency channels are zero, we drop this sub-integration by marking the
            # frequency-averaged single pulse as all NaNs
            if np.all(ar.getWeights()[i, :].astype(int) == 0):

                for j in range(len(data_channels)): total_data[last_index + i] = np.nan

            # Otherwise, average in frequency
            else:

                #                # Iterate over the frequency channels
                #                for j, freq in enumerate(data_channels):

                #                    # Assign a weight equal to 0 to all the null single pulses
                #                    if int(weights[i, j]) != 0:
                #                        ar.setWeights(val=1.0 / (np.std(data[i, j, offpulsewindow]) ** 2), t=i, f=j)

                # Now we need to average in frequency
                if noise_rms != 0:
                    total_data[last_index + i, :] = np.average(data[i, :, :], axis=0,
                                                               weights=ar.getWeights()[i, :]) + np.random.normal(0,
                                                                                                                 noise_rms,
                                                                                                                 ar.getNbin())
                else:
                    total_data[last_index + i, :] = np.average(data[i, :, :], axis=0, weights=ar.getWeights()[i, :])

        time += ar.getDuration()
        last_index = new_index

    # Before returning the dataframe, we make sure to remove any rows with NaNs
    return pd.DataFrame(data=total_data, index=total_times, columns=np.arange(ar.getNbin())).dropna(axis=0, how="any")
