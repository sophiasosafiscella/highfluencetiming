import matplotlib.pyplot as plt
import pypulse as pyp
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

from RFI_utils import zap_minmax, mask_RFI, chisq_filter, opw_peaks


def count_sp(files):
    n: int = 0
    for file in files:
        ar = pyp.Archive(file, verbose=True, lowmem=True)
        n += ar.getNsubint()

    return n


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


def to_binary(files, out_dir, sp_total, bandpass=None, shift: int = -220):

    # If a pair of values was provided to restrict the bandwidth, then multiply the second value by -1
    # in order to remove that many channels at the lower end of the bandwidth
    if bandpass is None:
        bandpass = [0, None]
    else:
        bandpass[1] *= -1

    total_times = np.zeros(sp_total)
    time: float = 0.0
    last_index: int = 0

    channels = pyp.Archive(files[0], prepare=False, center_pulse=False, baseline_removal=False,
                         lowmem=True, verbose=False).getAxis(flag="F", edges=True)[bandpass[0]: bandpass[1]]

    for file in tqdm(files):
        ar = pyp.Archive(file, prepare=False, center_pulse=False, baseline_removal=False,
                         lowmem=True, verbose=False)

        ar.pscrunch()
        ar.dedisperse()

        new_index = ar.getNsubint() + last_index

        data_times = ar.getTimes() + time  # We add the time at the end of the previous observation
        total_times[last_index:new_index] = data_times

        rolled = np.roll(ar.getData(), shift, axis=2)
        rolled -= np.average(np.average(np.average(rolled, axis=1), axis=0)[0:100])   # Subtract the baseline

        # Sometimes we don't want to use the channels at the edges. In that case we restrict the bandpass.
        np.save(out_dir + file[-35:-3] + ".npy", rolled[:, bandpass[0]: bandpass[1], :])

        time += ar.getDuration()
        last_index = new_index

    return total_times, channels

def merge(ds, binary_files, times_data, channels_data, full_weights, N_bin, sp_total, noise_rms, noise_factor):

    unnormalized_data = np.empty((sp_total, N_bin))
    normalized_data = np.empty((sp_total, N_bin))  # rows: single pulses, columns: phase bins

    # Parameters for the dynamic spectrum
    ds_channels = np.asarray(list(ds.index))           # Upper edges of the frequency bins
    ds_times = np.asarray(ds.columns.values.tolist())  # Upper edges of the time bins
    ds_arr = ds.to_numpy()                             # Dynamic spectrum data

    # Iterate over the files
    last_index: int = 0
    for file in tqdm(binary_files):

        data = np.load(file)
        Nsubint, Nchan, Nbin = np.shape(data)
        new_index = Nsubint + last_index

        weights = full_weights[last_index: new_index, :]

        # Iterate over the single pulses
        for i, t in enumerate(times_data[last_index:new_index]):

            # If the weights across all frequency channels are zero, we drop this sub-integration by marking the
            # frequency-averaged single pulse as all NaNs
            if np.all(weights[i, :] < 1e-5): # Instead of weights == 0, let's use a very small number

                normalized_data[last_index + i, :] = np.nan
                unnormalized_data[last_index + i, :] = np.nan
                print("One sub-integration was dropped...")

            # Otherwise, average in frequency
            else:

                # since we have the upper edges of the time bins, we look for all the DS edges that are bigger than the
                # given value of t by doing ds_times[ds_times >= t] and we keep the smaller one using np.amin()
#                sp_index = np.argwhere(ds_times == np.amin(ds_times[ds_times >= t]))[0][0]

                # We subtract the given value of t from each element of the array ds_times. The first positive
                # difference will correspond to the bin we are looking for.
                # argmax will stop at the first True ("In case of multiple occurrences of the maximum values,
                # the indices corresponding to the first occurrence are returned.")
                sp_index = np.argmax((ds_times - t) > 0)

                normalized_channels = np.zeros((Nchan, Nbin))

                # Iterate over the frequency channels
                for j, freq in enumerate(channels_data):

                    # frequency index
#                    chan_index = np.argwhere(ds_channels == np.amin(ds_channels[ds_channels >= freq]))[0][0]
                    chan_index = np.argmax((ds_channels - freq) > 0)

                    # Divide by the dynamic spectrum
                    normalized_channels[j, :] = np.divide(data[i, j, :], ds_arr[chan_index, sp_index])

                    # Add the same noise to the normalized and the unnormalized data
                    noise = np.random.normal(0.0, noise_factor * noise_rms[last_index + i, j], size=Nbin)
                    data[i, j, :] += noise
                    normalized_channels[j, :] += noise

                # Now we need to average in frequency
                normalized_data[last_index + i, :] = np.average(normalized_channels, axis=0, weights=weights[i, :])
                unnormalized_data[last_index + i, :] = np.average(data[i, :, :], axis=0, weights=weights[i, :])

        last_index = new_index

    # Before returning the dataframe, we make sure to remove any rows with NaNs
    normalized_df = pd.DataFrame(data=normalized_data, index=times_data, columns=np.arange(Nbin)).dropna(axis=0,
                                                                                                         how="any")
    unnormalized_df = pd.DataFrame(data=unnormalized_data, index=times_data, columns=np.arange(Nbin)).dropna(
        axis=0, how="any")

    return normalized_df, unnormalized_df
