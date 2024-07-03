import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import pypulse as pyp

import numpy as np
import pandas as pd
import pypulse
import glob
import string
import os
import sys

from lmfit import Model, Parameters

from scipy.integrate import trapezoid
from scipy.signal import peak_widths, find_peaks

from tqdm import tqdm


def make_plots(dir):
    # Make plots of the single pulses

    for file in glob.glob(dir + "/*npy")[0:1]:

        data = np.load(file)
        print(data.shape)
        for sp_index in range(data.shape[0]):
            plt.plot(data[sp_index, :])
            plt.xlabel("Bins")
            plt.ylabel("Intensity")
            plt.title("Pulse " + str(sp_index) + " out of " + str(data.shape[0]))
            plt.savefig(dir + '/single_pulses/' + str(sp_index) + '.png')
            plt.show()
            plt.close


def count_sp(files):
    n_sp: int = 0
    for file in tqdm(files):
        ar = pyp.Archive(file, verbose=False)
        n_sp += ar.getNsubint()

    N_bin: int = ar.getNbin()

    return n_sp, N_bin


def calculate_rms(files, n_sp, n_chan):

    # Create an array to store the RMS values
    arr = np.arange(0, 100)
    rms_values = np.full((n_sp, n_chan), np.nan)

    # Iterate over the files
    n: int = 0
    for k, file in tqdm(enumerate(files)):

        ar = pyp.Archive(file, verbose=False)

        # Extract the data form the observation
        data = ar.getData()  # (128 single pulses x 128 frequency channels x 2048 phase bins) -> in this order

        # Iterate over the sub-integrations
        for i in range(ar.getNsubint()):

            # Iterate over the frequency channels
            for j in range(n_chan):

                sp = pyp.SinglePulse(data[i, j, :], opw=arr)
                rms_values[n, j] = sp.getOffpulseNoise()

            n += 1

    return rms_values

def calculate_sp_snr(files, n_sp):

    print("Calculating SP SNR")

    # Create an array to store the sn values
    arr = np.arange(0, 50)
    snr_values = np.full(n_sp, np.nan)

    # Iterate over the files
    n: int = 0
    for file in tqdm(files):

        ar = pyp.Archive(file, verbose=False)
        ar.fscrunch()

        # Extract the data form the observationsp = pyp.SinglePulse(data[i, j, :], opw=arr)
        data = ar.getData()  # (128 single pulses x 128 frequency channels x 2048 phase bins) -> in this order

        # Iterate over the sub-integrations
        for i in range(ar.getNsubint()):

            sp = pyp.SinglePulse(data[i, :], opw=arr)
            snr_values[n] = sp.getSN()
            print(snr_values[n])
            print(n)
#            if np.isnan(snr_values[n]):
#                print("Error found calculating SNR")
#                print(snr_values[n])
#                print(data[i, :])

            n += 1

    return snr_values


def get_average_pulse(pulses_files, nbins):

    av_pulse_profile = np.zeros(nbins)

    for i, file in tqdm(enumerate(pulses_files)):

        data = pyp.Archive(file, verbose=False).fscrunch().getData()

        if np.any(np.isnan(data)):
            print(f"Found NaN in i={i}, file = {file}")
            print(data)
            sys.exit()
        av_pulse_profile += np.average(data, axis=0)

    av_pulse_profile /= len(pulses_files)

    return av_pulse_profile


def find_energy_windows(template_data, window_factor, bins_factor: float, plot:bool = False):

    # finds the peaks in the template data
    peaks_pos, properties = find_peaks(template_data, distance=50, width=10, prominence=0.2)

    energy_margins = np.zeros((len(peaks_pos), 2), dtype=float)

    for i, peak in enumerate(peaks_pos):
        peak_width = peak_widths(template_data, np.array([peak]), rel_height=0.5)[0][0]
        energy_margins[i, 0] = peak - window_factor * peak_width  # left margin
        energy_margins[i, 1] = peak + window_factor * peak_width  # right margin

    if plot:
        sns.set_style("darkgrid")
        sns.set_context("paper")

        fig, ax = plt.subplots()
        x = np.array([*range(len(template_data))])
        ax.plot(x, template_data)
        ax.scatter(peaks_pos, template_data[peaks_pos], c="red", label='peaks')

        for left, right in zip(energy_margins[:, 0], energy_margins[:, 1]):
            ax.fill_between(x, min(template_data), max(template_data),
                            where=((x < right) & (x > left)), color="C1", alpha=0.4)

        ax.set_xlabel("Bins")
        ax.set_ylabel("Intensity")
        plt.title("Pulse components' windows")
        plt.savefig("./figures/energy_windows.pdf")
        plt.show()
        plt.close()

    return np.rint(np.divide(energy_margins, bins_factor)).astype(int)


def find_windows(template_file: str,  # name of the template fits_file
                 pulses_directory: str,  # directory containing the single pulses
                 results_dir: str,  # Directory with the clusters_toas
                 files: str,         # Names of the Archive files
                 window_percentage: float,  # percentage of the window we'll use for the main peak window
                 windows_factor: float,  # scale factor for the energy windows
                 bscrunching_factor: float = 4,
                 plot=False):

    # find the peak of the template
    template = pypulse.Archive(template_file)
#    template.bscrunch(factor=bscrunching_factor)
    template_data = template.getData()
    template_peak_pos = np.argmax(template_data)
    offpulse = pypulse.SinglePulse(template_data, windowsize=int(template.getNbin() // 8)).calcOffpulseWindow()
    offpulsewindow = [min(offpulse), max(offpulse)]

    # find the average of the pulses
    av_pulse_file = glob.glob(results_dir + "av_pulse_profile.npy")
    if len(av_pulse_file) == 0:
        print("Calculating average pulse...")
        average_pulse_data = get_average_pulse(files[0:100], nbins=512)
        np.save(results_dir + "av_pulse_profile.npy", average_pulse_data)
    else:
        average_pulse_data = np.load(results_dir + "av_pulse_profile.npy")

    av_pulse_peak_pos = np.argmax(average_pulse_data)
    print(f"Av pulse peak pos = {av_pulse_peak_pos}")
    # If the template has 2048 bins and the pulses have 512, we divide:
    bins_ratio = int(len(template_data) / len(average_pulse_data))
    if bins_ratio != 1:
        print("bins_ratio = " + str(bins_ratio))
        template_data = template_data[0:len(template_data):bins_ratio]
        template_peak_pos = round(template_peak_pos / bins_ratio)

    # in case we want to plot
    if plot:
        fig, ax = plt.subplots()

        bins512 = range(len(average_pulse_data))

        ax.plot(bins512, average_pulse_data, c="C0", label="Average pulse")
        ax.scatter(bins512, average_pulse_data, c="C0")
        #        ax.plot(bins512, template_data, c="C1", label="Template")
        # ax.scatter(bins512, template_data, c="C1")

        ax.set_xlim([template_peak_pos - 50, template_peak_pos + 50])

        ax.axvline(x=av_pulse_peak_pos, ls="--", c='C4', label="Average peak = " + str(template_peak_pos))
        ax.axvline(x=template_peak_pos, ls="--", c='C3', label="Template peak = " + str(av_pulse_peak_pos))
        plt.title("Comparison of template and average")
        plt.legend(loc="upper left")
        plt.show()

    # calculate the offset between the peaks and correct the template peak position
    offset = template_peak_pos - av_pulse_peak_pos
    template_peak_pos -= offset
    # Get the pulse window as a fraction of the pulse phase: 10 or 15%,or 12.5% (1/8) of the pulse phase
    width = int(len(bins512) / 100.0 * window_percentage)
    left_margin = int(template_peak_pos - int(width / 2))
    right_margin = int(template_peak_pos + int(width / 2))

    # find the energy windows
    energy_windows = find_energy_windows(template_data, windows_factor, bins_ratio, plot=False)

    if plot:
        sns.set_style("darkgrid")
        sns.set_context("paper", font_scale=1.4)

        fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
        plt.subplots_adjust(wspace=0, hspace=0)

        x = np.array([*bins512])
        ax[0].plot(x, template_data)
        ax[0].axvline(x=template_peak_pos, ls="--", c='k')
        ax[0].axvline(x=left_margin, ls=":", c="grey")
        ax[0].axvline(x=right_margin, ls=":", c="grey")
        ax[0].fill_between(x, min(template_data), max(template_data),
                           where=((x < right_margin) & (x > left_margin)), color="C1", alpha=0.4)
        #        ax[0].set_xlabel("Bins")
        ax[0].set_ylabel("Intensity")
        ax[0].title.set_text("Pulse Windows")

        peaks_pos, properties = find_peaks(template_data, distance=50, width=10, prominence=0.2)
        ax[1].plot(x, template_data)
        ax[1].scatter(peaks_pos, template_data[peaks_pos], c="red", label='peaks')

        for left, right in zip(energy_windows[:, 0], energy_windows[:, 1]):
            ax[1].fill_between(x, min(template_data), max(template_data),
                               where=((x < right) & (x > left)), color="C2", alpha=0.4)
        ax[1].set_xlabel("Bins")
        ax[1].set_ylabel("Intensity")
        #        ax[1].title.set_text("Pulse components' windows")

        for n, sub in enumerate(ax):
            sub.text(0.05, 0.8, string.ascii_uppercase[n], transform=sub.transAxes,
                     size=20)

        plt.savefig("./windows.pdf")
        plt.show()
        plt.close

    return np.vstack((offpulsewindow, [left_margin, right_margin], energy_windows))


def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / 4 / stddev) ** 2) # + baseline


def estimate_peak(window_data, windows, baseline, window_index, plot=False):

    # Remove the baseline
    window_data -= baseline

    # Find the index of the peak of the (observed) pulse
    peak_index = np.argmax(window_data) + windows[1, 0]  # Index (in the window) of the biggest peak in the pulse window

    # Create bins in the pulse window
    x_data = np.arange(len(window_data)) + windows[1, 0]

    gmodel = Model(gaussian)
    params = Parameters()
    params.add("mean", value=peak_index)
    params.add("amplitude", value=np.max(window_data), min=0.0)
    #    print("peak index = " + str(peak_index))
    #    print("max in window = " + str(np.max(window_data)))
    params.add("stddev", value=1.0, min=0.0)
#    params.add("baseline", value=baseline)
#    params['baseline'].vary = False
    result = gmodel.fit(window_data, params, x=x_data)


    new_x = np.arange(x_data[0], x_data[-1], 0.5)
    new_y = gaussian(new_x, amplitude=result.params["amplitude"].value,
                     mean=result.params["mean"].value,
                     stddev=result.params["stddev"].value)
    #    peak_amp = np.max(new_y) - baseline
    peak_amp = np.max(window_data) - baseline
    peak_pos = new_x[np.argmax(new_y)]
#    peak_width = peak_widths(window_data, np.array([np.argmax(window_data)]), rel_height=0.5)[0][0]
#    peak_idx = [np.argmax(new_y)]

    peaks, _ = find_peaks(new_y)
    width = peak_widths(new_y, peaks, rel_height=0.5)
    peak_width = width[0][0]
#    peak_width = 2 * result.params["stddev"].value * np.sqrt(2 * np.log(2))

    if plot:
        sns.set_style("darkgrid")
        sns.set_context("paper", font_scale=1.4)

        fig, ax = plt.subplots()
        ax.plot(new_x[peaks], new_y[peaks], "x")
        ax.plot(x_data, window_data, c="#636EFA", label="Original")  # c="#f29ad8"
        ax.scatter(x_data, window_data, c="#636EFA")  #  c="#f29ad8"
        ax.plot(new_x, new_y, c="#EF553B", label="Fit")  # c="#e305ad",

        ax.axvline(x=peak_pos, ls="--", c='k', label="Peak position")
        ax.axvline(x=windows[1, 0], ls=":", c="grey")
        ax.axvline(x=windows[1, 1], ls=":", c="grey")
#        ax.fill_between(new_x, 0, peak_amp,
#                        where=(new_x < peak_pos + peak_width/20) &
#                              (new_x > peak_pos - peak_width/20),
#                        color='#B6E880', alpha=0.3, label="Width")  # color="#f9dd9a",

        print(*width[1:])
        sys.exit()
        ax.axhline(*width[1:], color='C1')

#        textstr = '\n'.join((
#            r'$\mathrm{Position}=%i$' % (peak_pos,),
#            r'$\mathrm{Amplitude}=%.4f$' % (peak_amp,),
#            r'$\mathrm{Width}=%.4f$' % (peak_width,)))
        #            ,r'baseline =%.4f' % (baseline,)))

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)

        # place a text box in upper left in axes coords
#        ax.text(0.70, 0.95, textstr, transform=ax.transAxes, fontsize=12,
#                verticalalignment='top', bbox=props)

        ax.set_xlabel("Bins")
        ax.set_ylabel("Intensity")
        #        ax.set_title("Pulse " + str(window_index))
        plt.legend(loc='upper left')

        #        plt.savefig("./clusters_toas/windows/pulse_" + str(window_index) + ".png")
        plt.tight_layout()
        plt.savefig("./figures/fits/pulse_fit_" + str(window_index) + ".png")
        plt.show()
        plt.close()

    return peak_amp, peak_pos, peak_width


def get_energy(pulse_data, windows, baseline):
    energy: float = 0.0

    for left, right in zip(windows[2:, 0], windows[2:, 1]):
        energy += (trapezoid(pulse_data[left: right], dx=1) - baseline)

    return energy


# @jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def get_params(data_file, windows, results_dir, plot=False):
    # Left and right margins of the main pulse window
    pulses_df = pd.read_pickle(data_file)
    pulses_data = pulses_df.to_numpy()
    window_data = pulses_data[:, windows[1, 0]:windows[1, 1]]

    n_pulses = window_data.shape[0]

    #    peak_pos = np.argmax(window_data, axis=1)   # position of the peak in each single pulse

    lse_peak_amp = np.full((n_pulses), np.nan)
    lse_pos = np.full((n_pulses), np.nan)
    lse_width = np.full((n_pulses), np.nan)
    lse_energy = np.full((n_pulses), np.nan)

    for i in tqdm(range(n_pulses)):
        # windows[0:0] and windows[0:1] are the left and right edges
        # of the off-pulse window, respectively
        baseline = np.average(pulses_data[i, windows[0, 0]:windows[0, 1]])

        lse_peak_amp[i], lse_pos[i], lse_width[i] = estimate_peak(window_data[i, :], windows,
                                                                  np.average(window_data[i, 0:4]), i, plot=plot)
        lse_energy[i] = get_energy(pulses_data[i, :], windows, baseline)

    #        if lse_energy[i] > 100.0 or lse_peak_amp[i] > 100.0 or lse_energy[i] < -30.0:
    #            sns.set_style("whitegrid", {'axes.grid': False})
    #            fig = plt.figure()
    #            plt.title(i)
    #            plt.plot(pulses_data[i, :])
    #            plt.savefig(results_dir + "/plots/" + str(i) + ".png")
    #            plt.show()
    #            plt.close()

    org_features = pd.DataFrame(data={'Pos': lse_pos, 'Width': lse_width, 'Amp': lse_peak_amp, 'Energy': lse_energy},
                                index=pulses_df.index)

    #    org_features = np.vstack((lse_pos, lse_width, lse_peak_amp, lse_energy)).T
    #    features = StandardScaler().fit_transform(org_features)

    # Drop the rows where the width is 0.0
    return org_features.loc[~((org_features['Width'] == 0.0))]
