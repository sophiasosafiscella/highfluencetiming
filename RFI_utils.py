from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pypulse as pyp
from numpy import ndarray, dtype

from MeerGuard import clean_archive
from pypulse.utils import xrange
from sklearn.preprocessing import normalize
import clfd
from clfd import DataCube, featurize, profile_mask
from tqdm import tqdm
import glob
import sys
import os
import subprocess
from tqdm import tqdm


def meerguard(files, pulses_dir, band, template_file):

    # Clean the .ar files
    for file in files:
        clean_archive.MeerGuard_clean(archive_path=file, template_path=template_file, output_name=file[:-3] + "_cleaned.ar")

    return


def zap(obs, val=0.0, t=None, f=None):
    '''
    Passes straight to archive's setWeights()
    '''
    obs.setWeights(val=val, t=t, f=f)


def zap_minmax(data, weights, opw, windowsize=20, threshold=4.0):
    '''
    Run NANOGrav algorithm, median zapping. Run per subintegration
    windowsize = 20 frequency bins long
    threshold = 4 sigma
    '''

    Nsubint, Nchan, Nbin = np.shape(data)

    if Nchan <= windowsize:
        for i in xrange(Nsubint):
            for j in xrange(Nchan):
                subdata = data[i, 0, :, opw]
                compptp = np.ptp(data[i, 0, j, opw])
                ptps = np.zeros(windowsize)
                for k in xrange(windowsize):
                    ptps[k] = np.ptp(subdata[k, :])

                med = np.median(ptps)
                if compptp > threshold * med:
                    weights[i, j] = 0.0
        return

    for i in xrange(Nsubint):              # subintegration index
        for j in xrange(Nchan):            # frequency channel index
            low = j - windowsize // 2      # lower edge of the frequency window
            high = j + windowsize // 2     # upper edge of the frequency window

            if low < 0:
                high = abs(low)
                low = 0
            elif high > Nchan:
                diff = high - Nchan
                high -= diff
                low -= diff

            subdata = data[i, low:high, opw]   # Data inside the off-pulse window for the channels INSIDE the window
            compptp = np.ptp(data[i, j, opw])  # Peak-to-peak range of values for the off-pulse window for ALL freq channels
            ptps = np.zeros(windowsize)        # Array containing the peak-to-peak ranges for the channels INSIDE the window

            # Save the off-pulse-window peak-to-peak ranges of the channels INSIDE the frequency window
            for k in xrange(windowsize):
                ptps[k] = np.ptp(subdata[k, :])

            # ptps = np.array(map(lambda subdata: np.ptp(subdata),data[i,0,low:high,opw]))

            med = np.median(ptps)              # Median value of the peak-to-peak ranges
            if compptp > threshold * med:
                weights[i, j] = 0.0

    return weights


def mask_RFI(data, weights, window_data, factor=6.0):

    Nsubint, Nchan, Nbin = np.shape(data)

    # Create a threshold that is factor times the maximum value inside the main peak window
    threshold = factor * np.expand_dims(np.abs(np.amax(data[:, :, window_data[1, 0]:window_data[1, 1]], axis=2)), axis=2)  # Maximum inside each pulse window

    # 0:window_data[1,0], window_data[1,0]  means "from the beginning, to the left edge of the main peak window"
    # window_data[1,0]:np.shape(data)[1]]   means "from the right edge of the main peak window, to the end
    # We're taking all the times (rows), all the frequencies) columns, and only the phase bins outside the main peak window.
    # FALSE MEANS THAT THERE IS NO RFIs
    # TRUE MEANS THAT THERE IS AN RFI
    RFI_mask = (
            data[:, :, np.r_[0:window_data[1, 0], window_data[1, 0]:Nbin]] > threshold).any(
        axis=2) # The any(axis=2) is to check where this condition became true along the phase bins axis for a given single pulse

    # Flag all the null single pulses
    null_mask = (data == 0.0).all(axis=2)  # Create a null_mask where True means a single pulse that is al zeros

    # Combine all masks
    complete_mask = np.logical_or(RFI_mask, null_mask)

    # Assign a weight equal to 0 to all the RFI-affected and null single pulses
    for i in xrange(Nsubint):               # subintegration indexes
        for j in xrange(Nchan):             # frequency channel index
            if complete_mask[i, j]:
                weights[i, j] = 0.0

    # If for any given subintegration there are more bad frequency channels than good ones, we get rid of the subintegration
        n_RFI_channels = complete_mask[i, :].sum()
        if n_RFI_channels > int(0.5 * Nchan):
            for j in xrange(Nchan):
                weights[i, j] = 0.0

    return weights


def chisq_filter(ar, template_file, threshold=0.3):

    # Load the template
    template = pyp.Archive(template_file)
    template.bscrunch(factor=4)
    template_data = template.getData()

    # Prepare data
    data = ar.getData()
    nbin: int = ar.getNbin()

    for i in xrange(ar.getNsubint()):              # sub-integration indexes

            sp = pyp.SinglePulse(np.average(data[i, :, :], axis=0), windowsize=nbin)

            if (abs(sp.fitPulse(template_data)[6]) > threshold):

                for j in xrange(ar.getNchan()):  # frequency channel index
                    ar.setWeights(val=0.0, t=i, f=j)


def opw_peaks(data, weights, window_data, threshold=0.75):

    Nsubint, Nchan, Nbin = np.shape(data)

    for i in xrange(Nsubint):  # sub-integration indexes

        for j in xrange(Nchan):  # frequency channel index

#            sp = pyp.SinglePulse(data[i, j, :], opw=np.arange(0, 100))
#            opw = sp.calcOffpulseWindow()
            opw_average = np.average(data[i, j, window_data[0, 0]: window_data[1, 0]])
            opw_maximum = np.amax(data[i, j, np.r_[0:window_data[1, 0], window_data[1, 0]:Nbin]])

            if ((opw_maximum - opw_average) > threshold * opw_maximum):
                weights[i, j] = 0.0

    return weights

def clfd(file, weights, plot=False):

    # Load folded archive produced with PSRCHIVE
    cube = DataCube.from_psrchive(file)

    # Compute chosen profile features.
    # The output is a pandas DataFrame with feature names as columns, and (subint, channel) tuples as rows.
    features = featurize(cube, features=('std', 'lfamp'))

    # From there, compute profile mask, optionally excluding some known bad channels from the analysis.
    stats, mask = profile_mask(features, q=2.0)

    if plot:

        data = pyp.Archive(file).getData()
        for i in xrange(np.shape(mask)[0]):  # subintegration indexes
            for j in xrange(np.shape(mask)[1]):  # frequency channel index
                if mask[i, j]:

                    plt.close()
                    plt.title("Subintegration " + str(i) + " Channel " + str(j))
                    plt.plot(data[i, j, :])
                    plt.savefig("./clfd/" + str(i) + "_" + str(j) + ".png")
                    plt.show()

    for i in xrange(np.shape(mask)[0]):                 # subintegration indexes
        for j in xrange(np.shape(mask)[1]):             # frequency channel index
            if mask[i, j]:
                weights[i, j] = 0.0

    return weights


def remove_RFIs(files, binary_files, windows_data, weights, rms_array, template_file,
                clfd_ok: bool = False,
                mask_rfi_ok: bool = False,
                zap_minmax_ok: bool = False,
                chisq_filter_ok: bool = False,
                opw_peaks_ok: bool = False):

    # Get the off-pulse window
    offpulsewindow: ndarray[Any, dtype[Any]] = np.linspace(windows_data[0, 0], windows_data[0, 1],
                                 num=(windows_data[0, 1] - windows_data[0, 0] + 1).astype(int))

    print(rms_array)
   # Calculate 1/sigma2 for each single pulse
    sigma2 = normalize(np.power(rms_array, -2), axis=0)
    sys.exit()
    print(sigma2)
    # IF THE WEIGHTS HAVE NOT ALREADY BEEN FLAGGED AS ZERO, assign a weight equal to 1/sigma2 to each single pulse
    for indexes in np.argwhere(weights > 0.0):
        weights[indexes[0], indexes[1]] = sigma2[indexes[0], indexes[1]]

    # Iterate over the files
    last_index: int = 0
    for n, binary_file in tqdm(enumerate(binary_files)):

        data = np.load(binary_file)

        new_index = np.shape(data)[0] + last_index

        if clfd_ok:
            weights[last_index: new_index, :] = clfd(files[n], weights[last_index: new_index, :])

        if mask_rfi_ok:       # Account for individual RFIs and null single pulses
            weights[last_index: new_index, :] = mask_RFI(data, weights[last_index: new_index, :], windows_data)

        if zap_minmax_ok:     # Zap noisy frequency channels
            weights[last_index: new_index, :] = zap_minmax(data, weights[last_index: new_index, :], offpulsewindow)

        if chisq_filter_ok:    # Filter RFIs by the chisq from fitting the SPs to the template
            chisq_filter(data, template_file=template_file)

        if opw_peaks_ok:      # Filter single pulses with sharp peaks in the off-window region
            weights = opw_peaks(data, weights, windows_data)

        last_index = new_index

    return weights
