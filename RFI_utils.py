import numpy as np
import pypulse as pyp
from pypulse.utils import xrange

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


def mask_RFI(data, weights, window_data, factor=8.0):

    Nsubint, Nchan, Nbin = np.shape(data)

    # Create a threshold that is factor times the maximum value inside the main peak window
    threshold = factor * np.expand_dims(np.abs(np.amax(data[:, :, window_data[1, 0]:window_data[1, 1]], axis=2)), axis=2)  # Maximum inside each pulse window

    # 0:window_data[1,0], window_data[1,0]  means "from the beginning, to the left edge of the main peak window"
    # window_data[1,0]:np.shape(data)[1]]   means "from the right edge of the main peak window, to the end
    # We're taking all the times (rows), all the frequencies) columns, and only the phase bins outside the main peak window.
    # FALSE MEANS THAT THERE IS NO RFIs
    # TRUE MEANS THAT THERE IS AN RFI
    RFI_mask = (
            data[:, :, np.r_[0:window_data[1, 0], window_data[1, 0]:np.shape(data)[2]]] > threshold).any(
        axis=2) # The any(axis=2) is to check where this condition became true along the phase bins axis for a given single pulse

    # Flag all the null single pulses
    null_mask = (data == 0.0).all(axis=2)  # Create a null_mask where True means a single pulse that is al zeros

    # Combine all masks
    complete_mask = np.logical_or(RFI_mask, null_mask)

    # Assign a weight equal to 0 to all the RFI-affected and null single pulses
    for i in xrange(Nsubint):              # subintegration indexes
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
            opw_average = np.average(data[i, j, window_data[0, 0], window_data[0, 0]])
            opw_maximum = np.amax(data[i, j, np.r_[0:window_data[1, 0], window_data[1, 0]:Nbin]])

            if ((opw_maximum - opw_average) > threshold * opw_maximum):
                weights[i, j] = 0.0

    return weights

