import numpy as np
from coast_guard import config
from coast_guard import cleaners
from coast_guard import clean_utils
from coast_guard.cleaners import config_types
from coast_guard import utils


class BandwagonCleaner(cleaners.BaseCleaner):
    name = 'bandwagon'
    description = 'De-weight profiles from subints/channels where most of ' \
                  'the profiles are already masked.'


    def _set_config_params(self):
        self.configs.add_param('badchantol', config_types.FloatVal, \
                               help='The fraction (0 to 1) of bad channels that ' \
                                    'can be tolerated before a sub-int is completely ' \
                                    'masked.')
        self.configs.add_param('badsubtol', config_types.FloatVal, \
                               help='The fraction (0 to 1) of bad sub-ints that ' \
                                    'can be tolerated before a channel is completely ' \
                                    'masked.')
        self.parse_config_string(config.cfg.bandwagon_default_params)


    def _clean(self, ar):
        nchan = ar.get_nchan()
        nsub = ar.get_nsubint()
        weights = (ar.get_weights() > 0)

        nchan_masked = np.sum(weights.sum(axis=0)==0)
        nsub_masked = np.sum(weights.sum(axis=1)==0)

        sub_badfrac = 1-weights.sum(axis=1)/float(nchan-nchan_masked)
        chan_badfrac = 1-weights.sum(axis=0)/float(nsub-nsub_masked)

        sub_is_bad = np.argwhere(sub_badfrac>self.configs.badchantol)
        for isub in sub_is_bad:
            clean_utils.zero_weight_subint(ar, isub)

        chan_is_bad = np.argwhere(chan_badfrac>self.configs.badsubtol)
        for ichan in chan_is_bad:
            clean_utils.zero_weight_chan(ar, ichan)


Cleaner = BandwagonCleaner
