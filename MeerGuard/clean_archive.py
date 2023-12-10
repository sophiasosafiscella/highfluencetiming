#!/usr/bin/env python

# For python3 and python2 compatibility
from __future__ import (absolute_import, division, print_function, unicode_literals)

# Import CoastGuard
from coast_guard import cleaners
import argparse
import psrchive as ps
import os


def apply_surgical_cleaner(ar, tmp, cthresh=7.0, sthresh=7.0, plot=False):
	print("Applying the surgical cleaner")
	print("\t channel threshold = {0}".format(cthresh))
	print("\t  subint threshold = {0}".format(sthresh))

	surgical_cleaner = cleaners.load_cleaner('surgical')
	surgical_parameters = "chan_numpieces=1,subint_numpieces=1,chanthresh={1},subintthresh={2},template={0},plot={3}".format(
		tmp, cthresh, sthresh, plot)
	surgical_cleaner.parse_config_string(surgical_parameters)
	surgical_cleaner.run(ar)


def apply_bandwagon_cleaner(ar, badchantol=0.95, badsubtol=0.95):
	print("Applying the bandwagon cleaner")
	print("\t channel threshold = {0}".format(badchantol))
	print("\t  subint threshold = {0}".format(badsubtol))

	bandwagon_cleaner = cleaners.load_cleaner('bandwagon')
	bandwagon_parameters = "badchantol={0},badsubtol={1}".format(badchantol, badsubtol)
	bandwagon_cleaner.parse_config_string(bandwagon_parameters)
	bandwagon_cleaner.run(ar)


def MeerGuard_clean(archive_path, template_path, output_name: str, chan_thresh: float = 7.0, subint_thresh: float = 7.0,
                    badchantol: float = 0.95, badsubtol: float = 0.95, plot: bool = False, output_path: str = os.getcwd()):

	# Load an Archive file
	loaded_archive = ps.Archive_load(archive_path)
	archive_path, archive_name = os.path.split(loaded_archive.get_filename())
	archive_name_pref = archive_name.split('.')[0]
	archive_name_suff = "".join(archive_name.split('.')[1:])
	# psrname = archive_name_orig.split('_')[0]

	# Renaming archive file with statistical thresholds
	if output_name is None:
		out_name = "{0}_ch{1}_sub{2}.ar".format(archive_name_pref, chan_thresh, subint_thresh, archive_name_suff)
	else:
		out_name = output_name

	apply_surgical_cleaner(loaded_archive, template_path, cthresh=chan_thresh, sthresh=subint_thresh, plot=plot)
	apply_bandwagon_cleaner(loaded_archive, badchantol=badchantol, badsubtol=badsubtol)

	# Unload the Archive file
	print("Unloading the cleaned archive: {0}".format(out_name))
	loaded_archive.unload(
		str(out_name))  # need to typecast to str here because otherwise Python converts to a unicode string which the PSRCHIVE library can't parse

    # Get the new weights
	return loaded_archive.get_weights()
