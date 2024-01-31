import glob
import subprocess
import your
from tqdm import tqdm

F0: float = 62.2958887973344346
psrfbins: int = 32
block_size: int = 156762

# Get all the PSRFits files
fits_files = glob.glob("./*calibP")

# For each PSRFits fits_file
for n, fits_file in tqdm(enumerate(fits_files)):
	fil_file: str = fits_file[:-7] + ".fil"
	cleaned_file: str = fits_file[:-7] + "_rficleaned.fil"
	new_fits_file: str = fits_file[:-7] + "_rficleaned.fil"

	# Convert the .fits files to SIGPROC filterbank format files
	subprocess.run("./digifil -b 8 -d 1 -o" + fil_file + " " + fits_file, shell=True)

	# Clean using RFIClean
	subprocess.run("./rficlean -t " + str(block_size) + " -psrf " + str(F0) + " -psrfbins " + str(psrfbins) +
	               " -o " + cleaned_file + fil_file, shell=True)

	# Convert back to PSRFits using Your
	fil_file = your.Your(cleaned_file)
	writer_object = your.Writer(fil_file, outdir=".", outname=new_fits_file)
	writer_object.to_fits()
