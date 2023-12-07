import glob
import subprocess
from tqdm import tqdm

F0 : float = 62.2958887973344346
psrfbins : int = 32
fits_files = glob.glob("./*fits")

fil_files = fits_files
for n, file in enumerate(fil_files):
	fil_files[n] = file[:-5] + ".fil"

# Convert the .fits files to SIGPROC filterbank format files
subprocess.run("digifil -b 8 -d 1 -cont -o" + ' '.join(fil_files) + " " + ' '.join(fits_files))

for file in tqdm(fil_files):

	subprocess.run("./rficlean " + file + " -o " + file[:-5] + "_rficleaned.fits " + "-psrf " + str(F0) + " -psrfbins " + str(psrfbins))
