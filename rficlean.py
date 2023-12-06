import numpy
import glob
import subprocess
import os
from tqdm import tqdm

F0 : float = 62.2958887973344346
psrfbins : int = 32
files = glob.glob("./*fits")

out_folder = "./rficleaned"
if not os.path.isdir(out_folder):
	os.makedirs(out_folder)

for file in tqdm(files):
	subprocess.run("./rficlean " + file + " -o " + out_folder + file[1:-3] + "_rficleaned.fits " + "-psrf " + str(F0) + " -psrfbins " + str(psrfbins))
