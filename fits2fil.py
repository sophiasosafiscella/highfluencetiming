import subprocess
import glob

files = glob.glob('*.fits')


for file in files:
	if len(glob.glob(file[:-4] + "fil")) == 0:
		print("Processing " + file)
		subprocess.run("python psrfits2fil.py " + file, shell=True, check=True)

files = glob.glob('*.fits')

F0: float = 92.2958887973344346
psrfbins : int = 16

for file in files:
		subprocess.run("./rficlean " + file[:-4] + "fil" + " -t 156762 -o " + file[:-5] + "_cleaned.fil" +
		                                                                 " -psrf " + str(F0) +
		                                                                 " -psrfbins " + str(psrfbins), shell=True, check=True)

