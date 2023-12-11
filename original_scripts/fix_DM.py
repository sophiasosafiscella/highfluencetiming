import glob
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model, Parameters
import astropy.units as u
import pypulse as pyp
from pint.models import get_model
import sys

def FD_curve(frequencies, FD1, FD2):

    return FD1 * np.power(np.log10(frequencies), 1) \
        + FD2 * np.power(np.log10(frequencies), 2)

def dispersion_curve_3terms(frequencies, t_infty, k_DMX):
    alpha = 4.4
    t_nu = np.array([t_infty + k_DMX / (nu ** 2) for nu in frequencies])

    return t_nu


band = "820_band"

# Get the observations
pulses_dir: str = "/media/svsosafiscella/D86091306091167A/NANOGrav/" + band + "/"
low_res_file = glob.glob(pulses_dir + "low_res/low*pF*")[0]  # Low-resolution file to create the dynamic spectrum
template_file = glob.glob("./data/*sm")[0]                   # Files containing the template
template_object = pyp.Archive(template_file, verbose=False)
template_object.bscrunch(factor=4)

# Get the timing model
timing_model = get_model("./J2145-0750_NANOGrav_12yv4.gls.par")

ar = pyp.Archive(low_res_file, verbose=False)  # (216, 8, 512)
ar.tscrunch()  # (8, 512)
frequencies_MHz = ar.getAxis(flag="F", edges=False) * u.MHz
frequencies_GHz = frequencies_MHz.to(u.GHz).value
toas = ar.fitPulses(template_object.getData(), nums=[1])[0] * u.us  # (8)

# create a set of Parameters
dispersion_model = Model(dispersion_curve_3terms)
params = Parameters()
params.add('t_infty', value=20.0, vary=True)
params.add('k_DMX', value=-20.0, vary=True)

# do fit, here with the default leastsq algorithm
result = dispersion_model.fit(toas.value, params=params, frequencies=frequencies_GHz)
fitted_function = dispersion_curve_3terms(frequencies_GHz, result.best_values['t_infty'],
                                          result.best_values['k_DMX'])

# correct for the FD delay curve
FD_values = (FD_curve(frequencies_GHz, timing_model.FD1.value, timing_model.FD2.value) * u.s).to(u.us)
corrected_toas = toas - FD_values

# plot
plt.xlabel("Frequency [GHz]")
plt.ylabel("TOA [$\mu$s]")
plt.plot(frequencies_GHz, fitted_function, c="C1",
         label='$r(\\nu) = r_\mathrm{\infty} + r_\mathrm{2} \\nu^{-2}$')
plt.plot(frequencies_GHz, FD_values, c="C2", label="FD curve")
plt.scatter(frequencies_GHz, toas, c="C0", label="Original TOAs")
plt.legend()
plt.savefig("./fix_DM/original.png")
plt.show()
plt.close()

# fit a new dispersion curve
dispersion_model_2 = Model(dispersion_curve_3terms)
result_2 = dispersion_model.fit(corrected_toas.value, params=params, frequencies=frequencies_GHz)
fitted_function_2 = dispersion_curve_3terms(frequencies_GHz, result_2.best_values['t_infty'],
                                            result_2.best_values['k_DMX'])

k_DMX = result_2.best_values['k_DMX'] * u.us * u.GHz * u.GHz
kMHz = 4.148808 * 10.0**3 * (u.MHz)**2 * (u.cm)**3 * u.s / (u.pc)
kGHz = kMHz.to((u.GHz)**2 * (u.cm)**3 * u.us / (u.pc))
DMX = k_DMX/kGHz
print(DMX)

# plot
plt.xlabel("Frequency [GHz]")
plt.ylabel("TOA [$\mu$s]")
plt.text(0.800, 1.5, "k_DMX = " + str(k_DMX))
plt.text(0.800, 1.3, "DMX = " + str(DMX))

plt.plot(frequencies_GHz, fitted_function_2, c="C1",
         label='$r(\\nu) = r_\mathrm{\infty} + r_\mathrm{2} \\nu^{-2}$')
plt.scatter(frequencies_GHz, corrected_toas, c="C3", label="TOAs - FD curve")
plt.legend()
plt.savefig("./fix_DM/corrected.png")
plt.show()
plt.close()
