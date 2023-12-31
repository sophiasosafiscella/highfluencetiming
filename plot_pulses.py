import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams['text.usetex'] = True

sns.set_style('ticks')
sns.set_context("paper", font_scale=1.5)

dirs = ["0.0_sigma", "0.5_sigma", "1.0_sigma", "1.5_sigma", "2.0_sigma", "2.5_sigma", "3.0_sigma", "3.5_sigma"]

fig, axs = plt.subplots(4, 2, figsize=(8, 10),
                        gridspec_kw = {'wspace':0, 'hspace':0}, sharex=True, sharey=True)

# set labels
plt.setp(axs[-1, :], xlabel='Phase Bins')
plt.setp(axs[:, 0], ylabel='Intensity')

axs = axs.flatten()

for i, dir_name in enumerate(dirs):

    data = pd.read_pickle("./results/820_band/" + dir_name + "/unnormalized_data.pkl")
    axs[i].plot(data.iloc[14].to_numpy())
    axs[i].text(0.5, 0.95, dir_name[:3] + " $\\sigma_{\\mathrm{med}}$",
                horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes)

plt.tight_layout()
plt.savefig("./plots/examples.pdf")
plt.show()
