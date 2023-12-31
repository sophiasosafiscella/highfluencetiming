import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dirs = ["0.0_sigma", "0.5_sigma", "1.0_sigma", "1.5_sigma", "2.0_sigma", "2.5_sigma", "3.0_sigma", "3.5_sigma", "4.0_sigma"]

for dir in dirs:
    data = pd.read_pickle("./results/820_band/" + dir + "/unnormalized_data.pkl")
    plt.close()
    plt.title(dir.replace("_", " "))
    data.iloc[14].plot()
    plt.tight_layout()
    plt.savefig("./results/820_band/" + dir + "/example.png")
    plt.show()
