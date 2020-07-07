import glob
import numpy as np
from matplotlib import pyplot as plt

for filename in glob.glob("*.dat"):
    print(filename)
    name = filename.split(".")[0]
    data = np.loadtxt(filename, delimiter=",")
    size = int(np.sqrt(len(data)))
    data = data.reshape((size, size))
    fig, ax = plt.subplots(figsize=(5.12, 5.12))
    ax.imshow(data)
    plt.tick_params(
        bottom=False, left=False, right=False, top=False,
        labelbottom=False, labelleft=False, labelright=False, labeltop=False
    )
    plt.tight_layout()
    plt.savefig(name + ".png")
    plt.close()