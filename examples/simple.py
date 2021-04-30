from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib import pyplot as plt

from imslice import Scene

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_data():
    d = np.load(DATA_DIR / "stent.npz")
    arr = d["arr_0"].astype(float)
    return arr / arr.max()


def load_slice():
    arr = load_data()
    z = arr.shape[0] // 2
    return arr[z : z + 1, :, :]


# %%

arr = load_data()
scene = Scene(arr, 200, 150)
scene.camera.translate_to(np.array(arr.shape) / 2)
snaps = dict()
snaps["orig"] = scene.snap()
scene.camera.rotate(90)
snaps["roll90"] = scene.snap()
roll90 = scene.snap()
scene.camera.rotate(0, 90)
snaps["pitch90"] = scene.snap()
scene.camera.rotate(0, 0, 90)
snaps["yaw90"] = scene.snap()

# %%


fig, ax_arr = plt.subplots(2, 2)
ax: Axes
for idx, ((name, snap), ax) in enumerate(zip(snaps.items(), ax_arr.flatten())):
    ax.imshow(snap)
    ax.set_title(f"{idx}: {name}")

plt.show()
