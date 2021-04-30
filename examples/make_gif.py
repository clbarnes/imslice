from pathlib import Path
import logging

import numpy as np
from tqdm import tqdm
import imageio

from imslice import Scene, animation as anim

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

logging.basicConfig(level=logging.INFO)


def load_data():
    d = np.load(DATA_DIR / "stent.npz")
    arr = d["arr_0"].astype(float)
    return arr / arr.max()


arr = load_data()
scene = Scene(arr, 512, 512)
scene.camera.translate_to([0, 64, 64])
scene.camera.set_scale_level(0.25)

FPS = 30

seq = anim.Sequence(
    anim.Wait(1),
    anim.Dolly(arr.shape[0], frames=2 * FPS),
    anim.Compound(
        anim.Dolly(-arr.shape[0] / 2, frames=2 * FPS),
        anim.Rotate(360, frames=2 * FPS),
    ),
    anim.Rotate(pitch=90, frames=2 * FPS),
    anim.Rotate(yaw=90, frames=2 * FPS),
    anim.Scale(0.5, FPS),
    anim.ChangeInterp(interp_order=1),
    anim.Wait(FPS // 2),
    anim.ChangeInterp(interp_order=2),
    anim.Wait(FPS // 2),
    anim.ChangeInterp(interp_order=3),
    anim.Wait(FPS // 2),
)

frames = []
for frame in tqdm(seq(scene), "processing", total=len(seq)):
    try:
        frames.append(frame)
    except Exception:
        break


def to_gif(fpath, frames):
    stacked = np.stack(frames, 0)
    stacked -= stacked.min()
    stacked /= stacked.max()
    stacked *= 255
    u8 = stacked.astype(np.uint8)
    imageio.mimwrite(fpath, u8)
