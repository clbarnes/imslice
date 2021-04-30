from abc import ABC, abstractmethod
from typing import Iterable, Optional
from itertools import zip_longest

import numpy as np

from .scene import Scene


class Instruction(ABC):
    def __init__(self, frames: int = 1) -> None:
        self.frames = frames

    @abstractmethod
    def __call__(self, scene: Scene, silent=False) -> Iterable[Optional[np.ndarray]]:
        pass

    def __len__(self):
        return self.frames


class Wait(Instruction):
    def __init__(self, frames: int = 1) -> None:
        super().__init__(frames=frames)

    def __call__(self, scene: Scene, silent=False) -> Iterable[Optional[np.ndarray]]:
        out = scene.snap()
        for _ in range(self.frames):
            if silent:
                yield None
            else:
                yield out


class SetScaleLevel(Instruction):
    def __init__(self, scale: float, frames: int = 1):
        super().__init__(frames)
        self.scale = scale

    def __call__(self, scene: Scene, silent=False) -> Iterable[Optional[np.ndarray]]:
        steps = np.linspace(scene.camera.scale_level, self.scale, self.frames + 1)[1:]
        for s in steps:
            scene.camera.set_scale_level(s)
            if silent:
                yield None
            else:
                yield scene.snap()


class Scale(Instruction):
    def __init__(self, factor: float, frames: int = 1):
        super().__init__(frames)
        self.factor = factor

    def __call__(self, scene: Scene, silent=False) -> Iterable[Optional[np.ndarray]]:
        final_scale = scene.camera.scale_level * self.factor
        yield from SetScaleLevel(final_scale, self.frames)(scene, silent)


class Translate(Instruction):
    def __init__(self, diff: np.ndarray, frames: int = 1) -> None:
        super().__init__(frames=frames)
        self.diff = diff

    def __call__(self, scene: Scene, silent=False) -> Iterable[Optional[np.ndarray]]:
        step = self.diff / self.frames
        for _ in range(self.frames):
            scene.camera.translate(step)
            if silent:
                yield None
            else:
                yield scene.snap()


class TranslateTo(Instruction):
    def __init__(self, location: np.ndarray, frames: int = 1) -> None:
        super().__init__(frames=frames)
        self.location = location

    def __call__(self, scene: Scene, silent=False) -> Iterable[Optional[np.ndarray]]:
        diff = self.location - scene.camera.center
        yield from Translate(diff, self.frames)(scene, silent)


class Dolly(Instruction):
    def __init__(self, forward=0, down=0, right=0, scale=True, frames: int = 1):
        super().__init__(frames=frames)
        self.forward = forward
        self.down = down
        self.right = right
        self.scale = scale

    def __call__(self, scene: Scene, silent=False) -> Iterable[Optional[np.ndarray]]:
        step_args = [
            self.forward / self.frames,
            self.down / self.frames,
            self.right / self.frames,
            self.scale,
        ]
        for _ in range(self.frames):
            scene.camera.dolly(*step_args)
            if silent:
                yield None
            else:
                yield scene.snap()


class Rotate(Instruction):
    def __init__(self, roll=0, pitch=0, yaw=0, degrees=True, frames: int = 1) -> None:
        super().__init__(frames=frames)
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.degrees = degrees

    def __call__(self, scene: Scene, silent=False) -> Iterable[Optional[np.ndarray]]:
        step_args = [
            self.roll / self.frames,
            self.pitch / self.frames,
            self.yaw / self.frames,
            self.degrees,
        ]
        for _ in range(self.frames):
            scene.camera.rotate(*step_args)
            if silent:
                yield None
            else:
                yield scene.snap()


class ChangeInterp(Instruction):
    def __init__(self, interp_order=None, extrap_mode=None, extrap_cval=None):
        super().__init__(frames=1)
        self.interp_order = interp_order
        self.extrap_mode = extrap_mode
        self.extrap_cval = extrap_cval

    def __call__(self, scene: Scene, silent=False) -> Iterable[Optional[np.ndarray]]:
        if self.interp_order is not None:
            scene.interp_order = self.interp_order
        if self.extrap_mode is not None:
            scene.extrap_mode = self.extrap_mode
        if self.extrap_cval is not None:
            scene.extrap_cval = self.extrap_cval

        if silent:
            yield None
        else:
            yield scene.snap()


class Compound(Instruction):
    def __init__(self, *instructions) -> None:
        if instructions:
            frames = max(len(i) for i in instructions)
        else:
            frames = 0
        super().__init__(frames)
        self.instructions = instructions

    def __call__(self, scene: Scene, silent=False) -> Iterable[Optional[np.ndarray]]:
        if not self.instructions:
            return
        iters = [i(scene, True) for i in self.instructions[:-1]]
        iters.append(self.instructions[-1](scene, silent))
        for *_, snap in zip_longest(*iters):
            if silent:
                yield None
            else:
                yield snap


class Sequence(Instruction):
    def __init__(self, *instructions) -> None:
        self.instructions = []
        for i in instructions:
            if isinstance(i, Sequence):
                self.instructions.extend(i.instructions)
            else:
                self.instructions.append(i)
        super().__init__(sum(len(i) for i in instructions))

    def __call__(self, scene: Scene, silent=False) -> Iterable[Optional[np.ndarray]]:
        for i in self.instructions:
            yield from i(scene, silent)
