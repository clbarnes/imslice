from abc import ABC, abstractmethod
from typing import Iterable, Optional, Dict, Any, List
from itertools import zip_longest

import numpy as np
import strictyaml as syml

from .scene import Scene


def first_lower(s: str):
    return s[0].lower() + s[1:]


def syml_union(*args):
    if len(args) == 0:
        return None
    union = args[0]
    for arg in args[1:]:
        union |= arg
    return union


class Instruction(ABC):
    action: str

    def __init__(self, frames: int = 1) -> None:
        self.frames = int(frames)

    @abstractmethod
    def __call__(self, scene: Scene, silent=False) -> Iterable[Optional[np.ndarray]]:
        pass

    def __len__(self):
        return self.frames

    @classmethod
    def _check_action(cls, jso: Dict[str, Any]):
        action = jso.pop("action", None)
        if action is not None:
            exp_action = instruction_action(cls)
            if action != exp_action:
                raise ValueError(
                    "Mismatched names; expected {exp_name}, expected {name}"
                )

    @classmethod
    def from_jso(cls, jso: Dict[str, Any]):
        cls._check_action(jso)
        return cls(**jso)

    @classmethod
    def _schema_base(cls):
        return {"action": syml.Enum([instruction_action(cls)])}

    @classmethod
    def schema(cls):
        return syml.Map(
            {
                **cls._schema_base(),
                syml.Optional("frames", 1): syml.Int,
            }
        )


class Wait(Instruction):
    action = "wait"

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
    action = "set_scale_level"

    def __init__(self, scale: float, frames: int = 1):
        super().__init__(frames)
        self.scale = float(scale)

    def __call__(self, scene: Scene, silent=False) -> Iterable[Optional[np.ndarray]]:
        steps = np.linspace(scene.camera.scale_level, self.scale, self.frames + 1)[1:]
        for s in steps:
            scene.camera.set_scale_level(s)
            if silent:
                yield None
            else:
                yield scene.snap()

    @classmethod
    def schema(cls):
        return syml.Map(
            {
                **cls._schema_base(),
                "scale": syml.Float(),
                syml.Optional("frames", 1): syml.Int(),
            }
        )


class Scale(Instruction):
    action = "scale"

    def __init__(self, factor: float, frames: int = 1):
        super().__init__(frames)
        self.factor = float(factor)

    def __call__(self, scene: Scene, silent=False) -> Iterable[Optional[np.ndarray]]:
        final_scale = scene.camera.scale_level * self.factor
        yield from SetScaleLevel(final_scale, self.frames)(scene, silent)

    @classmethod
    def schema(cls):
        return syml.Map(
            {
                **cls._schema_base(),
                "factor": syml.Float(),
                syml.Optional("frames", 1): syml.Int(),
            }
        )


class Translate(Instruction):
    action = "translate"

    def __init__(self, diff: np.ndarray, frames: int = 1) -> None:
        super().__init__(frames=frames)
        self.diff = np.asarray(diff, float)

    def __call__(self, scene: Scene, silent=False) -> Iterable[Optional[np.ndarray]]:
        step = self.diff / self.frames
        for _ in range(self.frames):
            scene.camera.translate(step)
            if silent:
                yield None
            else:
                yield scene.snap()

    @classmethod
    def schema(cls):
        return syml.Map(
            {
                **cls._schema_base(),
                "diff": syml.FixedSeq([syml.Float()] * 3),
                syml.Optional("frames", 1): syml.Int(),
            }
        )


class TranslateTo(Instruction):
    action = "translate_to"

    def __init__(self, location: np.ndarray, frames: int = 1) -> None:
        super().__init__(frames=frames)
        self.location = np.asarray(location, float)

    def __call__(self, scene: Scene, silent=False) -> Iterable[Optional[np.ndarray]]:
        diff = self.location - scene.camera.center
        yield from Translate(diff, self.frames)(scene, silent)

    @classmethod
    def schema(cls):
        return syml.Map(
            {
                **cls._schema_base(),
                "location": syml.FixedSeq([syml.Float()] * 3),
                syml.Optional("frames", 1): syml.Int(),
            }
        )


class Dolly(Instruction):
    action = "dolly"

    def __init__(self, forward=0, down=0, right=0, scale=True, frames: int = 1):
        super().__init__(frames=frames)
        self.forward = float(forward)
        self.down = float(down)
        self.right = float(right)
        self.scale = bool(scale)

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

    @classmethod
    def schema(cls):
        return syml.Map(
            {
                **cls._schema_base(),
                syml.Optional("forward", 0): syml.Float(),
                syml.Optional("down", 0): syml.Float(),
                syml.Optional("right", 0): syml.Float(),
                syml.Optional("right", 0): syml.Float(),
                syml.Optional("frames", 1): syml.Int(),
            }
        )


class Rotate(Instruction):
    action = "rotate"

    def __init__(self, roll=0, pitch=0, yaw=0, degrees=True, frames: int = 1) -> None:
        super().__init__(frames=frames)
        self.roll = float(roll)
        self.pitch = float(pitch)
        self.yaw = float(yaw)
        self.degrees = bool(degrees)

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

    @classmethod
    def schema(cls):
        return syml.Map(
            {
                **cls._schema_base(),
                syml.Optional("roll", 0): syml.Float(),
                syml.Optional("pitch", 0): syml.Float(),
                syml.Optional("yaw", 0): syml.Float(),
                syml.Optional("degrees", True): syml.Bool(),
                syml.Optional("frames", 1): syml.Int(),
            }
        )


EXTRAP_MODES = ["constant", "reflect", "nearest", "mirror", "wrap"]


def or_fn(obj, fn):
    if obj is None:
        return None
    return fn(obj)


class ChangeInterp(Instruction):
    action = "change_interp"

    def __init__(self, interp_order=None, extrap_mode=None, extrap_cval=None):
        super().__init__(frames=1)
        self.interp_order = or_fn(interp_order, int)
        self.extrap_mode = extrap_mode
        self.extrap_cval = or_fn(extrap_cval, float)

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

    @classmethod
    def schema(cls):
        return syml.Map(
            {
                **cls._schema_base(),
                syml.Optional("interp_order", None): syml.Enum(
                    [str(x) for x in range(6)]
                ),
                syml.Optional("extrap_mode", None): syml.Enum(EXTRAP_MODES),
                syml.Optional("extrap_cval", None): syml.Float() | syml.Int(),
            }
        )


class Compound(Instruction):
    action = "compound"

    def __init__(self, instructions) -> None:
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

    @classmethod
    def from_jso(cls, jso: Dict[str, Any]):
        cls._check_action(jso)
        return cls([from_jso(i) for i in jso["instructions"]])

    @classmethod
    def schema(cls):
        return syml.Map(
            {
                **cls._schema_base(),
                "instructions": instruction_schema,
            }
        )


class Sequence(Instruction):
    action = "sequence"

    def __init__(self, instructions) -> None:
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

    @classmethod
    def from_jso(cls, jso: Dict[str, Any]):
        cls._check_action(jso)
        return cls([from_jso(i) for i in jso["instructions"]])

    @classmethod
    def schema(cls):
        return syml.Map(
            {
                **cls._schema_base(),
                "instructions": instruction_schema,
            }
        )


SCALAR_INSTRUCTIONS = [
    Wait,
    SetScaleLevel,
    Scale,
    Translate,
    TranslateTo,
    Dolly,
    Rotate,
    ChangeInterp,
]

MULTI_INSTRUCTIONS = [Compound, Sequence]

INSTRUCTIONS = SCALAR_INSTRUCTIONS + MULTI_INSTRUCTIONS


def from_jso(jso: Dict[str, Any]):
    ins_class = INSTRUCTIONS[jso["action"]]
    return ins_class.from_jso(jso)


SCHEMA = syml.Map({
    "instructions": syml.Seq(syml_union(*[c for c in INSTRUCTIONS]))
})

multi_actions = {i.action for i in MULTI_INSTRUCTIONS}

instruction_schema = syml_union(*[i.schema() for i in SCALAR_INSTRUCTIONS], syml.Any)


def rec_validate(item: syml.YAML):
    if item["action"] in multi_actions:
        for inner in item["instructions"]:
            inner.revalidate(instruction_schema)
            rec_validate(inner)


def parse(f):
    try:
        s = f.read()
    except (AttributeError, TypeError):
        s = f

    loaded = syml.load(s, SCHEMA)
    out = []
    for item in loaded["instructions"]:
        rec_validate(item)
        out.append(from_jso(item.data))
    return Sequence(out)
