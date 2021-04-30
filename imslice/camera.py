import numpy as np
from scipy.spatial.transform import Rotation
import logging

logger = logging.getLogger(__name__)


def unitize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def product(iterable):
    it = iter(iterable)
    val = next(it)
    for item in it:
        val *= item
    return val


def rot_unit(degrees=True):
    return ("radians", "degrees")[degrees]


class Camera:
    def __init__(self, width: int, height: int):
        """point, direction, orientation given as zyx"""
        self.width = int(width)
        self.height = int(height)
        half_width = self.width / 2
        half_height = self.height / 2

        self._mat = np.array(
            [
                [0, -half_height, -half_width],
                [0, -half_height, half_width],
                [0, half_height, -half_width],
            ],
            float,
        )

    @property
    def top_left(self):
        return self._mat[0]

    @property
    def top_right(self):
        return self._mat[1]

    @property
    def bottom_left(self):
        return self._mat[2]

    @property
    def bottom_right(self):
        vec = self.top_right - self.top_left
        return self.bottom_left + vec

    @property
    def real_width(self):
        return np.linalg.norm(self.top_right - self.top_left)

    @property
    def real_height(self):
        return np.linalg.norm(self.bottom_left - self.top_left)

    @property
    def aspect_ratio(self):
        return self.width / self.height

    @property
    def center(self):
        top_middle = (self.top_left + self.top_right) / 2
        half_height = (self.bottom_left - self.top_left) / 2
        return top_middle + half_height

    @property
    def normal(self):
        return -np.cross(
            self.top_right - self.top_left, self.bottom_left - self.top_left
        )

    def _empty_coords(self):
        return np.empty((3, self.height, self.width), float)

    def coords(self):
        """Return compatible with ``scipy.ndimage.map_coordinates``"""
        top_row = np.linspace(self.top_left, self.top_right, self.width, False, axis=1)
        bottom_row = np.linspace(
            self.bottom_left, self.bottom_right, self.width, False, axis=1
        )
        out = np.linspace(top_row, bottom_row, self.height, False, axis=1)
        if out is None:
            logger.critical("NoneType coords")
        return out

    @property
    def scale_level(self):
        return self.real_width / self.width

    def set_scale_level(self, scale_level: float):
        logger.debug("Setting scale level to %s", scale_level)
        half_width_vec = (
            self.width * scale_level * unitize(self.top_right - self.top_left) / 2
        )
        half_height_vec = (
            self.height * scale_level * unitize(self.bottom_left - self.top_left) / 2
        )

        center = self.center
        self._mat = np.array(
            [
                center - half_height_vec - half_width_vec,
                center - half_height_vec + half_width_vec,
                center + half_height_vec - half_width_vec,
            ]
        )
        return self

    def scale(self, factor: float):
        """Scale out by the given factor.

        e.g. scaling out by a factor of 2 fits twice as much on screen;
        everything halves in size.

        Parameters
        ----------
        factor : float
            >1 scales out, <1 scales in
        """
        logger.debug("Scaling by factor %s", factor)
        return self.set_scale_level(self.scale_level * factor)

    def shape_viewport(self, width=None, height=None):
        if width is None:
            width = self.width
            if height is None:
                return self
        if height is None:
            height = self.height
        logger.debug("Reshaping viewport to %sx%s", width, height)

        scale_level = self.scale_level
        half_width_vec = (
            width * scale_level * unitize(self.top_right - self.top_left) / 2
        )
        half_height_vec = (
            height * scale_level * unitize(self.bottom_left - self.top_left) / 2
        )

        center = self.center
        self._mat = np.array(
            [
                center - half_height_vec - half_width_vec,
                center - half_height_vec + half_width_vec,
                center + half_height_vec - half_width_vec,
            ]
        )

        return self

    def translate(self, diff: np.ndarray):
        logger.debug("Translating by %s", diff)
        self._mat += diff
        return self

    def translate_to(self, location: np.ndarray):
        logger.debug("Translating to %s", location)
        return self.translate(location - self.center)

    def dolly(self, forward=0, down=0, right=0, scale=True):
        if scale:
            scale_level = self.scale_level
            forward *= scale_level
            down *= scale_level
            right *= scale_level

        logger.debug("Dollying %s fwd, %s down, %s right", forward, down, right)

        return self.translate(
            unitize(self.normal) * forward
            + unitize(self.bottom_left - self.top_left) * down
            + unitize(self.top_right - self.top_left) * right
        )

    def rotate_to(self, z=0, y=0, x=0, degrees=True):
        """z, y, x are the axis around which to rotate.
        Rotations are applied in that order.
        """
        logger.debug("Rotating to z%s, y%s, x%s %s", z, y, z, rot_unit(degrees))
        half_height = self.real_height / 2
        half_width = self.real_width / 2

        center = self.center

        self._mat = np.array(
            [
                [0, -half_height, -half_width],
                [0, -half_height, half_width],
                [0, half_height, -half_width],
            ]
        )
        self.rotate(z, y, x, degrees)

        return self.translate_to(center)

    def rotate(self, roll=0, pitch=0, yaw=0, degrees=True):
        """Rotations applied in that order"""
        logger.debug(
            "Rotating by roll %s, pitch %s, yaw %s %s",
            roll,
            pitch,
            yaw,
            rot_unit(degrees),
        )

        if degrees:
            roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])

        center = self.center
        rotvecs = []
        tau = 2 * np.pi
        if roll % tau:
            rotvecs.append(unitize(self.normal) * roll)
        if pitch % tau:
            rotvecs.append(unitize(self.bottom_left - self.top_left) * pitch)
        if yaw % tau:
            rotvecs.append(unitize(self.top_right - self.top_left) * yaw)

        if not rotvecs:
            return self

        self.translate_to(np.array([0, 0, 0]))
        rotation = product(Rotation.from_rotvec(rv) for rv in rotvecs)
        self._mat = rotation.apply(self._mat)
        self.translate_to(center)
        return self
