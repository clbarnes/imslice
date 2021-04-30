import logging

from scipy.ndimage import map_coordinates
import numpy as np

from .camera import Camera


logger = logging.getLogger(__name__)


class Scene:
    def __init__(
        self,
        array: np.ndarray,
        width: int,
        height: int,
        interp_order=0,
        extrap_mode="constant",
        extrap_cval=0,
    ):
        self.array = array
        self.camera = Camera(width, height)
        self.interp_order = interp_order
        self.extrap_mode = extrap_mode
        self.extrap_cval = extrap_cval

    def _empty(self):
        return np.empty((self.camera.height, self.camera.width), self.array.dtype)

    def snap(self, output=None):
        # if output is None or output.shape != (self.camera.height, self.camera.width):
        #     output = self._empty()
        coords = self.camera.coords()
        return map_coordinates(
            self.array,
            coords,
            output=output,
            order=self.interp_order,
            mode=self.extrap_mode,
            cval=self.extrap_cval,
        )
