import logging
from typing import Optional

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
        channel_dim: Optional[int] = None,
        interp_order=0,
        extrap_mode="constant",
        extrap_cval=0,
        camera_on=True,
    ):
        self.array = array
        self.camera = Camera(width, height)
        self.channel_dim = channel_dim
        self.interp_order = interp_order
        self.extrap_mode = extrap_mode
        self.extrap_cval = extrap_cval
        self.camera_on = camera_on

        self.frame_shape = (self.camera.height, self.camera.width)
        self._rolled = self.array
        if self.channel_dim is not None:
            self.frame_shape += (self.array.shape[self.channel_dim],)
            self._rolled = np.moveaxis(self.array, self.channel_dim, 0)

            self._channel_slice = [slice(None)] * len(self.array.shape)

    def _empty_frame(self):
        return np.empty(self.frame_shape, self.array.dtype)

    def _snap_single(self, array, coords, output=None):
        return map_coordinates(
            array,
            coords,
            output=output,
            order=self.interp_order,
            mode=self.extrap_mode,
            cval=self.extrap_cval,
        )

    def _channel_slices(self):
        for i in range(self.array.shape[self.channel_dim]):
            self._channel_slice[self.channel_dim] = i
            yield tuple(self._channel_slice)

    def _snap_multi(self, coords, output=None):
        if output is None:
            output = self._empty_frame()

        for slice in self._channel_slices():
            self._snap_single(self.array[slice], coords, output[slice])
        return output

    def snap(self, output=None):
        # if output is None or output.shape != (self.camera.height, self.camera.width):
        #     output = self._empty()
        if not self.camera_on:
            return None
        coords = self.camera.coords()
        if self.channel_dim is None:
            return self._snap_single(self.array, coords, output)
        return self._snap_multi(coords, output)
