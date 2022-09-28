from abc import abstractmethod
import numpy as np
from numba import njit

from ._math import (
    displaced,
    displaced_squeezed_amp,
    displaced_squeezed_angle,
    squeezing,
)

_quantum_kernels = {
    "squeezing": squeezing,
    "displaced": displaced,
    "displaced_squeezed_angle": displaced_squeezed_angle,
    "displaced_squeezed_amp": displaced_squeezed_amp,
}


@njit(nogil=True, fastmath=True)
def _gram_matrix(x: np.ndarray, y: np.ndarray, fnum: callable, *args):
    gram = np.zeros((x.shape[0], y.shape[0]))

    #    np.fill_diagonal(gram, 1.0)
    for index in range(len(x)):
        temp = x[index]
        for index2, value2 in enumerate(y):
            gram[index, index2] = fnum(temp, value2, *args)

    return gram


class BaseQKernel:
    def __init__(
        self,
        kernel: str,
        sq_mag: float = 0.7,
        dis_mag: float = 1.0,
        sq_phi: float = 0.0,
        dis_phi: float = 0.0,
        dis_mag_2: float = None,
        sq_mag_2: float = None,
        sq_phi_2: float = None,
        dis_phi_2: float = None,
    ) -> None:

        if kernel not in _quantum_kernels:
            raise ValueError("The kernel " + kernel + " is not valid")

        self._kernel = kernel
        self.sq_mag = sq_mag
        self.sq_mag_2 = sq_mag if sq_mag_2 is None else sq_mag_2
        self.sq_phi = sq_phi
        self.dis_mag = dis_mag
        self.dis_mag_2 = dis_mag if dis_mag_2 is None else dis_mag_2
        self.dis_phi = dis_phi
        self.dis_phi_2 = dis_phi if dis_phi_2 is None else dis_phi_2
        self.sq_phi_2 = sq_phi if sq_phi_2 is None else sq_phi_2

    @abstractmethod
    def _compute_kernel():
        raise NotImplementedError


class Squeezing(BaseQKernel):
    def __init__(self, kernel: str, sq_mag: float = 1.0):
        self.sq_mag = sq_mag
        self._kernel = _quantum_kernels[kernel]

    def _compute_kernel(self, x: np.ndarray, y: np.ndarray = None):
        return _gram_matrix(x, y, self._kernel, self.sq_mag)


class Displaced(BaseQKernel):
    def __init__(self, kernel: str, dis_mag: float = 1.0):
        self.dis_mag = dis_mag
        self._kernel = _quantum_kernels[kernel]

    def _compute_kernel(self, x: np.ndarray, y: np.ndarray = None):
        return _gram_matrix(x, y, self._kernel, self.dis_mag)


class DisplacedSqueezedAngle(BaseQKernel):
    def __init__(
        self,
        kernel: str,
        sq_mag: float = 0.7,
        dis_mag: float = 1.0,
        sq_phi: float = 0.0,
        dis_phi: float = 0.0,
        dis_mag_2: float = None,
        sq_mag_2: float = None,
        sq_phi_2: float = None,
        dis_phi_2: float = None,
    ):

        self.sq_mag = sq_mag
        self.sq_phi_2 = sq_phi if sq_phi_2 is not None else sq_phi_2
        self.sq_mag_2 = sq_mag if sq_mag_2 is None else sq_mag_2
        self.sq_phi = sq_phi
        self.dis_mag = dis_mag
        self.dis_mag_2 = dis_mag if dis_mag_2 is None else dis_mag
        self.dis_phi = dis_phi
        self.dis_phi_2 = dis_phi if dis_phi_2 is None else dis_phi
        self._kernel = _quantum_kernels[kernel]

    def _compute_kernel(self, x: np.ndarray, y: np.ndarray = None):
        return _gram_matrix(
            x, y, self._kernel, self.sq_mag, self.dis_mag, self.sq_mag_2, self.dis_mag_2
        )


class DisplacedSqueezedAmp(BaseQKernel):
    def __init__(
        self,
        kernel: str,
        sq_mag: float = 0.7,
        dis_mag: float = 1.0,
        sq_phi: float = 0.0,
        dis_phi: float = 0.0,
        dis_mag_2: float = None,
        sq_mag_2: float = None,
        sq_phi_2: float = None,
        dis_phi_2: float = None,
    ):

        self.sq_mag = sq_mag
        self.sq_mag_2 = sq_mag if sq_mag_2 is None else sq_mag_2
        self.sq_phi = sq_phi
        self.dis_mag = dis_mag
        self.dis_mag_2 = dis_mag if dis_mag_2 is None else dis_mag
        self.dis_phi = dis_phi
        self.sq_phi_2 = sq_phi if sq_phi_2 is None else sq_phi_2
        self.dis_phi_2 = dis_phi if dis_phi_2 is None else dis_phi_2
        self._kernel = _quantum_kernels[kernel]

    def _compute_kernel(self, x: np.ndarray, y: np.ndarray = None):
        return _gram_matrix(
            x,
            y,
            self._kernel,
            self.sq_phi,
            self.dis_phi,
            self.sq_phi_2,
            self.dis_phi_2,
        )
