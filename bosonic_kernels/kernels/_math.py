import numpy as np
from numba import njit


@njit(nogil=True, fastmath=True)
def displaced_squeezed_angle(
    x1: np.ndarray,
    x2: np.ndarray,
    sq_mag: float = 0.7,
    dis_mag: float = 0.7,
    sq_mag_2: float = None,
    dis_mag_2: float = None,
):
    if dis_mag_2 is None:
        dis_2 = dis_mag
    else:
        dis_2 = dis_mag_2

    if sq_mag_2 is None:
        mag2 = sq_mag
    else:
        mag2 = sq_mag_2

    dis_1 = dis_mag
    mag1 = sq_mag

    sq_mag_cosh_1 = np.cosh(-mag1)
    sq_mag_cosh_2 = np.cosh(-mag2)

    z1 = dis_1 * np.exp(1j * x1)
    z1_conj = np.conj(z1)

    z2 = dis_2 * np.exp(1j * x2)
    z2_conj = np.conj(z2)

    sigma = (sq_mag_cosh_2 * sq_mag_cosh_1) - (
        np.exp(1j * (x2 - x1)) * np.sinh(-mag2) * np.sinh(-mag1)
    )
    eta_21 = ((z2 - z1) * sq_mag_cosh_2) - (
        (z2_conj - z1_conj) * np.exp(1j * x2) * np.sinh(-mag2)
    )

    eta_12 = ((z1 - z2) * sq_mag_cosh_1) - (
        (z1_conj - z2_conj) * np.exp(1j * x1) * np.sinh(-mag1)
    )

    term_1 = np.sqrt(sigma)
    term_2 = (eta_21 * np.conj(eta_12)) / (2 * sigma)

    term_3 = 0.5 * ((z2 * z1_conj) - (z2_conj * z1))

    return np.prod(np.square(np.abs(np.exp(term_2 + term_3) * (1 / term_1))))


@njit(nogil=True, fastmath=True)
def displaced_squeezed_amp(
    x1: np.ndarray,
    x2: np.ndarray,
    sq_phi: float = 0.7,
    dis_phi: float = 0.7,
    sq_phi_2: float = None,
    dis_phi_2: float = None,
):
    dis_1 = x1
    mag1 = x1
    dis_2 = x2
    mag2 = x2

    if sq_phi_2 is not None:
        sq_phi_2 = sq_phi

    if dis_phi_2 is None:
        dis_phi_2 = dis_phi

    sq_mag_cosh_1 = np.cosh(-mag1)
    sq_mag_cosh_2 = np.cosh(-mag2)

    z1 = dis_1 * np.exp(1j * dis_phi)
    z1_conj = np.conj(z1)

    z2 = dis_2 * np.exp(1j * dis_phi_2)
    z2_conj = np.conj(z2)

    sigma = (sq_mag_cosh_2 * sq_mag_cosh_1) - (
        np.exp(1j * (dis_phi_2 - dis_phi)) * np.sinh(-mag2) * np.sinh(-mag1)
    )
    eta_21 = ((z2 - z1) * sq_mag_cosh_2) - (
        (z2_conj - z1_conj) * np.exp(1j * dis_phi_2) * np.sinh(-mag2)
    )

    eta_12 = ((z1 - z2) * sq_mag_cosh_1) - (
        (z1_conj - z2_conj) * np.exp(1j * dis_phi) * np.sinh(-mag1)
    )

    term_1 = np.sqrt(sigma)
    term_2 = (eta_21 * np.conj(eta_12)) / (2 * sigma)

    term_3 = 0.5 * ((z2 * z1_conj) - (z2_conj * z1))

    return np.prod(np.square(np.abs(np.exp(term_2 + term_3) * (1 / term_1))))


@njit(nogil=True, fastmath=True)
def displaced(x1: np.ndarray, x2: np.ndarray, dis_mag: float = 1.0):

    z1 = dis_mag * np.exp(1j * x1)
    z2 = dis_mag * np.exp(1j * x2)

    return np.prod(np.exp(-np.abs(z1 - z2)))


@njit(nogil=True)
def squeezing(x1: np.ndarray, x2: np.ndarray, sq_mag: float):
    sigma = np.cosh(sq_mag) ** 2 - (np.exp(1j * (x2 - x1)) * np.sinh(sq_mag) ** 2)
    return np.prod(np.square(np.abs(1 / np.sqrt(sigma))))
