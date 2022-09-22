from sklearn.svm import SVC

from ..kernels.q_kernel import (
    Squeezing,
    Displaced,
    DisplacedSqueezedAmp,
    DisplacedSqueezedAngle,
)

_feature_maps = {
    "squeezing": Squeezing,
    "displaced": Displaced,
    "displaced_squeezed_amp": DisplacedSqueezedAmp,
    "displaced_squeezed_angle": DisplacedSqueezedAngle,
}


class QSVM(SVC):
    def __init__(
        self,
        quantum_kernel: str,
        sq_mag: float = 1.0,
        dis_mag: float = 1.0,
        sq_phi: float = 0.0,
        dis_phi: float = 0.0,
        dis_mag_2: float = None,
        sq_mag_2: float = None,
        sq_phi_2: float = None,
        dis_phi_2: float = None,
        **kwargs
    ):

        self.quantum_kernel = quantum_kernel
        self.sq_mag = sq_mag
        self.sq_mag_2 = sq_mag if sq_mag_2 is None else sq_mag_2
        self.sq_phi = sq_phi
        self.dis_mag = dis_mag
        self.dis_mag_2 = dis_mag if dis_mag_2 is None else dis_mag_2
        self.dis_phi = dis_phi
        self.dis_phi_2 = dis_phi if dis_phi_2 is None else dis_phi_2
        self.sq_phi_2 = sq_phi if sq_phi_2 is None else sq_phi_2

        qkernel = _feature_maps[self.quantum_kernel]

        if self.quantum_kernel == "squeezing":
            self.kernel = qkernel(self.quantum_kernel, sq_mag=sq_mag)._compute_kernel

        elif self.quantum_kernel == "displaced":
            self.kernel = qkernel(self.quantum_kernel, dis_mag=dis_mag)._compute_kernel

        elif self.quantum_kernel == "displaced_squeezed_amp":
            self.kernel = qkernel(
                self.quantum_kernel,
                sq_mag=sq_mag,
                dis_mag=dis_mag,
                sq_phi=sq_phi,
                dis_phi=dis_phi,
                dis_mag_2=dis_mag_2,
                sq_mag_2=sq_mag_2,
                sq_phi_2=sq_phi_2,
                dis_phi_2=dis_phi_2,
            )._compute_kernel
        else:
            self.kernel = qkernel(
                self.quantum_kernel,
                sq_mag=sq_mag,
                dis_mag=dis_mag,
                sq_phi=sq_phi,
                dis_phi=dis_phi,
                dis_mag_2=dis_mag_2,
                sq_mag_2=sq_mag_2,
                sq_phi_2=sq_phi_2,
                dis_phi_2=dis_phi_2,
            )._compute_kernel

        super().__init__(
            kernel=self.kernel,
        )

    @classmethod
    def _get_param_names(cls):
        names = SVC._get_param_names()
        names.remove("kernel")
        return sorted(
            names
            + ["quantum_kernel"]
            + ["sq_phi_2"]
            + ["sq_phi"]
            + ["dis_phi_2"]
            + ["dis_phi"]
            + ["dis_mag_2"]
            + ["dis_mag"]
            + ["sq_mag_2"]
            + ["sq_mag"]
        )
