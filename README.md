# bosonic-kernels

## Working example

```
from bosonic_kernels.classifier import QSVM
from sklearn.datasets import make_circles

x, y = make_circles(100, random_state=0)

qsvm = QSVM(quantum_kernel='displaced_squeezed_angle', sq_mag=1., dis_mag=2)

qsvm.fit(x, y)
qsvm.score(x,y)
```

