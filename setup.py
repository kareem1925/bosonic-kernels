from setuptools import setup

version = "0.0.1"

requirements = [
    "numpy",
    "numba",
    "scikit-learn",
    "jupyter",
    "ipython",
    "notebook",
    "pandas",
    "seaborn"
    ]


info = {
    "name": "Bosonic-Kernels",
    "version": version,
    "packages": [
        "bosonic_kernels",
        "bosonic_kernels.classifier",
        "bosonic_kernels.kernels",
    ],
    "install_requires": requirements,
    "command_options": {
        "build_sphinx": {
            "version": ("setup.py", version),
            "release": ("setup.py", version),
        }
    },
}

classifiers = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: POSIX",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Physics",
]

setup(classifiers=classifiers, **(info))
