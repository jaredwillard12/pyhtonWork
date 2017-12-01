from distutils.core import setup
from Cython.Build import cythonize

setup(
        name = 'KNN',
        ext_modules = cythonize("k_nearest_neighbor2.pyx"),
        )
