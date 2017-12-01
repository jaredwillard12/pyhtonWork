from distutils.core import setup
from Cython.Build import cythonize

setup(
        name = 'Primes',
        ext_modules = cythonize("primes.pyx"),
        )
