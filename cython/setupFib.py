from distutils.core import setup
from Cython.Build import cythonize

setup(
        name = 'fibFun',
        ext_modules = cythonize("fib.pyx"),
        )
