from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extensions = [Extension("node", ["node.pyx"])]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()],
)