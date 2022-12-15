from setuptools import setup
from Cython.Build import cythonize

setup(
    name='demons',
    ext_modules=cythonize([
        '*.py',
        '**/*.py',
    ], language_level=3),
    extra_compile_args=["-O3"],
)
