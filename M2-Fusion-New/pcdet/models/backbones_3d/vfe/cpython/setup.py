from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension('slide_window',
              ['slide_window.pyx'])
]

setup(
  name = 'slide_window app',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)