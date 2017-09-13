from setuptools import setup

PACKAGES = [
    'py_ml_utils',
    'py_ml_utils.test'
]

setup(name='py_ml_utils',
      version='0.1',
      description='Python Utilities for Machine Learning',
      author='Olivier Grellier',
      author_email='goldentom42@gmail.com',
      license='Apache 2.0',
      install_requires=['numpy>=1.13.1', 'scipy>=0.19.1', 'scikit_learn>=0.19', 'pandas>=0.20.2'],
      packages=PACKAGES,
      zip_safe=False)
