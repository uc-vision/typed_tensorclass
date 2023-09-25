from setuptools import find_packages, setup

setup(name='typed_tensorclass',
      version='0.0.2',
      packages=find_packages(),
      python_version='>=3.10',
      install_requires=[
          'jaxtyping'
      ])
