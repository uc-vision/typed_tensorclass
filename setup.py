from setuptools import find_packages, setup

setup(name='typed_tensordict',
      version='0.0.1',
      packages=find_packages(),
      python_version='>=3.10',
      install_requires=[
          'jaxtyping'
      ])
