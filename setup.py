# setup.py

from setuptools import find_packages, setup

setup(
    name="diffundo",
    version="0.1.0",
    description="Tools for working with diffusion maps and geometric harmonics",
    author="Christopher Chaney",
    author_email="christopher.chaney@utsouthwestern.edu",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "tqdm",
        "pandas",
        "scanpy",
        "anndata",
    ],
    python_requires=">=3.8",
)
