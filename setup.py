from setuptools import setup, find_packages

setup(
    name="pytsc",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pyyaml",
    ],
)
