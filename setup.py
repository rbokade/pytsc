from setuptools import setup, find_packages

setup(
    name="pytsc",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pyyaml",
    ],
    include_package_data=True,
    package_data={
        "pytsc": [
            "scenarios/default/*.yaml",
            "scenarios/cityflow/*.yaml",
            "scenarios/sumo/*.yaml",
            "scenarios/test/*.yaml",
        ]
    },

)
