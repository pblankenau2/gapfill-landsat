#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = []

setup_requirements = ["pytest-runner"]

test_requirements = ["pytest>=3"]

setup(
    author="Philip Blankenau",
    author_email="philip.blankenau@idwr.idaho.gov",
    python_requires=">=3.5",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Tools for filling missing data in satellite images caused by sensor malfunctions or masked out clouds.",
    install_requires=requirements,
    license="None",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="cloud_filler",
    name="cloud_filler",
    packages=find_packages(include=["cloud_filler", "cloud_filler.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/pblankenau2/cloud_filler",
    version="0.1.0",
    zip_safe=False,
)
