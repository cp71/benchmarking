import os, sys, pathlib
import re
from setuptools import find_packages, setup, Command 

with open("CHANGELOG.md", "r") as fh:
    temp = fh.read()
    versions = re.findall(r"\[(.*)\]", temp)
    versions = [tuple(map(float, version.split("."))) for version in versions]
    max_version = max(versions)
    max_version = ".".join(list(map(lambda x: str(int(x)),max_version)))

NAME = "benchmark"
DESCRIPTION = "disaggregation of time series"
VERSION = max_version
REQUIRES_PYTHON = ">=3.6"
REQUIRED = ["numpy", "typing", "scikit-learn"]

setup(name = NAME,
      version = VERSION,
      description = DESCRIPTION,
      python_requires = REQUIRES_PYTHON,
      packages = find_packages(exclude = ["tests", "*.tests", "*.tests.*", "tests.*"]),
      install_requires = REQUIRED,
      include_package_data = True)