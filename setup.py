import os, sys, pathlib
import re
from setuptools import find_packages, setup, Command 

with open("CHANGELOG.md", "r") as fh:
    lines = fh.read()
    versions = re.findall(r"\[(.*)\]", lines)
    versions = [tuple(map(float, version.split("."))) for version in versions]
    max_version = max(versions)

NAME = "benchmark"
DESCRIPTION = "disaggregation of time series"
VERSION = max_version
REQUIRES_PYTHON = ">=3.6"
REQUIRED = ["numpy", "typing", "scikit-learn"]
EXTRAS_REQUIRE = {'tests': ['pytest']}

setup(name = NAME,
      version = VERSION,
      description = DESCRIPTION,
      author_email = EMAIL,
      python_requires = REQUIRES_PYTHON,
      packages = find_packages(exclude = ["tests", "*.tests", "*.tests.*", "tests.*"])
      zip_safe = False,
      install_requires = REQUIRED,
      extras_requires = EXTRAS_REQUIRE,
      include_package_data = True)