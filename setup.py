#!/usr/bin/env python
# -*- coding: utf-8 -*-

"The setup script."

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as requirements_file:
    requirements = requirements_file.read().split("\n")

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest",
]

setup(
    author="Tom Martensen",
    author_email="mail@tommartensen.de",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    description=(
        "Methods that bridge the gap between FIBER and MORPHER."
        "And generate some ML pipelines."
    ),
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords="robotehr",
    name="robotehr",
    packages=find_packages(include=["robotehr", "robotehr.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://gitlab.hpi.de/tom.martensen/robotehr",
    version="0.0.1",
    zip_safe=False,
)
