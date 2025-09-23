from setuptools import setup, find_packages
from os import path
import sys

from src.version import __version__  # fix later


def _requirements_from_text(filename):
    return open(filename).read().splitlines()


here = path.abspath(path.dirname(__file__))
sys.path.insert(0, path.join(here, "src"))  # fix later

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


# setup offline-prompts
setup(
    name="OfflinePrompts",
    version=__version__,
    description="Benchmark Suites for Prompt-guided Language Personalization (Python)",
    url="https://github.com/aiueola/offline-prompts",
    author="Haruka Kiyohara, Daniel Yiming Cao, Yuta Saito, Thorsten Joachims",
    author_email="hk844@cornell.edu",
    keywords=[
        "off-policy evaluation",
        "off-policy learning",
        "prompt tuning",
        "large language models",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=_requirements_from_text("requirements.txt"),
    license="Apache License 2.0",
    packages=find_packages(
        exclude=[".github", "docs", "examples", "tests"],
    ),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache Software License",
    ],
)
