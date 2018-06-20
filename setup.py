from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
        name="pyunfolding",
        version="0.0.1",
        description="unfolding framework for python",
        url="http://github.com/thoinka/pyunfolding",
        author="Tobias Hoinka",
        author_email="tobias.hoinka@tu-dortmund.de",
        license="MIT",
        classifiers=[
            "Development Status :: 1 - Alpha",
            "Intended Audience :: Developers",
            "Topic :: Software Development :: Build Tools",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: 3.6"
        ],
        keywords="unfolding",
        packages=find_packages(exclude=["contrib", "docs", "tests"]),
        install_requires=[
            "numpy",
            "scikit-learn>=0.18.1",
            "scipy",
            "matplotlib"
        ]
)
