from setuptools import setup, find_packages

# read the contents of your README file
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="py3dinterpolations",
    version="0.4.0",
    description="quick 3D interpolation with python",
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/giocaizzi/py3dinterpolations",
    author="giocaizzi",
    author_email="giocaizzi@gmail.com",
    license="MIT",
    packages=find_packages(include=["py3dinterpolations", "py3dinterpolations.*"]),
    setup_requires=[],
    tests_require=[],
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "plotly",
        "pykrige",
        "scikit-learn",
    ],
    extras_require={
        "docs": [],
        "dev": [],
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    project_urls={
        "Documentation": "https://giocaizzi.github.io/py3dinterpolations/",
        "Bug Reports": "https://github.com/giocaizzi/py3dinterpolations/issues",
        "Source": "https://github.com/giocaizzi/py3dinterpolations",
    },
)
