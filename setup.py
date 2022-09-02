import os
from setuptools import setup, find_packages
import sys
from tempfile import TemporaryDirectory


def read(*parts):
    script_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    return open(os.path.join(script_path, *parts)).read()


def find_version(*parts):

    vers_file = read(*parts).split("\n")
    old_path = sys.path.copy()
    with TemporaryDirectory() as td:
        with open(os.path.join(td, "tmp_frevaversion.py"), "w") as f:
            for line in vers_file:
                if "__version__" in line:
                    f.write(line)
        sys.path.insert(0, td)
        try:
            from tmp_frevaversion import __version__

            sys.path = old_path
            return __version__
        except ImportError:
            sys.path = old_path
    raise RuntimeError("Unable to find version string.")


meta = dict(
    description="Tracking facility to track rainfall and other non-continous data.",
    url="https://github.com/antarcticrainforest/tintX",
    author="Martin Bergemann",
    author_email="bergemann@dkrz.de",
    long_description=read("README.md"),
    include_package_data=True,
    long_description_content_type="text/markdown",
    license="BSD-3-Clause",
    python_requires=">=3.7",
    project_urls={
        "Documentation": "https://tintx.readthedocs.io/en/latest/",
        # 'Release notes': 'https://extra-data.readthedocs.io/en/latest/changelog.html',
        "Issues": "https://github.com/antarcticrainforest/tintX/issues",
        "Source": "https://github.com/antarcticrainforest/tintX",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    version=find_version("src", "tintx", "__init__.py"),
    package_dir={"": "src"},
    entry_points={"console_scripts": ["tintx = tintx.cli:tintx"]},
    install_requires=[
        "cartopy",
        "click",
        "cftime",
        "pandas",
        "tables",
        "numpy",
        "scipy",
        "netCDF4",
        "matplotlib",
        "xarray",
        "tqdm",
        "typing_extensions",
    ],
    extras_require={
        "tests": [
            "bash_kernel",
            "black",
            "dask[diagnostics]",
            "nbval",
            "pytest",
            "pytest-env",
            "pytest-cov",
            "testpath",
            "flake8",
            "mypy",
        ],
        "docs": [
            "bash_kernel",
            "black",
            "dask",
            "h5netcdf",
            "mypy",
            "pytest",
            "recommonmark",
            "nbsphinx",
            "sphinx",
            "sphinxcontrib_github_alt",
            "sphinx-execute-code-python3",
            "sphinx-rtd-theme",
        ],
    },
)

setup(name="tintx", packages=find_packages("src"), **meta)
