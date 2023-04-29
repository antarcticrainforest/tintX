# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))
os.environ["FILE_PATH"] = os.path.abspath(os.path.join(".", "_static", "data"))
from tintx import __version__
from recommonmark.parser import CommonMarkParser

project = "tintX"
copyright = f"{date.today().year}, Martin Bergemann"
author = "Martin Bergemann"
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "recommonmark",
    "sphinx_execute_code",
    "sphinxcontrib_github_alt",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

source_parsers = {
    ".md": CommonMarkParser,
}

source_suffix = [".rst", ".md"]
