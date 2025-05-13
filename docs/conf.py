import os
import sys
from pathlib import Path

# -- Project information -----------------------------------------------------
project = "PyMC-BART"
copyright = "2022, PyMC Community"
author = "PyMC Community"

# -- General configuration ---------------------------------------------------

sys.path.insert(0, os.path.abspath("../sphinxext"))

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_nb",
    "sphinx_design",
    "sphinxcontrib.bibtex",
    "sphinx_codeautolink",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "*import_posts*",
    "**/.ipynb_checkpoints/*",
    "**/*.myst.md",
]

if os.path.exists("examples"):
    external_docs = os.listdir("examples")
    for doc in external_docs:
        file = Path("examples", doc)
        if os.path.exists(file):
            os.remove(file)

os.system(
    "wget https://raw.githubusercontent.com/pymc-devs/pymc-examples/main/examples/bart/bart_introduction.ipynb -P examples"
)
os.system(
    "wget https://raw.githubusercontent.com/pymc-devs/pymc-examples/main/examples/bart/bart_quantile_regression.ipynb -P examples"
)
os.system(
    "wget https://raw.githubusercontent.com/pymc-devs/pymc-examples/main/examples/bart/bart_heteroscedasticity.ipynb -P examples"
)
os.system(
    "wget https://raw.githubusercontent.com/pymc-devs/pymc-examples/main/examples/bart/bart_categorical_hawks.ipynb -P examples"
)
os.system(
    "wget https://raw.githubusercontent.com/pymc-devs/pymc-examples/main/examples/references.bib -P examples"
)

# bibtex config
bibtex_bibfiles = ["examples/references.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# theme options
html_theme = "pymc_sphinx_theme"
html_theme_options = {
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink", "donate"],
    "search_bar_text": "Search within PyMC-BART...",
    "navbar_start": ["navbar-logo"],
    "icon_links": [
        {
            "url": "https://github.com/pymc-devs/pymc-bart",
            "icon": "fa-brands fa-github",
            "name": "GitHub",
        },
    ],
}

version = os.environ.get("READTHEDOCS_VERSION", "")
version = version if "." in version else "main"
html_context = {
    "github_url": "https://github.com",
    "github_user": "pymc-devs",
    "github_repo": "pymc-bart",
    "github_version": version,
    "default_mode": "light",
}


html_favicon = "../_static/PyMC.ico"
html_logo = "logos/pymc_bart.png"
html_title = "PyMC-BART"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["../_static", "logos"]
html_extra_path = ["../_thumbnails"]
templates_path = ["../_templates"]
html_sidebars = {
    "**": [
        "sidebar-nav-bs.html",
    ],
}

# MyST config
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath", "substitution"]
citation_code = f"""
```bibtex
@incollection{{citekey,
  author    = "<notebook authors, see above>",
  title     = "<notebook title>",
  editor    = "PyMC Team",
  booktitle = "PyMC examples",
}}
```
"""


myst_substitutions = {
    "pip_dependencies": "{{ extra_dependencies }}",
    "conda_dependencies": "{{ extra_dependencies }}",
    "extra_install_notes": "",
    "citation_code": citation_code,
}
nb_execution_mode = "off"


# bibtex config
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"

# OpenGraph config
# use default readthedocs integration aka no config here

codeautolink_autodoc_inject = False
codeautolink_concat_default = True

# intersphinx mappings
intersphinx_mapping = {
    "arviz": ("https://python.arviz.org/en/latest/", None),
    "bambi": ("https://bambinos.github.io/bambi", None),
    "einstats": ("https://einstats.python.arviz.org/en/latest/", None),
    "mpl": ("https://matplotlib.org/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "pymc": ("https://www.pymc.io/projects/docs/en/stable/", None),
    "pytensor": ("https://pytensor.readthedocs.io/en/latest/", None),
    "pmx": ("https://www.pymc.io/projects/experimental/en/latest/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}
