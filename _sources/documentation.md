# The Docs

The documentation is constructed using
[Jupyter-Book](https://jupyterbook.org/en/stable/intro.html),
a framework built on tyop of Sphinx that supports communication
about software as a first priority. Markdown and restructured
test files are both supported as documentation content.
Additional markdown features through MyST are supported and
Jupyter-Book integrates with all Sphinx extensions plus additional
extensions specific to this framework. The config file is at
`docs/_config.yml`. The `sphinx:` section allows for using any
configuration parameters used directly in Sphinx.

The online documentation is hosted through GitHub Pages and
built and deployed automatically when any file in `docs/` is
modified. See the workflow at `.github/workflows/deploy-pages.yaml`
for reference.


## Building locally

The documentation can be compiled locally using
Jupyter-Book and a few other packages. All dependencies are listed
at `docs/requirements.txt` and can be installed with `pip`.
The commands below describe installing and building the docs.

```bash
# Move into docs/ directory
cd docs/

# Install all depdencies including jupyter-book
pip install -r requirements.txt

# Build the docs
jupyter-book build .
```

Upon successfully compiling, a html file will be available
at `_build/html/index.html`. This can be opened and navigated
with any web browser.


