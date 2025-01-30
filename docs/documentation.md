# Documentation

The documentation is constructed using
[Jupyter-Book](https://jupyterbook.org/en/stable/intro.html),
a framework built on top of
[Sphinx](https://www.sphinx-doc.org/en/master/index.html)
that prioritizes software communication. Both Markdown (`.md`) and
reStructuredText (`.rst`) files are supported as documentation content.
Additional Markdown features through MyST are supported, and Jupyter-Book
integrates with all Sphinx extensions plus additional extensions
specific to the Jupyter-Book framework. The configuration file is located at
`docs/_config.yml`, where you can use any configuration parameters
directly from Sphinx in the `sphinx:` section.

The online documentation is hosted on GitHub Pages and is built and
deployed automatically whenever any file in the `docs/` directory is
modified. Refer to the workflow at `.github/workflows/deploy-pages.yaml`
for more details.

## Building Locally

The documentation can be compiled locally using Jupyter-Book and a
few other packages. All dependencies are listed in
`docs/requirements.txt` and can be installed with `pip`. The commands
below describe the process of installing and building the documentation.

```bash
# Navigate to the docs/ directory
cd docs/

# Install all dependencies including Jupyter-Book
pip install -r requirements.txt

# Build the documentation
jupyter-book build .
```

Upon successful compilation, an HTML file will be available at
`_build/html/index.html`. This can be opened and navigated with any
web browser.
