import sys

#--------------------------------------------------------------------------
# General configuration
#--------------------------------------------------------------------------
extensions = [
    'sphinx.ext.mathjax',       # MathJax for mathematical expressions
    'sphinx_copybutton',        # Copy button for code blocks
    'sphinx_rtd_theme',         # ReadTheDocs theme
    'sphinx_toolbox.collapse',  # Collapse sections in the documentation
    'sphinxcontrib.bibtex',     # BibTeX bibliography
    'sphinxcontrib.doxylink',   # Doxygen links
    'sphinxcontrib.mermaid',    # Mermaid diagrams
    'sphinxcontrib.spelling'    # Spelling checker
]

#--------------------------------------------------------------------------
# Project information
#--------------------------------------------------------------------------
templates_path = ['_templates']
source_suffix = ['.rst']
master_doc = 'index'  # The master toctree document
project = 'OpenTurbine'
copyright = '2023 - Present, MIT License'
author = 'National Renewable Energy Laboratory (NREL) and Sandia National Laboratories (SNL)'

# Version info
version = '0.0'  # The short X.Y version
release = '0.0.1'  # The full version, including alpha/beta/rc tags

#--------------------------------------------------------------------------
# Spelling configuration
#--------------------------------------------------------------------------
spelling_word_list_filename = "spelling_wordlist.txt"
spelling_exclude_patterns = ["doxygen/html/*"]
spelling_show_suggestions = True
spelling_warning = True
spelling_ignore_contributor_names = False

#--------------------------------------------------------------------------
# Build configuration
#--------------------------------------------------------------------------
# Patterns to ignore when looking for source files
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Pygments i.e. syntax highlighting style to use
pygments_style = 'sphinx'

# Include todos in the output?
todo_include_todos = False

# Figure, table, and code-block numbering
numfig = True
numfig_format = {'figure': '%s', 'table': '%s', 'code-block': '%s'}

#--------------------------------------------------------------------------
# HTML output configuration
#--------------------------------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_logo = '_static/oturb_logo_v1.jpeg'
html_static_path = ['_static']  # Path to static files
html_show_copyright = False  # Show copyright in the footer
htmlhelp_basename = 'openturbine_doc'  # Output file base name for HTML help builder

#--------------------------------------------------------------------------
# LaTeX output configuration
#--------------------------------------------------------------------------
# Group the document tree into LaTeX files
latex_documents = [(
    master_doc,                  # source start file
    'openturbine.tex',           # target name
    'OpenTurbine Documentation', # title
    author,                      # author
    'manual'                     # documentclass [howto, manual, or own class]
)]

#--------------------------------------------------------------------------
# Manual page output configuration
#--------------------------------------------------------------------------
# One entry per manual page
man_pages = [(
    master_doc,                  # source start file
    'openturbine',               # name
    u'OpenTurbine Documentation', # description
    [author],                    # authors
    1                            # manual section
)]

#--------------------------------------------------------------------------
# Texinfo output configuration
#--------------------------------------------------------------------------
# Grouping the document tree into Texinfo files
texinfo_documents = [(
    master_doc,                   # source start file
    'openturbine',                # name
    u'OpenTurbine Documentation', # description
    author,                       # author
    'OpenTurbine',                # project
    'Flexible Multibody Dynamics of Wind Turbines', # description
    'Miscellaneous'),             # category
]

def setup(app):
    app.add_object_type("input_param", "input_param",
                       objname="OpenTurbine input parameter",
                       indextemplate="pair: %s; OpenTurbine input parameter")
