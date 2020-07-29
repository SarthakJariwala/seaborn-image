from datetime import datetime
import sphinx_bootstrap_theme


project = "seaborn-image"
author = "Sarthak Jariwala"
copyright = f"{datetime.now().year}, {author}"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinxcontrib.images"]
html_static_path = ["_static"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build"]

# The reST default role (used for this markup: `text`) to use for all documents.
# default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
# keep_warnings = False


# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "bootstrap"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "source_link_position": "footer",
    "bootswatch_theme": "united",  # paper, simplex
    "navbar_sidebarrel": False,
    "bootstrap_version": "3",
    "navbar_links": [
        ("Quickstart", "quickstart"),
        ("How-to?", "how_to"),
        ("Gallery", "gallery"),
        ("Releases", "https://github.com/SarthakJariwala/seaborn-image/releases", True),
        ("Reference", "reference"),
    ],
}

html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
# try:
#     from seaborn_image import __version__ as version
# except ImportError:
#     pass
# else:
#     release = version

# # A shorter title for the navigation bar.  Default is the same as html_title.
# # html_short_title = None

# # The name of an image file (relative to this directory) to place at the top
# # of the sidebar.
# # html_logo = ""

# # The name of an image file (within the static path) to use as favicon of the
# # docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# # pixels large.
# # html_favicon = None

# # Add any paths that contain custom static files (such as style sheets) here,
# # relative to this directory. They are copied after the builtin static files,
# # so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# # If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# # using the given strftime format.
# # html_last_updated_fmt = '%b %d, %Y'

# # If true, SmartyPants will be used to convert quotes and dashes to
# # typographically correct entities.
# # html_use_smartypants = True

# # Custom sidebar templates, maps document names to template names.
# # html_sidebars = {}

# # Additional templates that should be rendered to pages, maps page names to
# # template names.
# # html_additional_pages = {}

# # If false, no module index is generated.
# # html_domain_indices = True

# # If false, no index is generated.
# # html_use_index = True

# # If true, the index is split into individual pages for each letter.
# # html_split_index = False

# # If true, links to the reST sources are added to the pages.
# # html_show_sourcelink = True

# # If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
# # html_show_sphinx = True

# # If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
# # html_show_copyright = True

# # If true, an OpenSearch description file will be output, and all pages will
# # contain a <link> tag referring to it.  The value of this option must be the
# # base URL from which the finished HTML is served.
# # html_use_opensearch = ''

# # This is the file name suffix for HTML files (e.g. ".xhtml").
# # html_file_suffix = None

# # Output file base name for HTML help builder.
# htmlhelp_basename = 'seaborn_image-doc'


# # -- Options for LaTeX output --------------------------------------------------

# latex_elements = {
# # The paper size ('letterpaper' or 'a4paper').
# # 'papersize': 'letterpaper',

# # The font size ('10pt', '11pt' or '12pt').
# # 'pointsize': '10pt',

# # Additional stuff for the LaTeX preamble.
# # 'preamble': '',
# }

# # Grouping the document tree into LaTeX files. List of tuples
# # (source start file, target name, title, author, documentclass [howto/manual]).
# latex_documents = [
#   ('index', 'user_guide.tex', u'seaborn-image Documentation',
#    u'Sarthak Jariwala', 'manual'),
# ]

# # The name of an image file (relative to this directory) to place at the top of
# # the title page.
# # latex_logo = ""

# # For "manual" documents, if this is true, then toplevel headings are parts,
# # not chapters.
# # latex_use_parts = False

# # If true, show page references after internal links.
# # latex_show_pagerefs = False

# # If true, show URL addresses after external links.
# # latex_show_urls = False

# # Documents to append as an appendix to all manuals.
# # latex_appendices = []

# # If false, no module index is generated.
# # latex_domain_indices = True

# -- External mapping ------------------------------------------------------------
intersphinx_mapping = {
    "sphinx": ("http://www.sphinx-doc.org/en/stable", None),
    "matplotlib": ("https://matplotlib.org", None),
    "numpy": ("https://docs.scipy.org/doc/numpy", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "scikit-image": ("https://scikit-image.org/", None),
}
