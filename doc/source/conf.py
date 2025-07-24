import sys
import sphinx_rtd_theme
project = 'DFTpy'
copyright = '2019-2023, Pavanello Research Group'
author = 'Pavanello Research Group'
release = '2.2.0dev'

source_suffix = '.rst'
master_doc = 'index'

nbsphinx_execute = 'never'

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.mathjax',
              'sphinx.ext.viewcode',
              'sphinx.ext.napoleon',
              'sphinx.ext.intersphinx',
              'nbsphinx',
              'sphinx_design']

templates_path = ['templates']
exclude_patterns = ['build']

html_theme = 'sphinx_rtd_theme'
# html_theme = 'sphinx_bootstrap_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_favicon = 'static/dftpy.ico'
html_logo = 'static/dftpy.png'
html_style = 'custom.css'
html_static_path = ['static']
html_last_updated_fmt = '%A, %d %b %Y %H:%M:%S'

html_theme_options = {
    'prev_next_buttons_location': 'both',
}

latex_show_urls = 'inline'
latex_show_pagerefs = True
latex_engine = 'xelatex'
# latex_documents = [('index', 'DFTpy.tex', 'DFTpy', 'DFTpy-Developers', 'manual')]
latex_documents = [('index', 'DFTpy.tex', 'DFTpy', 'DFTpy-Developers', 'howto', not True)]


#Add external links to source code
def linkcode_resolve(domain, info):
    print('info module', info)
    if domain != 'py' or not info['module']:
        return None

    filename = info['module'].replace('.', '/')+'.py'
    return "https ://gitlab.com/pavanello-research-group/dftpy/tree/master/%s" % filename
