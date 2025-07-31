# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    =
PAPER         ?= letter
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
PYTHON = python3

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

.PHONY: help Makefile html browse pdf upload config

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

config:
	@echo "Base on configentries.json generate config.rst"
	$(PYTHON) gen_doc.py

html: config
	$(SPHINXBUILD) -b html $(SPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)/html"
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

pdf: config
	@echo "Building PDF with $(PAPER) paper size..."
	$(SPHINXBUILD) -b latex $(SPHINXOPTS) -D latex_paper_size=$(PAPER) "$(SOURCEDIR)" "$(BUILDDIR)/latex"
	@echo "Running 'make' in $(BUILDDIR)/latex to generate PDF."
	$(MAKE) -C "$(BUILDDIR)/latex" all-pdf
	@echo
	@echo "Build finished. The PDF is in $(BUILDDIR)/latex/."

browse:
	x-www-browser build/html/index.html

upload: html
	scp -r ./build/html/* web@dftpy.rutgers.edu:/home/web/dftpy/html/
