
"""
Module to expose more detailed version info for the installed `dftpy`
"""
version = "2.1.3dev0+git20250424.00e36ae"
__version__ = version
full_version = version

git_revision = "00e36ae6ed9e9e3c73c2e6a07f26ce030da90648"
release = 'dev' not in version and '+' not in version
short_version = version.split("+")[0]
