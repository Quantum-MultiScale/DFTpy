
"""
Module to expose more detailed version info for the installed `dftpy`
"""
version = "2.1.3dev0+git20250424.a903bde"
__version__ = version
full_version = version

git_revision = "a903bde4b3ed79f69ba01a58e87b847d00da9fcf"
release = 'dev' not in version and '+' not in version
short_version = version.split("+")[0]
