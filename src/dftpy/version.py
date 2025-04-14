
"""
Module to expose more detailed version info for the installed `dftpy`
"""
version = "2.1.3dev0+git20250414.93b1ba7"
__version__ = version
full_version = version

git_revision = "93b1ba73b312b317e74555f53974731864fb7357"
release = 'dev' not in version and '+' not in version
short_version = version.split("+")[0]
