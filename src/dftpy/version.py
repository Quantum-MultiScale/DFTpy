
"""
Module to expose more detailed version info for the installed `dftpy`
"""
version = "2.1.3dev0+git20250423.caf003a"
__version__ = version
full_version = version

git_revision = "caf003a5561e4f323c5fe09b78f8eb9c9eb111d0"
release = 'dev' not in version and '+' not in version
short_version = version.split("+")[0]
