# DFTpy Manual, Tutorials, and More

See [DFTpy's website](http://dftpy.rutgers.edu) for details.

See also [DFTpy's forum](https://lists.rutgers.edu/mm3/archives/list/dftpy_forum@email.rutgers.edu/) for commonly-asked
questions.

## For Developers

### Manage this project with [`uv`](https://docs.astral.sh/uv/)
`uv` is an extremely fast Python package and project manager, written in Rust.

**Highlights**
- 🚀 A single tool to replace pip, pip-tools, pipx, poetry, pyenv, twine, virtualenv, and more.
- ⚡️ 10-100x faster than pip.
- 🗂️ Provides comprehensive project management, with a universal lockfile.
- ❇️ Runs scripts, with support for inline dependency metadata.
- 🐍 Installs and manages Python versions.
- 🛠️ Runs and installs tools published as Python packages.
- 🔩 Includes a pip-compatible interface for a performance boost with a familiar CLI.
- 🏢 Supports Cargo-style workspaces for scalable projects.
- 💾 Disk-space efficient, with a global cache for dependency deduplication.
- ⏬ Installable without Rust or Python via curl or pip.
- 🖥️ Supports macOS, Linux, and Windows.

`uv` is backed by *Astral*, the creators of [`Ruff`](https://docs.astral.sh/ruff/), an extremely fast Python linter and
code formatter, written in Rust.

#### Quickstart Guide for Busy Developers

This short video [UV for Python… (Almost) All Batteries Included](https://www.youtube.com/watch?v=qh98qOND6MI) covers
various `uv` use cases to meet all your needs.

#### Other Learning Materials

- [Official `uv` Documentation](https://docs.astral.sh/uv/).

- [Python Developer Tooling Handbook](https://pydevtools.com/handbook/how-to/how-to-fix-python-version-incompatibility-errors-in-uv/)
  This document primarily uses `uv`.

#### Set Up Develop Environment with [`uv`](https://docs.astral.sh/uv/)

Managing this project with **`uv`** is straightforward:

1. **Install `uv`**: Follow the [installation guide](https://docs.astral.sh/uv/getting-started/installation/).
2. **Navigate to the DFTpy project root directory**.
3. **Run the following commands**:
   ```shell
   uv venv --python 3.11
   source .venv/bin/activate
   uv sync --all-extras --all-groups
   ```
   **Notes on Python Version Compatibility**
   - For Python 3.9–3.11, all optional dependencies can be installed, so `--all-extras` will work.
   - For Python 3.12+, some optional dependencies (listed under [project.optional-dependencies]) are currently
     unavailable. In this case, omit the `--all-extras` flag.

#### Package Project with [`uv`](https://docs.astral.sh/uv/)

Before running the commands below, remember to activate your virtual environment with `source .venv/bin/activate`.

- `uv build`: build a source distribution and a binary distribution.
- `uv build --sdist`: build a source distribution.
- `uv build --wheel`: build a binary distribution.

### How to Add and Run Tests

We have a `tests` module where developers can place their test files. It is in the top-level of this project. We have
already included some example test cases in it.

The `tests` module consists of two submodules:
- `unit` - For unit tests.
- `integration` - For integration tests.

There are no specific layout requirements for integration tests. However, for unit tests, we follow a structured
convention: if there is a source file in *src/dftpy/xxx.py*, its corresponding test cases should be in
*tests/unit/test_xxx.py*.

We use **pytest** as our test framework. Its configuration is defined in the [tool.pytest.ini_options] section of
*pyproject.toml*. **pytest** follows specific rules for identifying tests in a project (details can be found in its
documentation). One such rule is that any function whose name starts or ends with `test` will be automatically
recognized as a test by **pytest**.
