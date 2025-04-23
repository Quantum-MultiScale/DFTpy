# DFTpy Manual, Tutorials and more

See [DFTpy's website](http://dftpy.rutgers.edu) for details.

See also [DFTpy's forum](https://lists.rutgers.edu/mm3/archives/list/dftpy_forum@email.rutgers.edu/) for commonly asked questions.

## For Developers

### Manage Project with [`uv`](https://docs.astral.sh/uv/) (the recommended way)

Managing this project with **`uv`** is straightforward:

1. **Install `uv`**: Follow the [installation guide](https://docs.astral.sh/uv/getting-started/installation/).
2. **Navigate to the DFTpy project root directory**.
3. **Run the following commands**:
   ```shell
   uv venv --python 3.11
   source .venv/bin/activate
   uv sync --active --all-extras --all-groups
   ```
   **Notes on Python Version Compatibility**
   - For Python 3.9–3.11, all optional dependencies can be installed, so `--all-extras` will work.
   - For Python 3.12+, some optional dependencies (listed under [project.optional-dependencies]) are currently
     unavailable. In this case, omit the `--all-extras` flag.

### How to Add and Run Tests

We have a *tests* module where developers can place their test files. It is in the top level of this project. We have
already included some example test cases in it.

The **tests** module consists of two submodules:
- **unit** for unit tests.
- **integration** for integration tests.

There are no specific layout requirements for integration tests. However, for unit tests, we follow a structured
convention: if there is a source file in *src/dftpy/xxx.py*, its corresponding test cases should be in
*tests/unit/test_xxx.py*.

We use **pytest** as our test framework. Its configuration is defined in the [tool.pytest.ini_options] section of
*pyproject.toml*. **pytest** follows specific rules for identifying tests in a project (details can be found in its
documentation). One such rule is that any function whose name starts or ends with `test` will be automatically
recognized as a test by **pytest**.
