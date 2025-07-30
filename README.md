# DFTpy Manual, Tutorials and more

See [DFTpy's website](http://dftpy.rutgers.edu) for details.

See also [DFTpy's forum](https://lists.rutgers.edu/mm3/archives/list/dftpy_forum@email.rutgers.edu/) for commonly asked questions.

## [uv](https://docs.astral.sh/uv/)

It is recommended to manage this project with **uv**. Use **uv** to set up this project is simple:
1. Install uv: <https://docs.astral.sh/uv/getting-started/installation/>
2. Change directory to DFTpy project root.
3. Run the commands:
   ```shell
   uv venv --python 3.11
   source .venv/bin/activate
   uv sync --active --all-extras --all-groups
   ```
   In the first command above, if you choose Python 3.9 ~ 3.11, you can install all the optional dependencies, and
   `uv sync --active --all-extras --all-groups` should work. If you choose to use Python 3.12+, currently you can't
   install some optional dependencies, which is listed under `[project.optional-dependencies]`, and you need to drop the
   `--all-extras` flag.

### **Recommended Project Setup with `uv`**

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
