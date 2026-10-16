# Reusing the notebook-backed package

The user explicitly retained the workspace rule: all executable authored code lives in `.ipynb` files. This replaces the brief's conventional `.py` packaging and script-based launcher suggestion. There is one implementation per module, not copies inside experiment notebooks. `pyproject.toml` records metadata/dependencies; this is not a pip-installable wheel.

Every consumer executes only the `ahr-export` cells in `fmm_jepa/bootstrap.ipynb`, calls `install_notebook_package(project_root)`, then uses ordinary imports such as `from fmm_jepa.models import JEPA`. The import hook executes only tagged definition cells in the selected module notebook. It never runs upstream training, benchmarks, test execution cells or `%run`. Bootstrap cells in module/test notebooks permit independent fresh-kernel execution and are skipped during imports.

Set `FMM_JEPA_ROOT` to an absolute project path when relocating. No working-directory dependency is required. Source hashes cover exported definitions; importing executes trusted local project source, so do not load arbitrary notebooks. Restart the kernel after changing module code. Notebook 01 must compare handoff source/schema hashes, load a fresh model and rerun the compact tests.

The same bootstrap code is shown as the first executable cell in every consuming notebook. It only selects tagged cells and delegates all loading behavior to the one shared bootstrap implementation.
