# STalign notebook setup

This workspace contains notebooks that use Squidpy's STalign support from the `feat/stalign-points` branch.

## Create an environment

Use a fresh virtual environment before installing Squidpy and JAX:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## Install Squidpy with JAX support

For a standard install with CPU JAX support, install Squidpy directly from the branch:

```bash
python -m pip install "squidpy[jax] @ git+https://github.com/scverse/squidpy.git@feat/stalign-points"
```

If you prefer an editable install:

```bash
git clone --branch feat/stalign-points https://github.com/scverse/squidpy.git
cd squidpy
python -m pip install -e ".[jax]"
```

## Optional GPU JAX install

If you have a supported NVIDIA GPU, install the CUDA-enabled JAX build first and then install Squidpy from the same branch without pulling in a second JAX build:

```bash
python -m pip install -U "jax[cuda12]"
python -m pip install --no-deps "squidpy @ git+https://github.com/scverse/squidpy.git@feat/stalign-points"
```

For an editable GPU install:

```bash
git clone --branch feat/stalign-points https://github.com/scverse/squidpy.git
cd squidpy
python -m pip install -U "jax[cuda12]"
python -m pip install -e . --no-deps
```

## Notes

- JAX is required for the STalign notebooks in this workspace.
- GPU JAX support is typically available on Linux with a compatible NVIDIA GPU and CUDA 12.
- If GPU installation is not available on your machine, use the CPU install above.
- If the CUDA JAX install fails, check the official JAX installation guide: https://docs.jax.dev/en/latest/installation.html
