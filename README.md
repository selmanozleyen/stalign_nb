# STalign notebook setup

This workspace contains notebooks that use Squidpy's STalign support from the `feat/stalign-points` branch.

## Quick instal

1. Install `uv` (macOS/Linux):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create an environment and activate it:

```bash
uv venv .venv
source .venv/bin/activate
```

3. Install Squidpy with CPU JAX from the fork branch:

```bash
uv pip install "squidpy[jax] @ git+https://github.com/selmanozleyen/squidpy.git@feat/stalign-points"
```

## Optional GPU install (uv, non-editable)

For supported NVIDIA/CUDA setups:

```bash
uv pip install -U "jax[cuda12]"
uv pip install --no-deps "squidpy @ git+https://github.com/selmanozleyen/squidpy.git@feat/stalign-points"
```

## Notes

- JAX is required for the STalign notebooks in this workspace.
- GPU JAX support is typically available on Linux with a compatible NVIDIA GPU and CUDA 12.
- If the CUDA JAX install fails, see the JAX installation guide: https://docs.jax.dev/en/latest/installation.html
