# MultiRefLfD-TPGP

![GitHub](https://img.shields.io/github/license/AshrithSagar/MultiRefLfD-TPGP)
![GitHub repo size](https://img.shields.io/github/repo-size/AshrithSagar/MultiRefLfD-TPGP)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Learning Multi-Reference Frame Skills from Demonstration with Task-Parameterized Gaussian Processes (TPGP)

[DOI](https://doi.org/10.1109/IROS58592.2024.10803060)
|
[PDF](http://www.jenskober.de/publications/RamirezMontero2024IROS.pdf)

## Installation

1. First, clone the repo.

```shell
git clone https://github.com/AshrithSagar/MultiRefLfD-TPGP.git
```

2. Install [`uv`](https://docs.astral.sh/uv/), if not already.
   Check [here](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.

> [!TIP]
> It is recommended to use `uv`, as it will automatically install the dependencies in a virtual environment.

TL;DR:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

The dependencies are listed in the [pyproject.toml](pyproject.toml) file.

3. Install the package in editable mode (recommended).

```shell
uv pip install -e .
```

## Usage

> [!WARNING]
> WIP

Use the `lfd` module as a library.

```python
import lfd

fdset = lfd.utils.load_fdset("s")

P = lfd.alignment.computeP(fdset)
D0_star = lfd.alignment.align_demonstrations(fdset, P)

lfd.alignment.plot_keypoints(fdset, P)
lfd.alignment.plot_alignments(fdset, D0_star, P)

X = lfd.utils.transform_data(D0_star)
```

Run scripts directly using `uv run`.

```shell
uv run lfd/run.py
```

## References

- Mariano Ramírez Montero, Giovanni Franzese, Jens Kober, and Cosimo Della Santina. Learning Multi-Reference Frame Skills from Demonstration with Task-Parameterized Gaussian Processes. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 2832–2839, 2024.

- <https://github.com/dfki-ric/movement_primitives>
