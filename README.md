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

Directly run a script:

```shell
uv run lfd/run.py
```

Additionally, use the `lfd` module as a library.

```python
from lfd import *

data, x, xd = load_data("s")
dset = append_progress_values([LineString(traj) for traj in data])

plot_index_points(dset, indices=[0, 1], only_between=True)
plot_keypoints(dset, indices=[1])

f1 = Frame(index=1, rotation=10, translation=(5, 5))
dset_f1 = f1.transform(dset)
plot_trajectories([dset[0], dset_f1[0]])

al_dset = resample(dset, frames=[f1])
```

## References

- Mariano Ramírez Montero, Giovanni Franzese, Jens Kober, and Cosimo Della Santina. Learning Multi-Reference Frame Skills from Demonstration with Task-Parameterized Gaussian Processes. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 2832–2839, 2024.

- <https://github.com/dfki-ric/movement_primitives>
