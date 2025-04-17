# MultiRefLfD-TPGP

![GitHub](https://img.shields.io/github/license/AshrithSagar/MultiRefLfD-TPGP)
![GitHub repo size](https://img.shields.io/github/repo-size/AshrithSagar/MultiRefLfD-TPGP)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Learning Multi-Reference Frame Skills from Demonstration with Task-Parameterized Gaussian Processes (TPGP)

[DOI](https://doi.org/10.1109/IROS58592.2024.10803060)
|
[PDF](http://www.jenskober.de/publications/RamirezMontero2024IROS.pdf)

## Installation

1. First, clone the repo.

    ```shell
    git clone https://github.com/AshrithSagar/MultiRefLfD-TPGP.git
    ```

2. Install the [dependencies](requirements.txt), creating a virtual environment if required.

    ```shell
    pip3 install -r requirements.txt
    cd lfd
    ```

    Or, just install through `pip` in editable mode.

    ```shell
    pip3 install -e .
    ```

## Usage

WIP

```python
from lfd import *

data, x, xd = load_data("s")
dset = append_progress_values([LineString(traj) for traj in data])

plot_index_points(dset, indices=[0, 1], only_between=True)
plot_keypoints(dset, indices=[1])

f1 = Frame(index=1, rotation=10, translation=(5, 5))
dset_f1 = f1.transform(dset)
plot_trajectories([dset[0], dset_f1[0]])

aligned_dset = resample(dset, frames=[f1])
```

## References

- Mariano Ramírez Montero, Giovanni Franzese, Jens Kober, and Cosimo Della Santina. Learning Multi-Reference Frame Skills from Demonstration with Task-Parameterized Gaussian Processes. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 2832–2839, 2024.

- <https://github.com/dfki-ric/movement_primitives>
