"""
lfd/utils/lasa/helpers.py \n
Helper functions
"""

from typing import Union

import matplotlib.pyplot as plt
import numpy as np

from .lasa import load_lasa


def plot_curves(x, show_start_end=False, **kwargs):
    """
    Plot 2D curves of trajectories.

    :param x: Data (n_traj, n_length, 2)
    :param show_start_end: Show start and end points
    :param kwargs: Additional parameters for plt.plot
    """
    if show_start_end:
        start_label, end_label = "start", "end"
    else:
        start_label, end_label = None, None
    for t in range(x.shape[0]):
        plt.scatter(x[t][0, 0], x[t][0, 1], c="k", label=start_label)
        plt.scatter(x[t][-1, 0], x[t][-1, 1], c="b", label=end_label)
        plt.plot(x[t][:, 0], x[t][:, 1], **kwargs)
        if t == 0:
            kwargs.pop("label", None)
            start_label, end_label = None, None
    if show_start_end:
        plt.legend()


def streamplot(f, n=1000, x_axis=(0, 100), y_axis=(0, 100), width=1, **kwargs):
    """
    Visualise the vector field.
    The dynamical system (DS) function predicts the velocities as a function of the state.
    x_dot = f(x); x_dot: (n_length, 2); x: (n_length, 2)

    :param f: Dynamical system function
    :param n: Number of points in each axis (Total n*n predictions happen)
    :param x_axis: X-axis limits
    :param y_axis: Y-axis limits
    :param width: Width of the vector
    :param **kwargs: Any additional params for plt.streamplot
    """
    a, b = np.linspace(x_axis[0], x_axis[1], n), np.linspace(y_axis[0], y_axis[1], n)
    X, Y = np.meshgrid(a, b)
    X_test = np.stack([X, Y], axis=-1).reshape(-1, 2)
    Y_pred = f(X_test)
    U, V = np.split(Y_pred.reshape(n, n, 2), 2, axis=-1)
    U, V = U[..., 0], V[..., 0]
    speed = np.sqrt(U**2 + V**2)
    lw = width * speed / speed.max()
    plt.streamplot(X, Y, U, V, linewidth=lw, **kwargs)


def derivative(x):
    """
    Difference method for calculating derivatives.
    Gets the velocity x_dot given x.

    :param x: Original (n_traj, n_length, 2)
    :return xd: Derivative (n_traj, n_length, 2)
    """
    xds = []
    for i in range(x.shape[0]):
        dt = 1 / (x[i].shape[0] - 1)
        xd = np.vstack((np.diff(x[i], axis=0) / dt, np.zeros((1, x[i].shape[1]))))
        v_factor = np.cos(np.linspace(0, np.pi / 2, len(xd))) ** 2
        xd = xd * (v_factor[..., None])
        xds.append(xd)
    xd = np.stack(xds)
    return xd


def load_data(key: Union[str, int], show_plot: bool = False):
    """
    Gets the trajectories corresponding to the given letter

    :param key: Letter or index for a trajectory
    :return data: (n_traj, n_length, 2)
    :return x: (n_traj * n_length, 2)
    :return xd: (n_traj * n_length, 2)
    """
    if isinstance(key, str):
        letter2id = dict(c=2, g=4, j=6, s=24)
        assert key.lower() in letter2id
        _index = letter2id[key.lower()]
    elif isinstance(key, int):
        _index = key

    _, x, _, _, _, _ = load_lasa(_index)
    xd = derivative(x)

    if show_plot:
        plot_curves(x)
        plt.show()

    data = x
    x = x.reshape(-1, 2)
    xd = xd.reshape(-1, 2)
    return data, x, xd


def plot_frame(A, b, scale=1):
    """Plot the transformation axes of frames"""
    plt.arrow(*b[1:], *(A[1:, 1] * scale), color="b")
    plt.arrow(*b[1:], *(A[1:, 2] * scale), color="r")


def getA(a1):
    """Transformation matrix"""
    a1 = a1 / np.linalg.norm(a1)
    rot_mat = np.array(
        [
            [np.cos(np.pi / 2), -np.sin(np.pi / 2)],
            [np.sin(np.pi / 2), np.cos(np.pi / 2)],
        ]
    )
    a2 = rot_mat @ a1
    A = np.eye(3)
    A[1:][:, 1] = a1
    A[1:][:, 2] = a2
    return A


def load_data_with_phi(letter: str):
    """
    Load demonstrations with added progress values (phi) in dimension 0.
    """
    letter2id = dict(c=2, j=6, s=24)
    _, x, _, _, _, _ = load_lasa(letter2id[letter.lower()])
    time = np.linspace(0, 1, x.shape[1])
    time = np.tile(time[None, ..., None], (x.shape[0], 1, 1))
    data = np.concatenate([time, x], axis=-1)
    return data, time


# plotting trajectories
# choosing the frames(x axis of the frames in the direction of starting and ending of trajectory)
# As are orientation, Bs are origin of the frames
def plot_trajectories(Data):
    plot_curves(Data[:, :, 1:], alpha=0.5)
    scale = 10
    As = []
    Bs = []
    for e, d in enumerate(Data):
        a1 = (d[100] - d[0])[1:]
        A1 = getA(a1)
        b1 = d[0]
        b1[0] = 0
        plot_frame(A1, b1, scale)

        a2 = (d[-100] - d[-1])[1:]
        A2 = getA(a2)
        b2 = d[-1]
        b2[0] = 0
        plot_frame(A2, b2, scale)

        As.append([A1, A2])
        Bs.append([b1, b2])
    As = np.array(As)
    As = np.transpose(As, (1, 0, 2, 3))
    Bs = np.array(Bs)
    Bs = np.transpose(Bs, (1, 0, 2))
    print(f"{As.shape=}")
    print(f"{Bs.shape=}")
    return As, Bs
