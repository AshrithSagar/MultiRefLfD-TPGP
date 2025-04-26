"""
lfd/utils/lasa/helpers.py \n
Helper functions
"""

from typing import Union

import matplotlib.pyplot as plt
import numpy as np

from .lasa import load_lasa


def plot_curves(x, show_start_end=True, **kwargs):
    """
    plots 2d curves of trajectories

    params:
        x: array of shape (number of curves,n_steps_per_curve,2)
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

    plt.legend()


def streamplot(f, x_axis=(0, 100), y_axis=(0, 100), n=1000, width=1, **kwargs):
    """
    helps visualizing the vector field.

    params:
        f: function to predict the velocities in DS(Dynamical system : x_dot = f(x),x of shape (n_points,2),x_dot of shape (n_points,2))
        x_axis: x axis limits
        y_axis: y axis limits
        n: number of points in each axis (so total n*n predictions happen)
        width: width of the vector
        **kwargs: goes into plt.streamplot
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


# gets the velocity x_dot given x
def derivative(x):
    """
    difference method for calculating derivative

    params:
        x: array of shape (number of trajectories,number of timesteps,2)

    returns
        xd: array of shape (number of trajectories,number of timesteps,2)
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


# loading the data and plotting
def load_data(key: Union[str, int], show_plot: bool = False):
    """
    gets the trajectories coresponding to the given letter

    params:
        letter: character in ["c","j","s"]

    returns:
        data: array of shape (number of trajectories,number of timesteps,2)
        x: array of shape(number of trajectories*number of timesteps,2)
        xd: array of shape(number of trajectories*number of timesteps,2)

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


def plot_curves3(x, alpha=1):
    """
    plots 2d curves

    params:
        x: array of shape (number of curves,n_steps_per_curve,2)
    """
    for t in range(x.shape[0]):
        plt.scatter(x[t][0, 0], x[t][0, 1], c="k")
        plt.scatter(x[t][-1, 0], x[t][-1, 1], c="b")
        plt.plot(x[t][:, 0], x[t][:, 1], alpha=alpha)


# plotting the transformation axes
def plot_frame(A, b, scale=1):
    """
    for plotting x and y axis of frames
    """
    plt.arrow(*b[1:], *(A[1:, 1] * scale), color="b")
    plt.arrow(*b[1:], *(A[1:, 2] * scale), color="r")


# transformation matrix
def getA(a1):
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


# load lasa data, note that here Data is 3 dimension (in additional to space , time dimensional is added)
# assumption is that time goes from 0 to 1 sec
def load_data3(letter: str):
    letter2id = dict(c=2, j=6, s=24)
    _, x, _, _, _, _ = load_lasa(letter2id[letter.lower()])
    time = np.linspace(0, 1, x.shape[1])
    time = np.tile(time[None, ..., None], (x.shape[0], 1, 1))
    Data = np.concatenate([time, x], axis=-1)
    print(f"{Data.shape=}")
    return Data, time


# plotting trajectories
# choosing the frames(x axis of the frames in the direction of starting and ending of trajectory)
# As are orientation, Bs are origin of the frames
def plot_trajectories(Data):
    plot_curves3(Data[:, :, 1:], alpha=0.5)
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
