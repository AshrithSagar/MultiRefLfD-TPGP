"""
utils/lasa/helpers.py \n
Helper functions
"""

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
def load_data(letter):
    """
    gets the trajectories coresponding to the given letter

    params:
        letter: character in ["c","j","s"]

    returns:
        data: array of shape (number of trajectories,number of timesteps,2)
        x: array of shape(number of trajectories*number of timesteps,2)
        xd: array of shape(number of trajectories*number of timesteps,2)

    """
    letter2id = dict(c=2, j=6, s=24)
    assert letter.lower() in letter2id
    _, x, _, _, _, _ = load_lasa(letter2id[letter.lower()])
    xd = derivative(x)
    plot_curves(x)
    data = x
    x = x.reshape(-1, 2)
    xd = xd.reshape(-1, 2)
    plt.show()
    return data, x, xd
