"""
lfd/run.py \n
Trial run
"""

import gpytorch
import numpy as np
import torch

import lfd


def main():
    lfd.set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    D0, _ = lfd.utils.load_data_with_phi("s")

    fdset = lfd.utils.transform_data(D0)
    P = lfd.alignment.computeP(fdset)
    D0_star = lfd.alignment.align_demonstrations(fdset, P)

    lfd.alignment.plot_keypoints(fdset, P)
    lfd.alignment.plot_alignments(fdset, D0_star, P)

    X = lfd.utils.transform_data(D0_star)

    Y = np.empty_like(X)
    for m, Dm in enumerate(X):
        Y[m] = lfd.utils.utils.vectorized_derivative(Dm)

    As, Bs = lfd.utils.get_frames(D0)

    X = lfd.utils.utils.toTensor(X, device=device)
    Y = lfd.utils.utils.toTensor(Y, device=device)
    As = lfd.utils.utils.toTensor(As, device=device)
    Bs = lfd.utils.utils.toTensor(Bs, device=device)

    n_frames, n_traj, n_length, n_dim = X.shape

    X_train, X_val = X[1:], X[0]
    Y_train, Y_val = Y[1:], Y[0]

    lgp = lfd.gp.LocalPolicyGP(X_train, Y_train, num_inducing=500)
    lgp.train(num_epochs=100, lr=0.01)
    lgp.predict()


if __name__ == "__main__":
    main()
