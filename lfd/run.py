"""
lfd/run.py \n
Trial run
"""

import lfd


def main():
    D0, _ = lfd.utils.load_data_with_phi("s")
    fdset = lfd.utils.transform_data(D0)

    P = lfd.alignment.computeP(fdset)
    D0_star = lfd.alignment.align_demonstrations(fdset, P)

    lfd.alignment.plot_keypoints(fdset, P)
    lfd.alignment.plot_alignments(fdset, D0_star, P)

    X = lfd.utils.transform_data(D0_star)


if __name__ == "__main__":
    main()
