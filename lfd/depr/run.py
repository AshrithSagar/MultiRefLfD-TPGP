"""
lfd/depr/run.py \n
Run the TPGP model with two simple 2D demonstrations.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from utils import TPGP, Demonstration


def main():
    # Two 2D demos of a simple linear motion.
    T = 50
    demo1_states = np.linspace([0, 0], [10, 10], T)
    demo2_states = np.linspace([1, 1], [11, 11], T)
    progress = np.linspace(0, 1, T)

    demo1 = Demonstration(demo1_states, progress)
    demo2 = Demonstration(demo2_states, progress)

    frame_transforms = {
        "frame1": (np.array([0, 0]), np.eye(2)),
        "frame2": (
            np.array([2, -1]),
            Rotation.from_euler("z", 30, degrees=True).as_matrix()[:2, :2],
        ),
    }

    tpgp_model = TPGP(frame_transforms=frame_transforms)
    tpgp_model.add_demonstration(demo1)
    tpgp_model.add_demonstration(demo2)

    # Train local policies for each frame
    tpgp_model.train_local_policies()

    # For the frame relevance GP, we provide synthetic progress vs. relevance data.
    # In a realistic scenario, this would be derived from additional demonstrations.
    progress_data = np.linspace(0, 1, 100)
    relevance_data = (
        np.sin(2 * np.pi * progress_data) * 0.5 + 0.5
    )  # Example relevance scores between 0 and 1
    tpgp_model.train_frame_relevance(progress_data, relevance_data)

    # Make a prediction from the current state and progress
    current_state = np.array([5.0, 5.0])
    current_progress = 0.5
    delta = tpgp_model.predict(current_state, current_progress)
    print(f"Predicted state transition: {delta}")


if __name__ == "__main__":
    main()
