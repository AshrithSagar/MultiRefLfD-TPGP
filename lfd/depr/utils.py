"""
lfd/depr/utils.py \n
Utility functions and classes for the TPGP framework.
"""

from typing import List

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class Demonstration:
    """
    Class representing a single demonstration.

    Attributes:
        states (np.ndarray): Recorded positions (2D/3D), shape (T, D)
        progress (np.ndarray): Normalized time (0 to 1) for each state, shape (T,)
    """

    def __init__(self, states: np.ndarray, progress: np.ndarray):
        self.states = states
        self.progress = progress

    def get_data(self):
        """
        Returns the concatenated state and progress.
        """
        # Concatenating along the feature axis so that each data point becomes [position, progress]
        return np.hstack([self.states, self.progress.reshape(-1, 1)])


def transform_demo(
    demo: Demonstration, translation: np.ndarray, rotation: np.ndarray
) -> Demonstration:
    """
    Transforms a demonstration from the global frame to a local reference frame.

    Args:
        demo (Demonstration): The demonstration in the global frame.
        translation (np.ndarray): Translation vector for the frame (e.g., shape (D,)).
        rotation (np.ndarray): Rotation matrix for the frame (e.g., shape (D, D)).

    Returns:
        Demonstration: Transformed demonstration.
    """
    # Transform only the positions; progress remains unchanged.
    transformed_states = (rotation @ demo.states.T).T + translation
    return Demonstration(transformed_states, demo.progress)


class DataAlignment:
    """
    Provides methods to align multiple demonstrations.
    This is a simplified alignment using interpolation at fixed progress values.
    """

    @staticmethod
    def align(demos: List[Demonstration], num_points: int = 100) -> List[Demonstration]:
        aligned_demos = []
        fixed_progress = np.linspace(0, 1, num_points)
        for demo in demos:
            # Interpolate each coordinate dimension using progress as the independent variable.
            interp_states = np.zeros((num_points, demo.states.shape[1]))
            for d in range(demo.states.shape[1]):
                interp_states[:, d] = np.interp(
                    fixed_progress, demo.progress, demo.states[:, d]
                )
            aligned_demos.append(Demonstration(interp_states, fixed_progress))
        return aligned_demos


class LocalPolicyGP:
    """
    Represents a local policy learned using Gaussian Process Regression.

    Attributes:
        gp (GaussianProcessRegressor): The GP model for predicting state transitions.
    """

    def __init__(self, kernel=None):
        if kernel is None:
            kernel = Matern(nu=1.5)
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

    def train(self, X: np.ndarray, Y: np.ndarray):
        """
        Trains the GP on the provided data.

        Args:
            X (np.ndarray): Input features (e.g., state + progress), shape (n_samples, n_features)
            Y (np.ndarray): Labels (state differences), shape (n_samples, state_dim)
        """
        self.gp.fit(X, Y)

    def predict(self, x: np.ndarray):
        """
        Predicts the state transition for a given input x.

        Args:
            x (np.ndarray): Input feature vector, shape (n_features,)

        Returns:
            tuple: (mean, std) of the prediction.
        """
        mean, std = self.gp.predict(x.reshape(1, -1), return_std=True)
        return mean.ravel(), std


class FrameRelevanceGP:
    """
    Predicts the relevance of a given reference frame as a function of the progress variable.
    Uses Gaussian Process Regression.
    """

    def __init__(self, kernel=None):
        if kernel is None:
            kernel = Matern(nu=1.5)
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

    def train(self, progress: np.ndarray, relevance: np.ndarray):
        """
        Trains the relevance GP.

        Args:
            progress (np.ndarray): Array of progress values, shape (n_samples, 1).
            relevance (np.ndarray): Relevance scores for the frame, shape (n_samples,).
        """
        self.gp.fit(progress.reshape(-1, 1), relevance)

    def predict(self, progress: float):
        """
        Predicts the relevance for a given progress value.

        Args:
            progress (float): A progress value between 0 and 1.

        Returns:
            float: Predicted relevance.
        """
        pred, _ = self.gp.predict(np.array([[progress]]), return_std=True)
        return pred[0]


class TPGP:
    """
    Task-Parameterized Gaussian Process (TPGP) framework for learning multi-reference frame skills.

    This class holds demonstrations, local policies for each frame, and the frame relevance GP.
    """

    def __init__(self, frame_transforms: dict):
        """
        Args:
            frame_transforms (dict): Dictionary where keys are frame names/ids and values are tuples of (translation, rotation)
        """
        self.demonstrations = []
        self.frame_transforms = (
            frame_transforms  # e.g., {"frame1": (translation, rotation), ...}
        )
        self.local_policies = {}  # Each key corresponds to a frame.
        self.frame_relevance_gp = None

    def add_demonstration(self, demo: Demonstration):
        self.demonstrations.append(demo)

    def transform_demonstrations(self):
        """
        Transforms all demonstrations to each of the reference frames.

        Returns:
            dict: Dictionary with keys as frame names and values as lists of transformed Demonstration objects.
        """
        transformed = {}
        for frame, (translation, rotation) in self.frame_transforms.items():
            transformed[frame] = [
                transform_demo(demo, translation, rotation)
                for demo in self.demonstrations
            ]
        return transformed

    def align_demonstrations(
        self, demos: List[Demonstration], num_points: int = 100
    ) -> List[Demonstration]:
        """
        Aligns the provided demonstrations.
        """
        return DataAlignment.align(demos, num_points=num_points)

    def train_local_policies(self):
        """
        For each frame, trains a local policy GP on the aligned demonstration data.
        Assumes that the state change (delta) is the target.
        """
        transformed = self.transform_demonstrations()
        for frame, demos in transformed.items():
            # Align demonstrations for the given frame
            aligned = self.align_demonstrations(demos)

            # Collect training data: X is [state, progress], Y is the state difference (delta)
            X_train = []
            Y_train = []
            for demo in aligned:
                data = demo.get_data()  # shape (T, state_dim + 1)
                # Compute state transitions (simple finite differences)
                X_train.append(data[:-1])
                Y_train.append(demo.states[1:] - demo.states[:-1])
            X_train = np.vstack(X_train)
            Y_train = np.vstack(Y_train)

            # Initialize and train the local policy GP for this frame
            policy_gp = LocalPolicyGP()
            policy_gp.train(X_train, Y_train)
            self.local_policies[frame] = policy_gp
            print(
                f"Trained local policy GP for {frame} with {X_train.shape[0]} samples."
            )

    def train_frame_relevance(
        self, progress_data: np.ndarray, relevance_data: np.ndarray
    ):
        """
        Trains the frame relevance GP given progress values and target relevance scores.
        In practice, these relevance scores might be derived from the local policy predictions.
        """
        self.frame_relevance_gp = FrameRelevanceGP()
        self.frame_relevance_gp.train(progress_data, relevance_data)
        print("Trained frame relevance GP.")

    def predict(self, current_state: np.ndarray, progress: float):
        """
        Predicts the desired state transition using a weighted combination of local policies.

        Args:
            current_state (np.ndarray): The current state vector.
            progress (float): The current progress value.

        Returns:
            np.ndarray: The predicted state transition.
        """
        # Get local predictions for each frame
        predictions = {}
        for frame, gp in self.local_policies.items():
            X_input = np.hstack([current_state, [progress]])
            mean, std = gp.predict(X_input)
            predictions[frame] = (mean, std)

        # Compute relevance weights using the frame relevance GP (using a softmax)
        if self.frame_relevance_gp is not None:
            # Here, we assume that higher relevance scores indicate a better match.
            # In a more detailed implementation, one might compute a relevance for each frame.
            relevance = self.frame_relevance_gp.predict(progress)
            # For demonstration, we equally weight if only one GP is used.
            weights = {
                frame: 1.0 / len(self.local_policies) for frame in self.local_policies
            }
        else:
            weights = {
                frame: 1.0 / len(self.local_policies) for frame in self.local_policies
            }

        # Combine the predictions based on weights
        combined_delta = np.zeros_like(current_state)
        for frame in predictions:
            mean, _ = predictions[frame]
            combined_delta += weights[frame] * mean

        return combined_delta
