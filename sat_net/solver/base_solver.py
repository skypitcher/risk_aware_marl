from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from sat_net.datablock import DataBlock


class BaseSolver(ABC):
    """Base class for all solvers."""

    def __init__(self, tf_writer: Optional[SummaryWriter] = None):
        self._tf_writer = tf_writer
        self._is_eval = False

    @property
    @abstractmethod
    def name(self):
        pass

    @staticmethod
    def class_name(cls):
        return cls.__class__.__name__

    @abstractmethod
    def route(self, obs: np.ndarray, info: dict) -> tuple[Optional[int], Optional[dict]]:
        """
        Determine the next hop for a DataBlock at a given node.

        Args:
            obs: The observation.
            info: The info provided by the environment.

        Returns:
            next_hop: The index of the next_hop, or None if no route is found or if the agent decides to drop.
            info: The info dict provided by the agent.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def on_action_over(self, packet: "DataBlock"):
        """
        Callback when a delayed action is over.
        """
        pass

    def on_episode_over(self, packet: "DataBlock"):
        """
        Callback when an asynchronous episode over (packet delivered or dropped).
        """
        pass

    def on_train_signal(self):
        """
        Callback when a periodical training event is triggered by the environment.
        """
        pass

    def set_train(self):
        """Set the solver to training mode (e.g., enable exploration, learning)."""
        self._is_eval = False

    def set_eval(self):
        """Set the solver to evaluation mode (e.g., disable exploration, use greedy policy)."""
        self._is_eval = True
    
    def is_train(self):
        """Check if the solver is in training mode."""
        return not self._is_eval
    
    def is_eval(self):
        """Check if the solver is in evaluation mode."""
        return self._is_eval

    def save_models(self, model_dir_path: str):
        """save the model to the given path. Used by RL-based solvers."""
        pass

    def load_models(self, model_dir_path: str):
        """load the model from the given path. Used by RL-based solvers."""
        pass

    def get_stats(self) -> str:
        pass
