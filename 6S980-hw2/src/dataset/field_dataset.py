from abc import ABC, abstractmethod

from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor


class FieldDataset(ABC):
    cfg: DictConfig

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def query(
        self,
        coordinates: Float[Tensor, "batch d_coordinate"],
    ) -> Float[Tensor, "batch d_out"]:
        pass

    @property
    @abstractmethod
    def d_coordinate(self) -> int:
        """The dimensionality of the coordinates the dataset is queried with."""
        pass

    @property
    @abstractmethod
    def d_out(self) -> int:
        """The dimensionality of the dataset's per-query output."""
        pass

    @property
    @abstractmethod
    def grid_size(self) -> tuple[int, ...]:
        """A reasonable grid size for visualizing the dataset."""
        pass
