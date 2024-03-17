from abc import ABC, abstractmethod

from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor, nn


class Field(nn.Module, ABC):
    cfg: DictConfig
    d_coordinate: int  # input coordinate dimensionality (in our case, 2D or 3D)
    d_out: int  # output dimensionality (e.g., 1D for occupancy, 3D for color, etc.)

    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_coordinate = d_coordinate
        self.d_out = d_out

    @abstractmethod
    def forward(
        self,
        coordinates: Float[Tensor, "batch d_coordinate"],
    ) -> Float[Tensor, "batch d_out"]:
        pass
