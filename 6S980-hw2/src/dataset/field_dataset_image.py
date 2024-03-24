from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

from  torch.nn.functional import grid_sample

import cv2

from .field_dataset import FieldDataset


class FieldDatasetImage(FieldDataset):
    def __init__(self, cfg: DictConfig) -> None:
        """Load the image in cfg.path into memory here."""
        super().__init__(cfg)
        self.image = Tensor(cv2.imread(cfg.path).transpose((2, 0, 1))).unsqueeze(0)


    def query(
        self,
        coordinates: Float[Tensor, "batch d_coordinate"],
    ) -> Float[Tensor, "batch d_out"]:
        """Sample the image at the specified coordinates and return the corresponding
        colors. Remember that the coordinates will be in the range [0, 1].

        You may find the grid_sample function from torch.nn.functional helpful here.
        Pay special attention to grid_sample's expected input range for the grid
        parameter.
        """
        # print(self.image.size())  # size = torch.Size([1, 3, 128, 128])
        coordinates = (coordinates * 2 - 1).unsqueeze(0).unsqueeze(1)
        # print(coordinates.shape)  # size = torch.Size([1, 4, 1, 2])

        sampled_colors = (grid_sample(self.image, coordinates, mode='bilinear', align_corners=True) / 255.0).squeeze(0).squeeze(1).flip(0).t()
        return sampled_colors 

    @property
    def d_coordinate(self) -> int:
        return 2

    @property
    def d_out(self) -> int:
        return 3

    @property
    def grid_size(self) -> tuple[int, ...]:
        """Return a grid size that corresponds to the image's shape."""
        return self.image.size()[1:]
