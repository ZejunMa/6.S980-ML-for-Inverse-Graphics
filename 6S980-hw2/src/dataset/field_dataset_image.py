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
        coordinates = (coordinates * 2 - 1).unsqueeze(0).unsqueeze(2)
        print(coordinates.shape)
        sampled_colors = grid_sample(self.image, coordinates, mode='bilinear')
        print(sampled_colors.shape)
        output = sampled_colors.squeeze(0).squeeze(-1).permute(1, 0)
        return output  # batch d_out

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

if __name__ == "__main__":
    dataset = FieldDatasetImage(
        DictConfig({"path": "data/tester.png"}))
