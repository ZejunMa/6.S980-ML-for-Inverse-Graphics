from jaxtyping import Float
from torch import Tensor

from .field_dataset import FieldDataset


class FieldDatasetImplicit(FieldDataset):
    def query(
        self,
        coordinates: Float[Tensor, "batch d_coordinate"],
    ) -> Float[Tensor, "batch d_out"]:
        if self.cfg.function in ("sphere", "circle"):
            return ((coordinates - 0.5).norm(dim=-1, keepdim=True) < 0.4).float()
        elif self.cfg.function == "torus":
            x, y, z = (coordinates - 0.5).split((1, 1, 1), dim=-1)
            return (((x**2 + y**2).sqrt() - 0.25) ** 2 + z**2 < 0.15**2).float()
        else:
            raise ValueError(f'Unrecognized function "{self.cfg.function}"')

    @property
    def d_coordinate(self) -> int:
        return {
            "circle": 2,
            "sphere": 3,
            "torus": 3,
        }[self.cfg.function]

    @property
    def d_out(self) -> int:
        return 1

    @property
    def grid_size(self) -> tuple[int, ...]:
        """Return a semi-fine grid."""

        return (128,) * self.d_coordinate
