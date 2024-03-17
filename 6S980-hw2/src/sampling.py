import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from .projection import get_world_rays


def generate_random_samples(
    d_coordinate: int,
    num_samples: int,
    device: torch.device,
) -> Float[Tensor, "batch d_coordinate"]:
    """Generate random samples in the range [0, 1]."""

    return torch.rand(
        (num_samples, d_coordinate),
        dtype=torch.float32,
        device=device,
    )


def sample_grid(
    shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
) -> Float[Tensor, "*dimensions d_coordinate"]:
    """Generate samples in an n-dimensional grid, where n is equal to d_coordinate.
    Samples span the range [0, 1] in each dimension.
    """

    # Generate linearly spaced coordinates in each dimension. The 0.5 offset is added so
    # that the centers of grid cells (2D) or voxels (3D) are queried.
    coordinates = [
        (torch.arange(length, dtype=torch.float32, device=device) + 0.5) / length
        for length in shape
    ]

    # Turn the coordinates into a grid.
    coordinates = torch.meshgrid(*coordinates, indexing="xy")
    return torch.stack(coordinates, dim=-1)


def sample_training_rays(
    image: Float[Tensor, "batch 3 height width"],
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    num_rays: int,
    device: torch.device,
) -> tuple[
    Float[Tensor, "ray 3"],  # origins
    Float[Tensor, "ray 3"],  # directions
    Float[Tensor, "ray 3"],  # sampled color
]:
    b, _, h, w = image.shape

    # Select a random image.
    index = torch.randint(0, b, tuple()).item()
    image = image[index]
    extrinsics = extrinsics[index]
    intrinsics = intrinsics[index]

    # Select random coordinates.
    xy = torch.rand((num_rays, 2), dtype=torch.float32)

    # Generate rays for the coordinates.
    origins, directions = get_world_rays(
        xy,
        extrinsics,
        intrinsics,
    )

    # Grab the pixel values at the coordinates. It would be better to interpolate, but
    # it does't really matter for a homework assignment.
    wh = torch.tensor((w, h), dtype=torch.float32)
    col, row = (xy * wh).type(torch.int64).unbind(dim=-1)
    pixels = rearrange(image[:, row, col], "c r -> r c")

    return tuple(x.to(device) for x in (origins, directions, pixels))
