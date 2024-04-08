from jaxtyping import Float
import torch
from src.geometry import *

def render_point_cloud(
    vertices: Float[torch.Tensor, "vertex 3"],
    extrinsics: Float[torch.Tensor, "batch 4 4"],
    intrinsics: Float[torch.Tensor, "batch 3 3"],
    resolution: tuple[int, int] = (256, 256),
) -> torch.Tensor:
# ) -> Float[torch.Tensor, "batch height width"]:
    """Create a white canvas with the specified resolution. Then, transform the points
    into camera space, project them onto the image plane, and color the corresponding
    pixels on the canvas black.
    """
    canvas = torch.ones((extrinsics.shape[0], resolution[0], resolution[1])).type(torch.float32)

    for view_number in range(0, len(extrinsics)):
        # Transform vertices in world space into camera space
        points_c = transform_world2cam(homogenize_points(vertices), extrinsics[view_number])
        # project points camera space into image plane
        projected = project(points_c, intrinsics[view_number])
        # TODO: currently square image, arbitrary resolution to be fixed
        projected[:, 0] = projected[:, 0] * resolution[0]
        projected[:, 1] = projected[:, 1] * resolution[1]
        colored_map = projected.floor().type(torch.int)

        for pixel in colored_map:
            canvas[view_number][pixel[1].item()][pixel[0].item()] = 0.0
            # [view_number] = canvas[view_number].flip()

    return canvas





