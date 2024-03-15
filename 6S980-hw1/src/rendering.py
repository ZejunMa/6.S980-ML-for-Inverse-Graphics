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
        # Transform vertices into camera space
        w2c = torch.inverse(extrinsics[view_number])
        points_c = transform_world2cam(homogenize_points(vertices), w2c)

        # project them onto the image plane
        projected = project(points_c, intrinsics[view_number])
        colored_map = (projected * 256).floor().type(torch.uint8)

        for pixel in colored_map:
            canvas[view_number][pixel[1].item()][pixel[0].item()] = 0.0

    return canvas





