from jaxtyping import Float
import torch


def render_point_cloud(
    vertices: Float[torch.Tensor, "vertex 3"],
    extrinsics: Float[torch.Tensor, "batch 4 4"],
    intrinsics: Float[torch.Tensor, "batch 3 3"],
    resolution: tuple[int, int] = (256, 256),
) -> Float[torch.Tensor, "batch height width"]:
    """Create a white canvas with the specified resolution. Then, transform the points
    into camera space, project them onto the image plane, and color the corresponding
    pixels on the canvas black.
    """
    for view_number in range(0, len(extrinsics)):
        # create a white canvas
        canvas = torch.ones((3, resolution[0], resolution[1]))

        # Transform vertices into camera space
        w2c = torch.inverse(extrinsics[view_number])
        points_homo_w = torch.cat((vertices,torch.ones((vertices.shape[0],1))),1)
        points_homo_c =w2c@points_homo_w.t()

        # project them onto the image plane


    raise NotImplementedError("This is your homework.")
