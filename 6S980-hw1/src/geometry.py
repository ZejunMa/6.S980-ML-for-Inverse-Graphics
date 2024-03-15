from jaxtyping import Float
from torch import Tensor, cat, ones, inverse


def homogenize_points(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional points into (n+1)-dimensional homogeneous points."""
    # return torch.cat([points, torch.tensor([1.0])])
    if points.ndim > 1:
        return cat((points,ones((points.shape[0], 1))), 1)
    return cat([points, Tensor([1.0])])


def homogenize_vectors(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional vectors into (n+1)-dimensional homogeneous vectors."""
    # return torch.cat([points, torch.tensor([0.0])])
    if points.ndim > 1:
        return cat((points,ones((points.shape[0], 0))), 1)
    return cat([points, Tensor([0.0])])


def transform_rigid(
    xyz: Float[Tensor, "*#batch 4"],
    transform: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Apply a rigid-body transform to homogeneous points or vectors."""
    # return torch.mm(transform, xyz.unsqueeze(1)).squeeze(1)
    if xyz.ndim > 1:
        return (transform @ xyz.t()).t()
    return transform @ xyz


def transform_world2cam(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D world coordinates to homogeneous
    3D camera coordinates.
    """
    # return torch.mm(torch.inverse(cam2world), xyz.unsqueeze(1)).squeeze(1)
    return transform_rigid(xyz, inverse(cam2world))


def transform_cam2world(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D camera coordinates to homogeneous
    3D world coordinates.
    """
    # return torch.mm(cam2world, xyz.unsqueeze(1)).squeeze(1)
    return transform_rigid(xyz, cam2world)


def project(
    xyz: Float[Tensor, "*#batch 4"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch 2"]:
    """Project homogenized 3D points in camera coordinates to pixel coordinates."""
    # projected = torch.mm(intrinsics, xyz[:-1].unsqueeze(1)).squeeze(1)
    # return torch.tensor([projected[0] / projected[-1], projected[1] / projected[-1]])
    if xyz.ndim == 1:
        projected = (intrinsics @ xyz[:-1].unsqueeze(1)).squeeze(1)
        return Tensor([projected[0] / projected[-1], projected[1] / projected[-1]])
    else:
        projected = (intrinsics @ (xyz.t()[:3, :])).t()
        for i in range(0, xyz.shape[0]):
            projected[i] = projected[i]/projected[i][2]
        return projected[:, 0:2]


# xyz = Tensor([[1, 1, 2, 1],[1, 1, 2, 1]])
# intrinsics = Tensor([[0.25, 0, 0.5], [0, 0.25, 0.5], [0, 0, 1]])
# projected = project(xyz, intrinsics)
# print(projected)
# f32([0.625, 0.625])
