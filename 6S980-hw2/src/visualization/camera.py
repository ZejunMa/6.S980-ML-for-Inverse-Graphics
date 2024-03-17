import numpy as np
import torch
from einops import repeat
from jaxtyping import Float
from scipy.spatial.transform import Rotation as R
from torch import Tensor


def generate_spin(
    num_steps: int,
    elevation: float,
    radius: float,
    device: torch.device = torch.device("cpu"),
) -> Float[Tensor, "batch 4 4"]:
    # Translate back along the camera's look vector.
    tf_translation = torch.eye(4, dtype=torch.float32, device=device)
    tf_translation[2, 3] = -radius
    tf_translation[:2, :2] *= -1  # Use +Y as world up instead of -Y.

    # Generate the transformation for the azimuth.
    t = np.linspace(0, 1, num_steps, endpoint=False)
    azimuth = [
        R.from_rotvec(np.array([0, x * 2 * np.pi, 0], dtype=np.float32)).as_matrix()
        for x in t
    ]
    azimuth = torch.tensor(np.array(azimuth), dtype=torch.float32, device=device)
    tf_azimuth = torch.eye(4, dtype=torch.float32, device=device)
    tf_azimuth = repeat(tf_azimuth, "i j -> b i j", b=num_steps).clone()
    tf_azimuth[:, :3, :3] = azimuth

    # Generate the transformation for the elevation.
    deg_elevation = np.deg2rad(elevation)
    elevation = R.from_rotvec(np.array([deg_elevation, 0, 0], dtype=np.float32))
    elevation = torch.tensor(elevation.as_matrix())
    tf_elevation = torch.eye(4, dtype=torch.float32, device=device)
    tf_elevation[:3, :3] = elevation

    return tf_azimuth @ tf_elevation @ tf_translation
