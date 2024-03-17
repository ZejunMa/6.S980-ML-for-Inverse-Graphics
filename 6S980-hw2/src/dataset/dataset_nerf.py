import json
from math import tan
from pathlib import Path

import torch
import torchvision.transforms as tf
from einops import repeat
from jaxtyping import Float
from omegaconf import DictConfig
from PIL import Image
from torch import Tensor
from tqdm import tqdm


class DatasetNeRF:
    images: Float[Tensor, "batch 3 height width"]
    extrinsics: Float[Tensor, "batch 4 4"]
    intrinsics: Float[Tensor, "batch 3 3"]
    near: float
    far: float

    def __init__(self, cfg: DictConfig) -> None:
        path = Path(cfg.path) / cfg.scene

        # Load the metadata.
        transforms = tf.ToTensor()
        with (path / "transforms_train.json").open("r") as f:
            metadata = json.load(f)

        # This converts the extrinsics to OpenCV style.
        conversion = torch.eye(4, dtype=torch.float32)
        conversion[1:3, 1:3] *= -1

        # Read the images and extrinsics.
        images = []
        extrinsics = []
        for frame in tqdm(metadata["frames"], "Loading frames"):
            extrinsics.append(
                torch.tensor(frame["transform_matrix"], dtype=torch.float32)
                @ conversion
            )

            # Read the image.
            image = Image.open(path / f"{frame['file_path']}.png")

            # Composite the image onto a black background.
            background = Image.new("RGB", image.size, (0, 0, 0)).convert("RGBA")
            rgb = Image.alpha_composite(background, image)
            rgb = transforms(rgb.convert("RGB"))

            images.append(rgb)
        self.images = torch.stack(images)
        self.extrinsics = torch.stack(extrinsics)
        self.extrinsics[:, :3, 3] *= cfg.scale_factor
        self.extrinsics[:, :3, 3] += torch.tensor(cfg.offset, dtype=torch.float32)

        # Rotate the extrinsics so that +Y is the world up vector.
        make_y_up = torch.tensor(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )
        self.extrinsics = make_y_up @ self.extrinsics

        # Convert the intrinsics to (normalized) OpenCV style.
        camera_angle_x = float(metadata["camera_angle_x"])
        focal_length = 0.5 / tan(0.5 * camera_angle_x)
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics[:2, :2] *= focal_length
        intrinsics[:2, 2] = 0.5
        self.intrinsics = repeat(intrinsics, "i j -> b i j", b=self.extrinsics.shape[0])

        # Use near/far from the original nerf paper.
        self.near = 2.0 * cfg.scale_factor
        self.far = 6.0 * cfg.scale_factor
