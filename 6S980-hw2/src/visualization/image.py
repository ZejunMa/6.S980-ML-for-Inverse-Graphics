from pathlib import Path
from typing import Union

import numpy as np
import torch
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor

from .annotation import add_label
from .layout import add_border, hcat

FloatImage = Union[
    Float[Tensor, "height width"],
    Float[Tensor, "channel height width"],
    Float[Tensor, "batch channel height width"],
]


def prep_image(image: FloatImage) -> UInt8[np.ndarray, "height width channel"]:
    # Handle batched images.
    if image.ndim == 4:
        image = rearrange(image, "b c h w -> c h (b w)")

    # Handle single-channel images.
    if image.ndim == 2:
        image = rearrange(image, "h w -> () h w")

    # Ensure that there are 3 or 4 channels.
    channel, _, _ = image.shape
    if channel == 1:
        image = repeat(image, "() h w -> c h w", c=3)
    assert image.shape[0] in (3, 4)

    image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
    return rearrange(image, "c h w -> h w c").cpu().numpy()


def save_image(
    image: FloatImage,
    path: Union[Path, str],
) -> None:
    """Save an image. Assumed to be in range 0-1."""

    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Save the image.
    Image.fromarray(prep_image(image)).save(path)


def save_rendered_field(values: Float[Tensor, "*dimensions d_out"], path: Path) -> None:
    if values.ndim == 3:
        image = rearrange(values, "h w c -> c h w")
        if image.shape[0] == 1:
            image = repeat(image, "() h w -> c h w", c=3)
        save_image(image, path)
    elif values.ndim == 4:
        halfway = [length // 2 for length in values.shape[:3]]

        # Save slices.
        slices = []
        for x_axis in range(3):
            y_axis = (x_axis + 1) % 3
            slice_axis = (x_axis + 1) % 3
            selector = [slice(None)] * 3
            selector[slice_axis] = halfway[slice_axis]
            value_slice = repeat(values[selector], "h w () -> c h w", c=3)
            label = f"{'XYZ'[x_axis]}{'XYZ'[y_axis]} Slice"
            slices.append(add_label(value_slice, label))

        save_image(add_border(hcat(*slices)), path)
    else:
        raise NotImplementedError(f"Cannot save results with {values.ndim} dimensions!")
