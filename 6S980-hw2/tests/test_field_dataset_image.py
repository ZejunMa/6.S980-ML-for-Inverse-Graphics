import torch
from jaxtyping import install_import_hook
from omegaconf import DictConfig

# Add runtime type checking to all imports.
with install_import_hook(("src",), ("beartype", "beartype")):
    from src.dataset.field_dataset_image import FieldDatasetImage

    from .f32 import f32


def test_sampling():
    dataset = FieldDatasetImage(
        DictConfig(
            {
                "path": "data/tester.png",
            }
        )
    )

    coordinates = [
        [7 / 16, 7 / 16],
        [7 / 16, 9 / 16],
        [9 / 16, 7 / 16],
        [9 / 16, 9 / 16],
    ]

    expected = [
        [1, 0, 0],
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 0],
    ]

    assert torch.allclose(
        dataset.query(f32(coordinates)),
        f32(expected),
    )
