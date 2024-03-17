from omegaconf import DictConfig

from .field_dataset import FieldDataset
from .field_dataset_image import FieldDatasetImage
from .field_dataset_implicit import FieldDatasetImplicit

FIELD_DATASETS: dict[str, FieldDataset] = {
    "image": FieldDatasetImage,
    "implicit": FieldDatasetImplicit,
}


def get_field_dataset(cfg: DictConfig) -> FieldDataset:
    return FIELD_DATASETS[cfg.dataset.name](cfg.dataset)
