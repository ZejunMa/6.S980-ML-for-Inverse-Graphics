from omegaconf import DictConfig

from .field import Field
from .field_grid import FieldGrid
from .field_ground_plan import FieldGroundPlan
from .field_hybrid_grid import FieldHybridGrid
from .field_mlp import FieldMLP
from .field_siren import FieldSiren

FIELDS: dict[str, Field] = {
    "grid": FieldGrid,
    "ground_plan": FieldGroundPlan,
    "hybrid_grid": FieldHybridGrid,
    "mlp": FieldMLP,
    "siren": FieldSiren,
}


def get_field(
    cfg: DictConfig,
    d_coordinate: int,
    d_out: int,
) -> Field:
    return FIELDS[cfg.field.name](cfg.field, d_coordinate, d_out)
