from pathlib import Path

import hydra
import torch
from jaxtyping import install_import_hook
from omegaconf import DictConfig
from torch import nn
from tqdm import trange

with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.dataset import get_field_dataset
    from src.field import get_field
    from src.sampling import generate_random_samples, sample_grid
    from src.visualization.image import save_rendered_field


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="train_field",
)
def train(cfg: DictConfig):
    # Set up the dataset, field, optimizer, and loss function.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = get_field_dataset(cfg)
    field = get_field(cfg, dataset.d_coordinate, dataset.d_out).to(device)
    optimizer = torch.optim.Adam(
        field.parameters(),
        lr=cfg.learning_rate,
    )
    loss_fn = nn.MSELoss()

    # Optionally re-map the outputs to the neural field so they're in range [0, 1].
    if cfg.remap_outputs:
        field = nn.Sequential(
            field,
            nn.Sigmoid(),
        )

    # Fit the field to the dataset.
    for iteration in (progress := trange(cfg.num_iterations)):
        optimizer.zero_grad()
        samples = generate_random_samples(dataset.d_coordinate, cfg.batch_size, device)
        predicted = field(samples)
        ground_truth = dataset.query(samples)
        loss = loss_fn(predicted, ground_truth)
        loss.backward()
        optimizer.step()

        # Intermittently visualize training progress.
        if iteration % cfg.visualization_interval == 0:
            with torch.no_grad():
                # Render the field in a grid.
                samples = sample_grid(dataset.grid_size, device)
                *dimensions, d_coordinate = samples.shape
                samples = samples.reshape(-1, d_coordinate)
                values = [field(batch) for batch in samples.split(cfg.batch_size)]
                values = torch.cat(values).reshape(*dimensions, -1)

                # Save the result.
                path = Path(f"{cfg.output_path}/{iteration:0>6}.png")
                save_rendered_field(values, path)

        progress.desc = f"Training (loss: {loss.item():.4f})"


if __name__ == "__main__":
    train()
