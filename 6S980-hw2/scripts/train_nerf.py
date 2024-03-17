import hydra
import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float, install_import_hook
from omegaconf import DictConfig
from torch import Tensor, nn
from tqdm import tqdm, trange

with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.dataset.dataset_nerf import DatasetNeRF
    from src.field import get_field
    from src.nerf import NeRF
    from src.projection import get_world_rays
    from src.sampling import sample_grid, sample_training_rays
    from src.visualization.annotation import add_label
    from src.visualization.camera import generate_spin
    from src.visualization.image import save_image
    from src.visualization.layout import add_border, hcat


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="train_nerf",
)
def train(cfg: DictConfig):
    # Set up the dataset, field, optimizer, and loss function.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = DatasetNeRF(cfg.dataset)
    field = get_field(cfg, 3, 4).to(device)
    nerf = NeRF(cfg.nerf, field)
    optimizer = torch.optim.Adam(
        field.parameters(),
        lr=cfg.learning_rate,
    )
    loss_fn = nn.MSELoss()

    def render_image(
        h: int,
        w: int,
        extriniscs: Float[Tensor, "4 4"],
        intrinsics: Float[Tensor, "3 3"],
    ) -> Float[Tensor, "3 height width"]:
        # Generate a grid of training rays.
        xy = sample_grid((h, w))
        origins, directions = get_world_rays(xy, extriniscs, intrinsics)

        # Render the image.
        size = cfg.batch_size
        predicted = [
            nerf(
                origin_batch.to(device),
                direction_batch.to(device),
                dataset.near,
                dataset.far,
            )
            for origin_batch, direction_batch in zip(
                rearrange(origins, "h w xy -> (h w) xy").split(size, dim=0),
                rearrange(directions, "h w xy -> (h w) xy").split(size, dim=0),
            )
        ]
        predicted = torch.cat(predicted, dim=0)
        predicted = rearrange(predicted, "(h w) c -> c h w", h=h, w=w)
        return predicted

    # Define the resolution at which previews and final images are rendered.
    _, _, h, w = dataset.images.shape
    vis_h, vis_w = (h // 4, w // 4)

    # Fit the field to the dataset.
    for iteration in (progress := trange(cfg.num_iterations)):
        optimizer.zero_grad()

        origins, directions, ground_truth = sample_training_rays(
            dataset.images,
            dataset.extrinsics,
            dataset.intrinsics,
            cfg.batch_size,
            device,
        )

        predicted = nerf(origins, directions, dataset.near, dataset.far)

        loss = loss_fn(predicted, ground_truth)
        loss.backward()
        optimizer.step()

        # Intermittently visualize training progress.
        if iteration % cfg.visualization_interval == 0:
            with torch.no_grad():
                # Pick a random view to render.
                b, _, _, _ = dataset.images.shape
                rb = torch.randint(b, (1,)).item()

                predicted = render_image(
                    vis_h,
                    vis_w,
                    dataset.extrinsics[rb],
                    dataset.intrinsics[rb],
                )

                # Create a side-by-side visualization.
                ground_truth = F.interpolate(
                    dataset.images[rb][None],
                    (vis_h, vis_w),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                visualization = add_border(
                    hcat(
                        add_label(ground_truth, "Ground Truth"),
                        add_label(predicted, "Predicted"),
                    )
                )
                save_image(
                    visualization, f"{cfg.output_path}/progress/{iteration:0>6}.png"
                )

        progress.desc = f"Training (loss: {loss.item():.4f})"

    # Once training is done, render a spin.
    radius = 0.5 * (dataset.near + dataset.far)
    for i, c2w in enumerate(tqdm(generate_spin(30, 30.0, radius), desc="Rendering")):
        c2w[:3, 3] += 0.5
        image = render_image(vis_h, vis_w, c2w, dataset.intrinsics[0])
        save_image(image, f"{cfg.output_path}/spin/{i:0>6}.png")


if __name__ == "__main__":
    train()
