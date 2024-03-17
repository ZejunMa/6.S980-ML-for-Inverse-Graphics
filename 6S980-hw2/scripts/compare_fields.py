import os
import re
from pathlib import Path

from PIL import Image
from torchvision.transforms import ToTensor

from src.visualization.annotation import add_label
from src.visualization.image import save_image
from src.visualization.layout import add_border, hcat

commands = {
    "MLP (w/o Encoding)": "field=mlp dataset=image remap_outputs=true field.positional_encoding_octaves=null field.num_hidden_layers=2 field.d_hidden=128",
    "MLP (w/ Encoding)": "field=mlp dataset=image remap_outputs=true field.positional_encoding_octaves=6 field.num_hidden_layers=2 field.d_hidden=128",
    "Grid (32x32)": "field=grid dataset=image remap_outputs=true field.side_length=33",
    "Hybrid Grid (32x32)": "field=hybrid_grid dataset=image remap_outputs=true field.d_grid_feature=128 field.grid.side_length=33 field.mlp.positional_encoding_octaves=null field.mlp.num_hidden_layers=2 field.mlp.d_hidden=64",
    "SIREN": "field=siren dataset=image remap_outputs=false learning_rate=1e-4",
    "Ground Plan": "field=ground_plan dataset=implicit dataset.function=torus field.d_grid_feature=128 field.positional_encoding_octaves=8 field.grid.side_length=65 field.mlp.positional_encoding_octaves=null field.mlp.num_hidden_layers=2 field.mlp.d_hidden=64",
}

NUM_STEPS = 2000

if __name__ == "__main__":
    images = []
    to_tensor = ToTensor()
    for title, parameters in commands.items():
        print(f"Training {title}")
        key = re.sub(r"\W+", "", title)
        output_path = Path(f"compare/{key}")
        os.system(
            f"python3 -m scripts.train_field {parameters} output_path={output_path} num_iterations={NUM_STEPS} visualization_interval={NUM_STEPS - 1} batch_size=512"
        )
        image = to_tensor(Image.open(output_path / f"{NUM_STEPS - 1:0>6}.png"))
        image = add_label(image, title)
        images.append(image)
    images = add_border(hcat(*images, align="top"))
    save_image(images, "compare/comparison.png")
