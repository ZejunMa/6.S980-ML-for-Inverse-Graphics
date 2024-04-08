import torch
from einops import repeat
from jaxtyping import install_import_hook

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# Add runtime type checking to all imports.

with install_import_hook(("src",), ("beartype", "beartype")):
    from src.provided_code import generate_spin, get_bunny, save_image
    from src.rendering import render_point_cloud

if __name__ == "__main__":
    vertices, faces = get_bunny()

    # Generate a set of camera extrinsics for rendering. (code for generating camera extrinsics)
    NUM_STEPS = 16
    c2w = generate_spin(NUM_STEPS, 15.0, 2.0)

    # Generate a set of camera intrinsics for rendering.
    # 这里对相机内参进行了简化, 相机参考readme对于内参的矩阵解读
    k = torch.eye(3, dtype=torch.float32)
    k[:2, 2] = 0.5
    k = repeat(k, "i j -> b i j", b=NUM_STEPS)

    # Render the point cloud.
    images = render_point_cloud(vertices, c2w, k, resolution =(2048, 2048))

    # Save the resulting images.
    import numpy as np
    torch.set_printoptions(threshold=np.inf)
    for index, image in enumerate(images):
        save_image(image, f"outputs/1_projection/view_{index:0>2}.png")
