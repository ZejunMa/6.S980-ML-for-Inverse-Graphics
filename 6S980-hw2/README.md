# 6.S980 Homework 2

## Getting Started

**Using Python 3.9 or newer,** create a virtual environment as follows:

```
python3 -m venv venv
source venv/bin/activate
```

Next, install PyTorch using the instructions [here](https://pytorch.org/get-started/locally/). Select pip as the installation method. **If you're on Linux and have a CUDA-capable GPU, select the latest CUDA version.** This will give you a command like this:

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Finally, install this homework's dependencies:

```
pip install -r requirements.txt
```

You can now open the project directory in VS Code. Within VS Code, open the command palette (<key>⌘ command</key> <key>⇧ shift</key> <key>P</key>), run `Python: Select Interpreter`, and choose the virtual environment you created in the previous steps.

For the best editing experience, install the following VS Code extensions:

- Python (`ms-python.python`)
- Pylance (`ms-python.vscode-pylance`)
- Black Formatter (`ms-python.black-formatter`)
- Ruff (`charliermarsh.ruff`)

## Part 1: Neural Fields

In physics, functions that output values for space-time input coordinates are referred to as **fields**. Following [Neural Fields in Visual Computing and Beyond, Xie et al., 2022](https://arxiv.org/abs/2111.11426), we refer to fields that are fully or partially parametrized by neural networks as **neural fields**. We will use neural fields to parametrize the following functions:

- **2D Images:** $f: \mathbb{R}^2 \to \mathbb{R}^3$, where outputs are RGB colors
- **3D Occupancy:** $f: \mathbb{R}^3 \to \mathbb{R}$, where outputs represent "density"

In this part of the project, you'll implement a variety of neural field architectures.

### Image Datasets

To get started, you'll have to implement `FieldDatasetImage` in `src/dataset/field_dataset_image.py`. This dataset functions differently from regular PyTorch image datasets that you may be used to: instead of returning whole images, it defines a way to query a single image at particular locations using the `query` function. In other words, `FieldDatasetImage` implements a field that represents a particular image!

Before moving on, we recommend running `test_sampling` in `tests/test_field_dataset_image.py`.

### Multilayer Perceptron (MLP) Neural Field

One of the simplest neural field architectures one can implement is an MLP, which you must implement in `src/field/field_mlp.py`.

#### Hydra Configurations

This homework's scripts use [Hydra](https://hydra.cc/docs/intro/) to manage configuration options. Hydra makes it easy to define hierarchical configurations and override them from the command line when running a script. You might notice that the `Train Field` run configuration in `.vscode/launch.json` has an `args` field with two arguments: `field=mlp` and `dataset=image`. These are command line arguments that are appended to the command VS Code runs, yielding the following:

```
python3 -m scripts.train_field field=mlp dataset=image
```

Both of these arguments specify overrides in Hydra's [override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/).

#### Positional Encodings

If you implement the MLP field without positional encodings and run it, you'll probably notice that it struggles to represent high-frequency details. To fix this, you can encode the MLP's unputs using a positional encoding. You must implement one in `src/components/positional_encoding.py`. If you're interested in learning more about positional encodings, check out [Fourier Features Let Networks Learn
High Frequency Functions in Low Dimensional Domains](https://bmild.github.io/fourfeat/).

### SIREN Neural Field

Next up, you'll be constructing a [SIREN](https://www.vincentsitzmann.com/siren/) network. This is a neural field whose architecture makes it well-suited for working with derivatives and integrals of functions. In our case, it serves as an alternative to using positional encodings. Fun fact: Vincent is the author of the original SIREN paper.

### Explicit and Hybrid Neural Fields

Neural fields can be categorized into two categories—explicit and implicit—as follows:

- **Implicit:** Coordinates are mapped to values via neural networks.
- **Explicit:** Coordinates are mapped to values via spatial data structures (grids, octrees, point clouds, etc.).

You'll have to implement a grid-based (explicit) neural field in `src/field/field_grid.py`. Then, you'll combine this neural field with your existing MLP field to create two hybrid implicit-explicit neural fields: `FieldHybridGrid` in `src/field/field_hybrid_grid.py` and FieldGroundPlan `src/field/field_ground_plan.py`.

Note that `FieldGroundPlan` is a neural field designed for 3D inputs. It does not need to support 2D inputs.

### 3D Neural Fields

To test out your 3D neural fields, you can use `FieldDatasetImplicit` in `src/dataset/field_dataset_implicit.py`. This dataset implements a few geometric shapes in two and three dimensions. You can enable the dataset by appending `dataset=implicit` to your terminal command or VS Code launch configuration. You can select between the shapes by specifying `dataset.function=<shape>`.

### Comparison Script

Once you've implemented your fields, you can use `scripts/compare_fields.py` to visualize them side-by-side.

## Part 2: Neural Radiance Fields (NeRFs)

In this part of the assignment, you'll implement a neural radiance field (NeRF). This will allow you to use 2D inputs (images) to generate 3D fields via differentiable rendering. Specifically, you'll implement a simplified version of volume rendering as described in [Neural Radiance Fields, Mildenhall et al., 2020](https://arxiv.org/abs/2003.08934). Your neural field will parametrize the following function:

$$
\Phi: \mathbb{R}^3 \to \mathbb{R}^3 \times \mathbb{R}^+, \quad \Phi(\mathbf{x}) = (\sigma, \mathbf{c})
$$

This function maps a 3D coordinate $\mathbf{x}$ to a tuple of emitted radiance $\mathbf{c}$ and local density $\sigma$. Note that you can use any of your 3D-compatible neural fields to parametrize the neural radiance field!

### Downloading the Dataset

Download the `nerf_synthetic` dataset [from here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1). Place it at `data/nerf_synthetic`.

### NeRF Rendering Procedure

To render an image, the NeRF model does the following:

- For each pixel that should be rendered, generate a camera ray origin and direction.
- Generate samples along the camera rays. You will implement this in `generate_samples()`.
- Evaluate the neural field at the sample locations to obtain color (radiance) and volumetric density.
- Convert the volumetric densities to alpha values. You will implement this in `compute_alpha_values()` using the formula below. In this formula, $\alpha_i$ is alpha, $\sigma_i$ is volumetric density, and $\delta_i$ is the segment length (distance between consecutive samples) corresponding to the density sample.

$$
\alpha_i = 1 - \exp(-\sigma_i \delta_i)
$$

- Composite the aforementioned colors and alpha values. You will implement this in `alpha_composite()`. To do so, compute transmittance as follows:

$$
T_i = \prod_{j=1}^{i-1} (1 - \alpha_j)
$$

Given transmittance and alphas, you can then compute weights $w_i$ as follows:

$$
w_i = \alpha_i T_i
$$

You can then compute the expected radiance along the ray as:

$$
\mathbf{c} = \Sigma_{i=1}^n w_i \mathbf{c}_i
$$

### Note on GPUs

This assignment is designed to work on any relatively modern computer. However, NeRF training runs significantly faster on CUDA devices. For example, training the NeRF takes about 1 minute on a 3090 Ti (graphics card) and 10 minutes on an M2 Macbook Pro (CPU).

## Running Tests

We've included some tests that you can use as a sanity check. These tests only test basic functionality—it's up to you to ensure that your code can handle more complex use cases (e.g. batch dimensions, corner cases). Run these tests from the project root directory as follows:

```
python -m pytest tests
```

You can also run the tests from inside VS Code's graphical interface. From the command palette (<key>⌘ command</key> <key>⇧ shift</key> <key>P</key>), run `Testing: Focus on Test Explorer View`.

## Collaboration Policy

You may work with other students and use AI tools (e.g. ChatGPT, GitHub Copilot), but must submit code that you understand fully and have written yourself.

## Submission Policy

Before submitting, ensure that your code has been formatted using Black and linted using Ruff. Otherwise, you will lose points. Submit your work using Canvas.

## Bug Bounty

If you are the first to find and report a bug of any kind in this assignment or README, you will be given extra points. Bugs could include, but are not limited to:

- Code bugs
- Incorrect instructions
- Grammatical errors
- Broken links
- Outdated or broken dependencies

To report a bug, post to the bug bounty thread on Piazza. Make sure your name is visible to the instructors if you want to receive credit for your discovery. Rewards for individual bugs will be proportional to their severity, and total rewards per student can be up to 3% of this assignment's total point value
