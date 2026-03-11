# PET-TDA

Topological Data Analysis (TDA) framework for Positron Emission Tomography (PET) imaging. This project applies persistent homology to Lines of Response (LORs) sampled from simulated PET scanner data, enabling topological comparison of phantom frames across shape morphing, spatial motion, and respiratory gating scenarios.

## Overview

The pipeline works as follows:

1. **Phantom generation** — 3D activity distributions (spheres, cubes, morphed shapes) are created and moved along configurable trajectories.
2. **PET forward projection** — Phantoms are projected through a simulated PET scanner geometry (via `parallelproj`) to produce sinograms, from which LOR endpoints are stochastically sampled.
3. **Line representation** — LOR endpoint pairs are converted to canonical Plücker coordinates, providing a 6D representation of each line in 3D space.
4. **Distance computation** — A custom hybrid metric combining angular and geometric distances between lines produces a pairwise distance matrix.
5. **Persistent homology** — Persistence diagrams are computed from the distance matrices using Vietoris–Rips filtration (`ripser`).
6. **Inter-frame comparison** — Wasserstein or bottleneck distances between persistence diagrams quantify topological differences across frames, visualised as heatmaps.

## Project Structure

```
PET-TDA/
├── main.py                     # Entry point with simulation experiments
├── environment.yml             # Conda environment specification
├── phantoms/
│   ├── primitives.py           # Parameterised geometric phantoms (sphere, box, ellipsoid, morphed, simplex)
│   ├── trajectories.py         # Motion paths (static, linear, circular, periodic/sinusoidal)
│   ├── generator.py            # Frame sequence generation and 2D/3D visualisation
│   └── noise.py                # Noise models (Gaussian, Poisson, salt-and-pepper)
├── scanner/
│   └── scanner.py              # PET scanner geometry, forward projection, LOR sampling
├── lines/
│   ├── representations.py      # Plücker coordinate conversion
│   └── metrics.py              # Pairwise line distance metrics (Euclidean, hybrid weighted)
└── tda/
    ├── diagrams.py             # Persistent homology computation and visualisation
    ├── distances.py            # Bottleneck & Wasserstein distances between diagrams
    └── vectorization.py        # Persistence image generation for ML pipelines
```

## Experiments

`main.py` provides the following runnable experiments:

| Function | Description |
|---|---|
| `linear_motion_test()` | Translates a sphere phantom along a linear trajectory and measures inter-frame topological distances. |
| `deformation_test()` | Morphs a phantom from sphere to cube over several steps and tracks topological changes. |
| `intra_variability_deformation_test()` | Multiple LOR samples per deformation frame to compare intra-frame noise against inter-frame signal. |
| `sinusoidal_motion_test()` | Simulates periodic sinusoidal motion and computes inter-frame distances. |
| `intra_variability_sinusoidal__motion_test()` | Intra-frame variability analysis for the sinusoidal motion scenario. |
| `size_test(phantom_fn)` | Grows then shrinks a phantom (sphere, box, or ellipsoid) and tracks topological changes. |
| `intra_variability_size_test()` | Intra-frame variability analysis for the size variation scenario. |
| `visualize_2d_phantom(shape_type)` | Generates a 2D phantom, projects it, and visualises the Plücker point cloud interactively. |
| `visualize_sphere_sinogram()` | Displays a sphere phantom alongside its forward-projected sinogram. |

## Installation

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/) or [Mamba](https://mamba.readthedocs.io/)

### Setup

```bash
conda env create -f environment.yml
conda activate pet-tda
```

### Dependencies

| Package | Purpose |
|---|---|
| Python 3.11 | Runtime |
| PyTorch | Tensor operations and GPU support |
| NumPy (< 2) | Array operations |
| SciPy | Scientific computing utilities |
| Matplotlib | Visualisation |
| parallelproj | PET scanner geometry and forward/back projection |
| array_api_compat | Array API compatibility layer for PyTorch |
| scikit-tda | TDA toolkit (includes `ripser` and `persim`) |

## Usage

```bash
conda activate pet-tda
python main.py
```

By default `size_test(primitives.create_sphere_phantom)` is executed. Edit the `if __name__ == "__main__"` block in `main.py` to switch experiments.

## Key Concepts

- **Plücker coordinates** — A 6D representation `[d, m]` where `d` is the unit direction and `m = p × d` is the moment vector, uniquely identifying an oriented line in 3D.
- **Hybrid weighted distance** — Combines an angular component (angle between direction vectors) and a geometric component (shortest distance between skew/parallel lines), each with configurable weights `α` and `β`.
- **Persistence diagrams** — Encode the birth and death of topological features (connected components H₀, loops H₁) across filtration scales.
- **Wasserstein / Bottleneck distance** — Standard metrics for comparing persistence diagrams; Wasserstein measures total transport cost while bottleneck captures the single largest difference.

## License

This project does not currently include a license file. Contact the repository owner for usage terms.

## Acknowledgements

This project relies on the following open-source libraries:

- **parallelproj** — Georg Schramm and Kris Thielemans. *Parallelproj—an open-source framework for fast calculation of projections in tomography*. Frontiers in Nuclear Medicine, 2023. DOI: [10.3389/fnume.2023.1324562](https://doi.org/10.3389/fnume.2023.1324562). [github.com/gschramm/parallelproj](https://github.com/gschramm/parallelproj)
- **scikit-tda** — Saul, Nathaniel and Tralie, Chris. *Scikit-TDA: Topological Data Analysis for Python*. [github.com/scikit-tda](https://github.com/scikit-tda), 2019. DOI: [10.5281/zenodo.2533369](https://doi.org/10.5281/zenodo.2533369)

## Authors
- **Filip Stenlund** — [GitHub profile](https://github.com/fstenl)
- **Matilda Skogman** - [GitHub profile](https://github.com/matildaskogman)
- **Prarthana Duraisamy** - [GitHub profile](https://github.com/medtech-code)
