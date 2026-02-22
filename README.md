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
├── main.py                     # Entry point with three simulation experiments
├── environment.yml             # Conda environment specification
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

`main.py` provides three runnable experiments:

| Function | Description |
|---|---|
| `run_morph_test()` | Morphs a phantom from sphere to cube over several steps and tracks topological changes in H₁. |
| `run_motion_analysis()` | Translates a sphere phantom along a linear trajectory and measures inter-frame topological distances. |
| `run_gating_simulation()` | Simulates periodic respiratory motion and produces persistence images suitable for gating or ML classification. |

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

By default `run_morph_test()` is executed. Edit the `if __name__ == "__main__"` block in `main.py` to switch experiments.

## Key Concepts

- **Plücker coordinates** — A 6D representation `[d, m]` where `d` is the unit direction and `m = p × d` is the moment vector, uniquely identifying an oriented line in 3D.
- **Hybrid weighted distance** — Combines an angular component (angle between direction vectors) and a geometric component (shortest distance between skew/parallel lines), each with configurable weights `α` and `β`.
- **Persistence diagrams** — Encode the birth and death of topological features (connected components H₀, loops H₁) across filtration scales.
- **Wasserstein / Bottleneck distance** — Standard metrics for comparing persistence diagrams; Wasserstein measures total transport cost while bottleneck captures the single largest difference.

## License

This project does not currently include a license file. Contact the repository owner for usage terms.

## Authors
- **Filip Surname** — [GitHub profile](https://github.com/fstenl)
- **Matilda Skogman** - [GitHub profile](https://github.com/matildaskogman)
- **Prarthana Duraisamy** - [GitHub profile](https://github.com/medtech-code)
