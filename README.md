# Procedural Engines Model (PRCE)

**Physics-Informed Neural Engine Sound Synthesis**

This repository contains the complete implementation and trained models for the Procedural Engines Model (PRCE), a novel deep learning architecture for engine sound synthesis that integrates physics-informed inductive biases within differentiable synthesis pipelines.

## Overview

Engine sound synthesis presents unique challenges that distinguish it from musical audio paradigms. Unlike sustained musical tones, engine sounds emerge from sequential combustion events creating acoustic phenomena with significant inharmonicity, extremely low fundamental frequencies (down to 5 Hz), and rapid temporal sequences with intervals under 15 milliseconds.

The PRCE framework addresses these challenges through two complementary synthesis configurations:

- **Harmonic-Plus-Noise (HPN)**: Modified harmonic synthesis with systematic inharmonicity and temporal-spectral structuring
- **Pulse-Train-Resonator (PTR)**: Direct modeling of combustion pulses aligned to engine firing patterns with differentiable resonator networks

## Key Features

- **Physics-informed architecture** incorporating domain-specific acoustic principles
- **Dual synthesis strategies** providing complementary optimization pathways
- **Time-varying embeddings** of RPM, torque, throttle, and DFCO parameters
- **Custom loss function** prioritizing spectral energy near engine-order harmonics
- **Campbell diagram-inspired** training objectives from NVH analysis
- **Comprehensive evaluation** on 2.5 hours of procedurally generated engine sounds

## Repository Structure

```
prce-model/
├── checkpoints/           # Pre-trained model weights
│   └── 2025-08-31_models_and_weights.zip
├── configs/              # Base configuration files
├── scripts/              # Training and inference scripts
│   ├── train.py         # Training pipeline with CLI
│   └── inference.py     # Model inference with CLI
├── src/                 # Source code
│   ├── audio/           # Audio processing utilities
│   ├── data/            # Data loading and processing
│   ├── models/          # Model implementations
│   │   ├── hpn_model.py    # Harmonic-Plus-Noise variant
│   │   ├── hpn_synth.py    # HPN synthesis modules
│   │   ├── ptr_model.py    # Pulse-Train-Resonator variant
│   │   ├── ptr_synth.py    # PTR synthesis modules
│   │   └── model.py        # Base model architecture
│   ├── training/        # Training utilities
│   └── utils/           # General utilities
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── LICENSE             # License file

```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rdoerfler/prce-model.git
cd prce-model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Extract pre-trained models:
```bash
cd checkpoints
unzip 2025-08-31_models_and_weights.zip
```

## Usage

### Training

Train the HPN variant:
```bash
python scripts/train.py --model_type hpn --dataset C_full_set
```

Train the PTR variant:
```bash
python scripts/train.py --model_type ptr --dataset C_full_set
```

Customize model architecture:
```bash
python scripts/train.py --model_type ptr --num_harmonics 150 --hidden_size 512 --gru_size 1024
```

### Inference

Generate engine sounds using trained models:
```bash
python scripts/inference.py --model_type hpn --dataset C_full_set
```

## Command Line Interface

The training and inference scripts use a unified CLI with the following key parameters:

- `--model_type`: Choose between `hpn` (Harmonic-Plus-Noise) or `ptr` (Pulse-Train-Resonator)
- `--dataset`: Specify dataset name (default: `C_full_set`)
- `--num_harmonics`: Number of harmonics (default: 100)
- `--num_noisebands`: Number of noise bands (default: 256) 
- `--hidden_size`: Hidden layer size (default: 256)
- `--gru_size`: GRU layer size (default: 512)

Configuration is managed through a base config system combined with CLI parameter overrides.

## Model Variants
- Modified harmonic synthesis with systematic inharmonicity
- Temporal-spectral structuring of noise components
- Robust to harmonic irregularities
- Greater flexibility across diverse engine configurations

### Pulse-Train-Resonator (PTR)
- Direct modeling of combustion pulse sequences
- Differentiable resonator networks for exhaust acoustics
- Superior validation performance (5.7% improvement in total loss)
- More consistent training-validation transfer

## Dataset

This work utilizes the **Procedural Engine Sounds Dataset**, a comprehensive collection of procedurally generated engine audio with time-aligned control annotations. The dataset includes:

- 2.5 hours of engine audio across varied operating conditions
- Time-aligned RPM, torque, throttle, and DFCO annotations
- Multiple engine configurations and acoustic scenarios
- Systematic coverage of engine operating parameters

**Dataset Availability:**
- **Zenodo**: https://doi.org/10.5281/zenodo.16883336
- **Hugging Face Datasets**: https://huggingface.co/datasets/rdoerfler/procedural-engine-sounds

## Results

Evaluation reveals complementary strengths between synthesis approaches:

- **PTR**: 5.7% superior validation performance, consistent training-validation transfer
- **HPN**: Greater flexibility across engine configurations, robust to harmonic irregularities
- Both variants successfully capture authentic engine acoustic behaviors with distinct signatures

## Citation

If you use this code or the Procedural Engines Dataset in your research, please cite:

```bibtex
@mastersthesis{doerfler2025,
  title={Neural Engine Sound Synthesis with Physics-Informed Inductive Biases and Differentiable Signal Processing},
  author={Robin Doerfler},
  year={2025},
  month={August},
  school={Universitat Pompeu Fabra},
  type={Master's thesis in Sound and Music Computing},
  note={Supplementary code and models available at: https://github.com/rdoerfler/prce-model}
}
```

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). See the `LICENSE` file for details.

**You are free to:**
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material

**Under the following terms:**
- Attribution — You must give appropriate credit and indicate if changes were made
- NonCommercial — You may not use the material for commercial purposes

For commercial use, please contact the author.

## Additional Resources

- **Audio Examples**: Supplementary audio examples demonstrating model outputs are available at: https://rdoerfler.github.io/prce-examples/
- **Dataset**: Procedural Engine Sounds Dataset on [Zenodo](https://doi.org/10.5281/zenodo.16883336) and [Hugging Face](https://huggingface.co/datasets/rdoerfler/procedural-engine-sounds)

## Acknowledgments

This research demonstrates systematic integration of physics-informed inductive biases into differentiable synthesis architectures, providing a methodological framework applicable to physically-constrained audio generation beyond automotive contexts.

## Contact

For questions or collaboration opportunities, please contact doerflerrobin@gmail.com or open an issue on this repository.

---

**Keywords**: Engine Sound Synthesis, Differentiable Signal Processing, Physics-Informed Neural Networks, Inductive Biases, Neural Audio Synthesis