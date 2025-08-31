import os
from typing import Dict, Any
from datetime import datetime
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset"))
DEFAULT_SET = 'C_full_set'

"""
Configurations for Training and Synthesis Pipeline.
"""

DATASET_CONFIG = dict(
    dataset_path=os.path.join(DATA_DIR, DEFAULT_SET),
    control_channels=[-2, -1],
    audio_channels=[0],
    delta_features=3,
    delta_shift=50,
    rpm_range=(0, 10000),
    chunk_size=65536,
    chunk_overlap=0.5,
    frame_size=128,
    hop_size=128,
    sample_rate=16000
)

SYNTH_CONFIG = dict(
    num_harmonics=100,
    num_noisebands=256,
)

MODEL_CONFIG = dict(
    model_type='ptr',  # ptr | hpn
    hidden_size=256,
    gru_size=512,
    gru_layers=1
)

TRAINER_CONFIG = dict(
    # trainer
    epochs=100,
    batch_size=8,
    shuffle=True,
    drop_last=True,

    # optimizer
    learning_rate=1e-3,
    weight_decay=5e-3,
    betas=(0.9, 0.999),

    # scheduler
    max_lr=2e-4,
    pct_start=0.075,
    div_factor=10.0,
    final_div_factor=100.0,
    anneal_strategy='cos'
)

LOSS_CONFIG = dict(
    w_harm=1.0,
    w_stft=1.0,
    harm_loss_params=dict(
        fft_size=65536,
        win_size=16384,
        hop_size=256,
        silhouette='clip',
        difference='energy'
    ),
    stft_loss_params=dict(
        fft_sizes=[
            32768,
            16384,
            8192,
            4096,
            2048,
            1024,
            512,
            256,
            128,
            64,
            32
        ],
        overlap=0.75,
        window='hann_window',
        w_sc=1.0,
        w_log_mag=1.0,
        w_lin_mag=1.0,
        w_energy=1.0,
        scale_invariance=True,
        equal_contribution=True,
    )
)

LOGGER_CONFIG = dict(
    checkpoints_dir=os.path.join(BASE_DIR, 'checkpoints'),
    checkpoint_interval=2000,
)


def get_config() -> Dict[str, Dict[str, Any]]:
    """Get structured configuration with all components."""
    return {
        'dataset': DATASET_CONFIG,
        'model': MODEL_CONFIG,
        'synth': SYNTH_CONFIG,
        'trainer': TRAINER_CONFIG,
        'loss': LOSS_CONFIG,
        'logger': LOGGER_CONFIG,
        'metadata': {
            'experiment_name': f"experiment_{DEFAULT_SET}",
            'dataset_used': DEFAULT_SET,
            'created_at': datetime.now().isoformat(),
            'base_dir': BASE_DIR,
            'data_dir': DATA_DIR
        }
    }


def update_config(config: Dict[str, Any], args) -> Dict[str, Any]:
    """Update config with CLI arguments."""

    # Model parameters
    config['model']['model_type'] = args.model_type
    config['model']['hidden_size'] = args.hidden_size
    config['model']['gru_size'] = args.gru_size

    # Synthesis parameters
    config['synth']['num_harmonics'] = args.num_harmonics
    config['synth']['num_noisebands'] = args.num_noisebands

    # Dataset
    config['dataset']['dataset_path'] = os.path.join(DATA_DIR, args.dataset)
    config['metadata']['dataset_used'] = args.dataset
    config['metadata']['experiment_name'] = f"experiment_{args.dataset}"

    return config


def load_config(path: str) -> Dict[str, Any]:
    """Load JSON config from a file path."""
    with open(path, 'r') as f:
        config_data = json.load(f)
    return config_data
