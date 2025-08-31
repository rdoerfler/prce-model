import torch
from typing import Dict, Any, Union

from .hpn_model import HarmonicNoiseModel
from .hpn_synth import HarmonicNoiseSynth
from .ptr_model import PulseTrainModel
from .ptr_synth import PulseTrainSynth

DDSPSynth = Union[PulseTrainSynth, HarmonicNoiseSynth]
DDSPModel = Union[PulseTrainModel, HarmonicNoiseModel]

MODEL_REGISTRY = {
    'hpn': HarmonicNoiseModel,
    'ptr': PulseTrainModel
}

SYNTH_REGISTRY = {
    'hpn': HarmonicNoiseSynth,
    'ptr': PulseTrainSynth
}


def create_model(
        config: Dict[str, Any],
        device: torch.device) -> DDSPModel:
    """
    Initialize Model from config.
    """

    synth_cfg = config['synth']
    dataset_cfg = config['dataset']
    model_cfg = config['model']

    synthesizer = SYNTH_REGISTRY[model_cfg['model_type']](
        num_harmonics=synth_cfg["num_harmonics"],
        num_noisebands=synth_cfg["num_noisebands"],
        chunk_size=dataset_cfg["chunk_size"],
        sample_rate=dataset_cfg["sample_rate"],
    ).to(device)

    return MODEL_REGISTRY[model_cfg['model_type']](
        synthesizer=synthesizer,
        num_harmonics=synth_cfg["num_harmonics"],
        num_noisebands=synth_cfg["num_noisebands"],
        feature_size=dataset_cfg["delta_features"],
        rpm_range=dataset_cfg["rpm_range"],
        hidden_size=model_cfg["hidden_size"],
        gru_size=model_cfg["gru_size"],
        gru_layers=model_cfg["gru_layers"]
    ).to(device)
