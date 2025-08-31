import os
import torch
import torchaudio
import numpy as np
from typing import Dict, Any
from torch.utils.data import Dataset


class FeatureExtractor:
    def __init__(
            self,
            frame_size: int = 256,
            hop_size: int = 256,
            delta_shift: float = 10.0,
            audio_channels: list = [0],
            control_channels: list = [2, 3],
            sample_rate: int = 48000):

        # Initialise Params
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.audio_channels = audio_channels
        self.control_channels = control_channels
        self.shift = int(delta_shift / 1000 * sample_rate)
        self.sample_rate = sample_rate

    def __call__(self, waveform):
        """Transform raw waveform signal into frames of inputs and targets."""

        # Calculate delta features
        rpm_data = self._calculate_deltas(waveform[self.control_channels[0]], shift=self.shift)
        nm_data = self._calculate_deltas(waveform[self.control_channels[1]], shift=self.shift)

        # Obtain Features
        rpm_features = self._prepare_sequence(rpm_data)
        nm_features = self._prepare_sequence(nm_data)

        # Prepare Inputs | Targets
        inputs = torch.cat((rpm_features, nm_features), dim=-1)
        control = waveform[self.control_channels]
        target = waveform[self.audio_channels]

        return inputs, control, target

    def _prepare_sequence(self, waveform):
        """Prepares features sequence of frames. """

        # Get Frames
        frames = waveform.unfold(dimension=-1, size=self.frame_size, step=self.hop_size)

        # Prepare Frame Features
        frame_means = torch.mean(frames, dim=-1)

        return frame_means.permute(1, 0)

    @staticmethod
    def _calculate_deltas(waveform, shift: int = 128, derivative: bool = False):
        """Prepares Features of delta 01, delta 02"""

        # pad size for delta calculation
        pad_samps = shift * 2

        # pad waveform by reflection
        waveform = torch.nn.functional.pad(
            waveform.unsqueeze(0),
            (pad_samps, pad_samps),
            mode='reflect'
        ).squeeze(0)

        # first order delta
        delta_01 = waveform - torch.roll(waveform, shift)
        delta_01 = delta_01 / shift if derivative else delta_01

        # second order delta
        delta_02 = delta_01 - torch.roll(delta_01, shift)
        delta_02 = delta_02 / shift if derivative else delta_02

        # Stack inputs, skip padded samples
        delta_waveforms = torch.stack((
            waveform[pad_samps:-pad_samps],
            delta_01[pad_samps:-pad_samps],
            delta_02[pad_samps:-pad_samps]
        ), dim=0)

        return delta_waveforms


class FeatureScaler:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def scale(self, x):
        if self.mean is None or self.std is None:
            return self.fit_transform(x)
        else:
            return self.transform(x)

    def fit(self, x):
        self.mean = x.mean(dim=(0, 1), keepdim=True)
        self.std = x.std(dim=(0, 1), keepdim=True) + 1e-8
        return self

    def transform(self, x):
        return (x - self.mean) / self.std

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    @staticmethod
    def normalize(x):
        return x / x.abs().max()


class AudioDataset(Dataset):
    def __init__(
            self,
            audio_path: str,
            chunk_size: int = 65536,
            chunk_overlap: float = 0.5,
            control_channels: list = [2, 3],
            audio_channels: tuple = [0],
            sample_rate: int = 48000,
            split: str = 'train',
            train_ratio: float = 0.9,
            extractor: object = FeatureExtractor):
        super().__init__()

        self.split = split
        self.train_ratio = train_ratio
        self.chunk_size = chunk_size
        self.chunk_hop = int((1 - chunk_overlap) * self.chunk_size)
        self.control_channels = control_channels
        self.audio_channels = audio_channels
        self.extractor = extractor
        self.sample_rate = sample_rate

        # Initialise Dataset
        self.audio_files = [
            os.path.join(audio_path, f)
            for f in os.listdir(audio_path)
            if f.endswith('.wav')
        ]

        # Split train, val
        self.audio_files = self._get_split()

        # Obtain frames with features
        self.inputs, self.controls, self.targets = self._prepare_data()

    def _get_split(self):
        np.random.seed(42)
        shuffled_files = self.audio_files.copy()
        np.random.shuffle(shuffled_files)

        split_idx = int(self.train_ratio * len(shuffled_files))

        if self.split == 'train':
            return shuffled_files[:split_idx]
        elif self.split == 'val':
            return shuffled_files[split_idx:]
        else:
            return self.audio_files

    def _prepare_data(self):

        # Initialise inputs | targets
        self.raw_inputs = []
        self.raw_targets = []
        inputs = []
        controls = []
        targets = []

        for audio_path in self.audio_files:

            # Load and resample audio data
            raw_waveform, source_sr = torchaudio.load(audio_path)
            waveform = torchaudio.functional.resample(raw_waveform, source_sr, self.sample_rate)

            # Collect raw waveforms
            self.raw_inputs.append(raw_waveform[self.control_channels].transpose(1, 0))
            self.raw_targets.append(raw_waveform[self.audio_channels])

            # Slice into audio chunks and feature sequences
            for i in np.arange(0, waveform.shape[-1] - self.chunk_size, self.chunk_hop):

                # Obtain waveform chunk
                chunk = waveform[:, i: i + self.chunk_size]

                # Transform features to sequences
                feature_frames, control_chunk, audio_chunk = self.extractor(chunk)

                # Collect sequences
                inputs.append(feature_frames)
                controls.append(control_chunk)
                targets.append(audio_chunk)

        return torch.stack(inputs), torch.stack(controls), torch.stack(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.controls[idx], self.targets[idx]


class DataLoader:
    def __init__(self, dataset, batch_size=16, shuffle=True, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.indices = np.arange(len(self.dataset))
        self.reset()

    def __len__(self):
        """Number of Batches in Dataset."""
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __iter__(self):
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        stop_marker = len(self.dataset) if not self.drop_last else len(self.dataset) - self.batch_size
        if self.current_index >= stop_marker:
            raise StopIteration

        # Obtain Batch of Sequences
        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        batch = [self.dataset[i] for i in batch_indices]

        # Derive inputs | targets
        batch_inputs = torch.stack([item[0] for item in batch])
        batch_controls = torch.stack([item[1] for item in batch])
        batch_targets = torch.stack([item[2] for item in batch])

        # Advance index
        self.current_index += self.batch_size
        return batch_inputs, batch_controls, batch_targets

    def get_raw_data(self, idx):
        return self.dataset.raw_inputs[idx], self.dataset.raw_targets[idx]

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def create_extractor(config: Dict[str, Any]) -> FeatureExtractor:
    """Initialize Feature Extractor from config."""
    dataset_cfg = config['dataset']
    return FeatureExtractor(
        frame_size=dataset_cfg["frame_size"],
        hop_size=dataset_cfg["hop_size"],
        delta_shift=dataset_cfg["delta_shift"],
        audio_channels=dataset_cfg["audio_channels"],
        control_channels=dataset_cfg["control_channels"],
        sample_rate=dataset_cfg["sample_rate"]
    )


def create_scaler(config: Dict[str, Any]) -> FeatureScaler:
    """Initialize Feautre Scaler from config."""
    dataset_cfg = config['dataset']
    mean = dataset_cfg.get('mean', None)
    std = dataset_cfg.get('std', None)
    mean = torch.tensor(mean).view(1, 1, len(mean)) if mean else None
    std = torch.tensor(std).view(1, 1, len(std)) if std else None
    return FeatureScaler(mean, std)


def create_dataset(config: Dict[str, Any], extractor: FeatureExtractor, split: str = 'train') -> AudioDataset:
    """Initialize Dataset from config."""
    dataset_cfg = config['dataset']
    return AudioDataset(
        audio_path=dataset_cfg["dataset_path"],
        control_channels=dataset_cfg["control_channels"],
        audio_channels=dataset_cfg["audio_channels"],
        chunk_size=dataset_cfg["chunk_size"],
        chunk_overlap=dataset_cfg["chunk_overlap"],
        sample_rate=dataset_cfg["sample_rate"],
        extractor=extractor,
        split=split
    )


def create_dataloader(config: Dict[str, Any], split: str = 'train') -> DataLoader:
    """Initialize DataLoader from config."""
    extractor = create_extractor(config)
    scaler = create_scaler(config)
    dataset = create_dataset(config, extractor, split)

    if split == 'train':
        dataset.inputs = scaler.fit_transform(dataset.inputs)

        # Store dataset mean, std for inference
        config['dataset']['mean'] = scaler.mean[-1].tolist()[0]
        config['dataset']['std'] = scaler.std[-1].tolist()[0]
    else:
        # Scaler gets mean, var from config
        dataset.inputs = scaler.transform(dataset.inputs)

    trainer_cfg = config['trainer']
    return DataLoader(
        dataset=dataset,
        batch_size=trainer_cfg["batch_size"],
        shuffle=trainer_cfg["shuffle"],
        drop_last=trainer_cfg["drop_last"]
    )
