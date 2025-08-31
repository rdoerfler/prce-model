import torch
import torch.nn as nn
import torch.nn.functional
from src.audio.noisebank import NoiseBank


class HarmonicNoiseSynth(nn.Module):
    """Procedural Engine Sound Synthesizer (Harmonic-Plus-Noise based)."""
    def __init__(
            self,
            num_harmonics: int = 128,
            num_noisebands: int = 32,
            num_modulators: int = 4,
            batch_size: int = 16,
            chunk_size: int = 65536,
            sample_rate: int = 48000):
        super().__init__()

        # Hyperparameters
        self.num_harmonics = num_harmonics
        self.num_noisebands = num_noisebands
        self.num_modulators = num_modulators
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate

        # Initialise NoiseBank
        self.noise_bank = NoiseBank(
            num_noisebands=num_noisebands,
            chunk_size=self.chunk_size,
            sample_rate=sample_rate)

        self.oscillator = ComplexOscillator(
            sample_rate=self.sample_rate,
        )

    def forward(
            self,
            harmonic_frequencies: torch.Tensor,
            harmonic_amplitudes: torch.Tensor,
            noisebank_amplitudes: torch.Tensor,
            noisebank_mod_exponents: torch.Tensor,
            noisebank_mod_weights: torch.Tensor,
            pulse_noise_gain: torch.Tensor,
            flow_noise_gain: torch.Tensor,
    ) -> torch.Tensor:

        """Synthesize Predicted Parameters."""

        # Synthesize Harmonics
        harmonics = self.oscillator(harmonic_frequencies)
        harmonic_component = torch.sum(harmonic_amplitudes * harmonics, dim=1, keepdim=True)

        # Synthesize Modulated Noisebank
        noise = self.noise_bank(noisebank_amplitudes)
        noise_bursts = self._synthesize_noise_bursts(
            noise=noise,
            mod_signal=harmonics[:, :self.num_modulators],
            mod_exponents=noisebank_mod_exponents,
            mod_weights=noisebank_mod_weights,
            mod_gain=pulse_noise_gain
        )

        # Construct Turbulence Distortion Signal
        turbulence_gain = (pulse_noise_gain + flow_noise_gain) * 0.7
        turbulence = self._turbulence_distortion(harmonic_component, noise, turbulence_gain)

        # Flow Noise (Deceleration Fuel Cutoff)
        flow_noise = noise * flow_noise_gain * 0.3

        # Sum noise components
        noise_components = noise_bursts + turbulence + flow_noise

        return harmonic_component + noise_components

    @property
    def device(self):
        return self.buffer_size.device

    @staticmethod
    def _synthesize_noise_bursts(
            noise: torch.Tensor,
            mod_signal: torch.Tensor,
            mod_exponents: torch.Tensor,
            mod_weights: torch.Tensor,
            mod_gain: torch.Tensor) -> torch.Tensor:
        """
        Modulate noise to pulses linked to rpm.
        """

        # Optional: Convert to triangle wave
        mod_signal = torch.arcsin(mod_signal * 0.99) * 2 / torch.pi

        # Control pulse width / sharpness
        mod_shaped = mod_signal.abs().pow(mod_exponents)

        # Weighted sum as modulator signal
        mod_sum = torch.sum(mod_shaped * mod_weights, dim=1, keepdim=True)

        # Apply modulatoin and gain
        output = mod_sum * noise * mod_gain

        return output

    @staticmethod
    def _turbulence_distortion(harmonics, noise: object = None, depth: float = 0.03):
        return harmonics * noise * depth


class ComplexOscillator(nn.Module):
    def __init__(self, sample_rate: float):
        """
        Time-varying complex oscillator that maintains gradient flow for frequency estimation.

        This version is specifically designed for cases where you predict frequencies and
        amplitudes separately and want to maintain the complex surrogate benefits.
        """
        super().__init__()
        self.sample_rate = sample_rate

    def _create_antialias_mask(self, frequencies):
        mask = frequencies < (self.sample_rate / 2)
        return frequencies * mask

    @staticmethod
    def _random_phase_shift(frequencies):
        shape = frequencies.shape[:-1] + (1,)
        shift = torch.rand(shape, device=frequencies.device, dtype=frequencies.dtype) * 2 * torch.pi
        return shift

    def forward(self, frequencies, initial_phase=None):
        """
        Generate time-varying oscillator output.

        Arguments:
            frequencies (torch.Tensor): Predicted frequencies in Hz, shape [..., time_steps]
            initial_phase (torch.Tensor, optional): Initial phase, shape [...] or [..., 1]

        Returns:
            torch.Tensor: Real-valued oscillator output
        """

        # Mute above nyquist
        frequencies = self._create_antialias_mask(frequencies)

        # Convert to angular frequency
        omega = 2 * torch.pi * frequencies / self.sample_rate

        # Integrate frequency to get phase
        phase = torch.cumsum(omega, dim=-1)

        # Add random or specified initial phase
        if initial_phase is None:
            batch_shape = frequencies.shape[:-1]
            initial_phase = torch.rand(*batch_shape, 1, device=frequencies.device) * 2 * torch.pi

        if initial_phase.dim() < phase.dim():
            initial_phase = initial_phase.unsqueeze(-1)

        phase = phase + initial_phase

        # Create complex representation for gradient flow
        z = torch.exp(1j * phase)

        return z.real


class SinOscillator(nn.Module):
    def __init__(self, sample_rate, length: int = 65536):
        """Sinusoidal oscillator with proper phase accumulation.

        Arguments:
            sample_rate (float): Sample rate in Hz
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.register_buffer('n', torch.arange(length, dtype=torch.float32).unsqueeze(0))

    @staticmethod
    def _random_phase_shift(frequencies):
        shape = frequencies.shape[:-1] + (1,)
        shift = torch.rand(shape, device=frequencies.device, dtype=frequencies.dtype) * 2 * torch.pi
        return shift

    def _create_antialias_mask(self, frequencies):
        mask = frequencies < (self.sample_rate / 2)
        return frequencies * mask

    def forward(self, frequencies, amplitudes):
        """
        Generate audio with proper phase accumulation

        Args:
            frequencies (Tensor): Instantaneous frequencies at each time step
                [batch_size, num_voices, sequence_length]
            amplitudes (Tensor): Instantaneous amplitudes of same shape as frequencies
        Returns:
            torch.Tensor: Generated audio [batch_size, num_voices, sequence_length]
        """
        frequencies = self._create_antialias_mask(frequencies)
        phase_increments = 2 * torch.pi * frequencies / self.sample_rate
        phases = phase_increments.cumsum(dim=-1)
        phases = phases + self._random_phase_shift(frequencies)
        return torch.sin(phases) * amplitudes
