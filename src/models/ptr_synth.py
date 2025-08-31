import torch
import torch.nn as nn
import torch.nn.functional
from src.audio.noisebank import NoiseBank
from torchlpc import sample_wise_lpc

EPS = 1e-5


class PulseTrainSynth(nn.Module):
    """Differentiable Prodcedural Engine Synthesizer"""

    def __init__(
            self,
            num_harmonics: int = 64,
            num_noisebands: int = 32,
            num_cylinders: int = 8,
            chunk_size: int = 65536,
            sample_rate: int = 48000):
        super().__init__()

        # Hyperparameters
        self.num_harmonics = num_harmonics
        self.num_cylinders = num_cylinders
        self.num_noisebands = num_noisebands
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate

        # Initialise NoiseBank
        self.noise_bank = NoiseBank(
            num_noisebands=num_noisebands,
            chunk_size=self.chunk_size,
            sample_rate=sample_rate)

        # Resonators
        self.resonator_bank_a = KarplusStrongResonator(
            min_delay=40,
            max_delay=125,
            temperature=0.0,
            low_pass_bias=False
        )
        self.resonator_bank_b = KarplusStrongResonator(
            min_delay=40,
            max_delay=125,
            temperature=0.0,
            low_pass_bias=False
        )
        self.resonator_c = KarplusStrongResonator(
            min_delay=32,
            max_delay=40,
            temperature=0.0,
            low_pass_bias=True
        )

        # Harmonics per Cylinder
        self.register_buffer('engine_harmonics', torch.arange(1, self.num_harmonics + 1))

    def forward(
            self,
            f0: torch.Tensor,
            harmonic_amplitudes: torch.Tensor,
            harmonic_fm: torch.Tensor,
            cylinder_phase_offsets: torch.Tensor,
            cylinder_pulse_shapes: torch.Tensor,
            noisebank_amplitudes: torch.Tensor,
            noise_pulse_shapes: torch.Tensor,
            noise_pulse_gain: torch.Tensor,
            flow_noise_gain: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pressure wave synthesis

        Firing Order:
        - 1-8-7-2-6-5-4-3 | [0, 7, 6, 1, 5, 4, 3, 2]
        or
        - 1-5-4-8-6-3-7-2 | [0, 4, 3, 7, 5, 2, 6, 1]
        with
            - bank 1 = cylinders 1, 2, 3, 4
            - bank 2 = cylinders 5, 6, 7, 8
        """

        B, C, T = cylinder_phase_offsets.shape

        # Phase Sequence over Power Cycle
        firing_order = torch.tensor([0, 4, 3, 7, 5, 2, 6, 1], device=f0.device)[:self.num_cylinders]
        firing_angles = firing_order / self.num_cylinders * 2 * torch.pi

        # Cylinder phases (4 Stroke mapped to 2pi)
        power_cycle_phase = (2 * torch.pi * f0 / self.sample_rate).cumsum_(dim=-1) * 0.5
        cylinder_phases = (power_cycle_phase + firing_angles.view(1, self.num_cylinders, 1)) % (2 * torch.pi)

        # Apply Frequency Modulation
        harmonic_phases = cylinder_phases / (2 * torch.pi)
        harmonic_phases = harmonic_phases ** harmonic_fm
        harmonic_phases = harmonic_phases * 2 * torch.pi
        harmonic_phases = harmonic_phases.unsqueeze(2) * self.engine_harmonics.view(1, 1, self.num_harmonics, 1)
        harmonic_oscillations = -torch.sin(harmonic_phases)

        # Undo Reshaping: -> B, cylinders, orders, T
        harmonic_amplitudes = harmonic_amplitudes.view(B, self.num_cylinders, self.num_harmonics, T)

        # Apply Amplitudes
        harmonics = harmonic_oscillations * harmonic_amplitudes

        # Sum all Harmonic Oscillations to Piston Pressure Pulse
        harmonics = torch.sum(harmonics, dim=2)

        # Amplitude Modulate Harmonics
        cyl_alpha = cylinder_pulse_shapes[:, :self.num_cylinders, :]
        cyl_beta = cylinder_pulse_shapes[:, self.num_cylinders:, :]
        cyl_env = (1 - torch.exp(-cyl_alpha * cylinder_phases)) * torch.exp(-cyl_beta * cylinder_phases)
        harmonics = harmonics * cyl_env * 10

        # Noise Pulses
        noise = self.noise_bank(noisebank_amplitudes)
        noise_alpha = noise_pulse_shapes[:, 0:1, :]
        noise_beta = noise_pulse_shapes[:, 1:2, :]
        noise_phase = power_cycle_phase % (2 * torch.pi)   # only power cycle (0.5 order) modulation
        noise_env = (1 - torch.exp(-noise_alpha * noise_phase)) * torch.exp(-noise_beta * noise_phase)
        noise_bursts = noise * noise_env * noise_pulse_gain

        # Flow Noise (Deceleration Fuel Cutoff)
        flow_noise = noise * flow_noise_gain * 0.3

        # Jitter, Turbulences
        turbulence = harmonics * noise * (noise_pulse_gain + flow_noise_gain) * 0.7

        # Split Cylinder Banks
        cylinders = harmonics + noise_bursts + flow_noise + turbulence
        bank_a = torch.sum(cylinders[:, :4, :], dim=1, keepdim=True)
        bank_b = torch.sum(cylinders[:, 4:, :], dim=1, keepdim=True)

        # Resonators
        resonance_a = self.resonator_bank_a(bank_a)
        resonance_b = self.resonator_bank_b(bank_b)
        resonance_c = self.resonator_c(resonance_a + resonance_b)

        # Sum
        output = torch.sum(resonance_c, dim=1, keepdim=True)

        return output

    @property
    def device(self):
        return self.buffer_size.device


class KarplusStrongResonator(nn.Module):
    """Specialized head for resonator bank parameters"""

    def __init__(
            self,
            min_delay: int = 60,
            max_delay: int = 420,
            temperature: float = 0.0,
            low_pass_bias: bool = False):
        super().__init__()
        self.max_delay = max_delay
        self.min_delay = min_delay
        self.tau = temperature
        self.lowpass = low_pass_bias

        # Setup Learnable Parameters
        self.delay_param = nn.Parameter(torch.randn(max_delay - min_delay))
        self.feedback_gain = nn.Parameter(torch.ones(1))
        self.reflection_coeffs = nn.Parameter(torch.randn(2))

    def forward(self, excitation, temperature=1.0, hard=True):
        B, C, T = excitation.shape

        # Gumbel-Softmax for delay time selection
        delay_selection = torch.nn.functional.gumbel_softmax(
            self.delay_param,
            tau=temperature,
            hard=hard,
        )

        # Format coefficients
        coefficients = self._format_coefficients(
            delay_selection,
            self.feedback_gain,
            self.reflection_coeffs
        )

        # Zero pad to add minimum delay
        coefficients = torch.cat((
            torch.zeros(self.min_delay, device=excitation.device),
            coefficients
        ), dim=-1)

        # Expand to B, T
        all_pole_coeffs = coefficients.view(1, 1, self.max_delay).expand(B, T, self.max_delay)

        resonance = self._apply_resonator(excitation, all_pole_coeffs)

        return resonance

    def _format_coefficients(self, one_hot, fb_gain, reflection_coeffs):
        """Convert one-hot selection to adjacent weights with guaranteed stability"""
        shifted = torch.roll(one_hot, shifts=1, dims=-1)

        if self.lowpass:
            reflection_coeffs = torch.sigmoid(reflection_coeffs)
        else:
            reflection_coeffs = torch.tanh(reflection_coeffs)

        # Constrain reflection coefficients to (-1, 1)
        k1 = self.resonant_activation(reflection_coeffs[..., 0], min_magnitude=self.tau)
        k2 = self.resonant_activation(reflection_coeffs[..., 1], min_magnitude=self.tau)

        # Convert to direct form
        a1 = k1 * (1 - k2)

        # Ensure |a2| < 0.99 for numerical stability
        a2 = torch.clamp(k2, -0.999, 0.999)

        # Ensure stability triangle constraints
        a1_bound = 0.999 - torch.abs(a2)
        a1 = torch.clamp(a1, -a1_bound, a1_bound)

        fb_coeffs_stable = torch.stack([a1, a2], dim=-1)

        # Apply feedback gain
        fb_gain = torch.sigmoid(fb_gain) ** 0.45
        fb_coeffs_stable = fb_coeffs_stable * fb_gain

        # Set coefficients
        one_hot = one_hot * fb_coeffs_stable[..., 0].unsqueeze(-1)  # <- a1
        shifted = shifted * fb_coeffs_stable[..., 1].unsqueeze(-1)  # <- a2

        return one_hot + shifted

    @staticmethod
    def resonant_activation(x, min_magnitude=0.5):
        """Push values away from zero while preserving sign"""
        sign = torch.sign(x)
        magnitude = torch.abs(torch.tanh(x))
        boosted_magnitude = min_magnitude + (1 - min_magnitude) * magnitude
        return sign * boosted_magnitude

    @staticmethod
    def lowpass_activation(x, bias_strength=0.5, stability_margin=0.1):
        """
        Bias reflection coefficients towards positive values for low-pass behavior
        while maintaining stability

        Args:
            x: input logits
            bias_strength: how much to bias towards positive values
            stability_margin: safety margin from stability boundary
        """
        tanh_x = torch.tanh(x)

        # Apply positive bias
        biased = tanh_x + bias_strength * (1 - torch.abs(tanh_x))

        # Ensure stability bounds: |k| < (1 - stability_margin)
        max_val = 1 - stability_margin
        return torch.clamp(biased, -max_val, max_val)

    @staticmethod
    def _apply_resonator(excitation, resonator_coefficients):
        """All pole filter as resonator."""

        resonance = sample_wise_lpc(
            excitation.squeeze(1),
            resonator_coefficients
        ).unsqueeze(1)

        return resonance
