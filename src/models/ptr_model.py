import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

LOG10 = torch.log(torch.tensor(10.0))
eps = 1e-7


class PulseHarmonicsHead(nn.Module):
    """Specialized head for pressure wave synthesis."""

    def __init__(self, hidden_dim, num_cylinders: int = 8, num_harmonics: int = 64):
        super().__init__()

        # Properties
        self.num_cylinders = num_cylinders
        self.num_harmonics = num_harmonics

        # Separate networks for different parameter types
        self.harm_amps = nn.Linear(hidden_dim, num_harmonics)
        self.harmonics_gain = nn.Linear(hidden_dim, 1)
        self.fm_depth = nn.Linear(hidden_dim, num_cylinders)
        self.cyl_amps = nn.Linear(hidden_dim, num_cylinders)
        self.cyl_phase_offsets = nn.Linear(hidden_dim, num_cylinders)
        self.cyl_harm_decays = nn.Linear(hidden_dim, num_cylinders)
        self.cyl_shaping_coeffs = nn.Linear(hidden_dim, 2 * num_cylinders)

        # Constants
        self.cyl_offset_depth = 40 / 720 * 2 * torch.pi
        self.register_buffer('engine_orders', torch.arange(1, num_harmonics + 1) * 0.5)

    def forward(self, x):

        B, T, hidden_size = x.shape

        # Cylinder Processing
        cyl_amps = torch.sigmoid(self.cyl_amps(x))

        # Cylinder firing variation based on operation state
        cyl_phase_offset = torch.tanh(self.cyl_phase_offsets(x)) * self.cyl_offset_depth

        # Add jitter (1% crank)
        cyl_jitter = torch.randn(B, T, self.num_cylinders, device=x.device) * self.cyl_offset_depth / 40
        cyl_phase_offset = cyl_phase_offset + cyl_jitter

        # Amplitude processing
        amps_raw = torch.clamp(self.harm_amps(x), min=-10, max=10)
        amps = torch.softmax(amps_raw, dim=-1)

        # Apply physics-based harmonic decay
        decay_factor = torch.sigmoid(self.cyl_harm_decays(x)) ** 0.75
        cyl_harm_decays = torch.exp(-decay_factor.unsqueeze(-1) * self.engine_orders)
        amps = amps.unsqueeze(-2) * cyl_harm_decays

        # Re-normalize after decay
        amps = amps / torch.sum(amps, dim=-1, keepdim=True).clamp(min=1e-6)

        # Overall gain
        harmonic_gain = 2 * torch.pow(torch.sigmoid(self.harmonics_gain(x)), LOG10) + eps
        harmonic_amps = amps * harmonic_gain.unsqueeze(-1) * cyl_amps.unsqueeze(-1)

        # Temporarily Stack Harmonics per Cylinder for upsampling
        harmonic_amps = harmonic_amps.view(B, T, self.num_cylinders * self.num_harmonics)

        # Frequency Modulation
        # Centered at 0.9, on max depth lowers to 0.8, min depth to 1
        harmonic_fm = 1 - torch.sigmoid(self.fm_depth(x)) * 0.2

        # Pulse Shaping
        cyl_shape_coeffs = torch.sigmoid(self.cyl_shaping_coeffs(x))
        cyl_shape_coeffs = cyl_shape_coeffs ** 0.5  # Bias towards longer decays for softer pulses
        cyl_shape_coeffs = torch.clamp(cyl_shape_coeffs, 0.001, 0.999)

        # Modulate Harmonic Pulses by rounded Exhaust Valve Shape
        cyl_alpha = cyl_shape_coeffs[..., :self.num_cylinders]
        cyl_alpha = self._deg_to_rad(cyl_alpha * 5 + 10)

        cyl_beta = cyl_shape_coeffs[..., self.num_cylinders:]
        cyl_beta = self._deg_to_rad(cyl_beta * 50 + 50)

        # Combine Shape defining coefficients
        harm_shape_coeffs = torch.concatenate([cyl_alpha, cyl_beta], dim=-1)

        return harmonic_amps, harmonic_fm, harm_shape_coeffs, cyl_phase_offset

    @staticmethod
    def _deg_to_rad(x, deg_reference: int = 720):
        return x / deg_reference * 2 * torch.pi


class PulseNoiseHead(nn.Module):
    """Specialized head for modulated noise bank parameters"""

    def __init__(self, hidden_dim, num_noisebands=256):
        super().__init__()
        self.num_noisebands = num_noisebands

        self.noise_amps = nn.Linear(hidden_dim, num_noisebands)
        self.noise_shape = nn.Linear(hidden_dim, 2)
        self.pulse_gain = nn.Linear(hidden_dim, 1)
        self.flow_gain = nn.Linear(hidden_dim, 1)

    def forward(self, x):

        # Noise band gains
        noise_amps = 0.03 * torch.pow(torch.sigmoid(self.noise_amps(x)), LOG10) + eps

        # Noise Shaping
        noise_shape_coeffs = torch.sigmoid(self.noise_shape(x))
        noise_shape_coeffs = torch.clamp(noise_shape_coeffs, 0.001, 0.999)
        noise_alpha = noise_shape_coeffs[..., 0:1]
        noise_alpha = self._deg_to_rad(noise_alpha * 5 + 10)  # 5 - 15 degree | 0,044 - 0,13 rad
        noise_alpha = 2 * noise_alpha
        noise_beta = noise_shape_coeffs[..., 1:2]
        noise_beta = self._deg_to_rad(noise_beta * 50 + 50)  # 50 - 100 degree | 0,44 - 0,87 rad
        noise_beta = 2 * noise_beta
        noise_shape_coeffs = torch.concatenate([noise_alpha, noise_beta], dim=-1)

        # Noise Gains
        pulse_noise_gain = torch.sigmoid(self.pulse_gain(x)) * 0.75 + 0.25
        flow_noise_gain = torch.sigmoid(self.flow_gain(x)) * 0.75 + 0.25

        # Combine parameters
        noise_params = torch.cat((
            noise_amps,
            noise_shape_coeffs,
            pulse_noise_gain,
            flow_noise_gain
        ), dim=-1)

        return noise_params

    @staticmethod
    def _deg_to_rad(x, deg_reference: int = 720):
        return x / deg_reference * 2 * torch.pi


class MLPBlock(nn.Module):
    """MLP Block consisting of Linear -> LayerNorm -> LeakyReLU"""

    def __init__(self, in_size: int = 128, out_size: int = None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size or in_size

        self.block = nn.Sequential(
            nn.Linear(self.in_size, self.out_size),
            nn.LayerNorm(self.out_size),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.block(x)


class PulseTrainModel(nn.Module):
    def __init__(
            self,
            synthesizer: object,
            num_harmonics: int = 64,
            num_cylinders: int = 8,
            num_noisebands: int = 256,
            feature_size: int = 3,
            rpm_range: tuple = (0, 10000),
            hidden_size: int = 128,
            gru_size: int = 128,
            gru_layers: int = 1
    ):
        super().__init__()

        # Store hyperparameters
        self.feature_size = feature_size
        self.num_harmonics = num_harmonics
        self.num_noisebands = num_noisebands
        self.num_cylinders = num_cylinders
        self.rpm_range = rpm_range

        # Feature processors
        self.rpm_net = MLPBlock(feature_size, 128)
        self.nm_net = MLPBlock(feature_size, 128)

        # Learnable hidden state
        self.h0 = nn.Parameter(
            torch.zeros(gru_layers, 1, gru_size)
        )
        nn.init.xavier_uniform_(self.h0)

        # GRU with skip connection
        self.gru = nn.GRU(
            input_size=128 * 2,
            hidden_size=gru_size,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False
        )

        # Post-GRU processing
        self.gru_norm = nn.LayerNorm(gru_size)
        self.post_gru = nn.Sequential(
            MLPBlock(gru_size + feature_size * 2, int(gru_size / 4)),
            MLPBlock(int(gru_size / 4), int(gru_size / 2.5)),
            MLPBlock(int(gru_size / 2.5), hidden_size),
        )

        # Specialized heads
        self.cylinder_head = PulseHarmonicsHead(
            hidden_size,
            num_cylinders=num_cylinders,
            num_harmonics=num_harmonics
        )
        self.noise_head = PulseNoiseHead(
            hidden_size,
            num_noisebands=num_noisebands,
        )

        # Setup procedural engines synth
        self.synthesizer = synthesizer

        # Apply Xavier initialization
        self.apply(self._init_weights)

    def forward(self, inputs: torch.Tensor, controls: torch.Tensor) -> dict:
        """Forward pass through ProceduralEnginesModel.

        Args:
            inputs (Tensor): Features from Dataloader [batch, seq, features]
            controls (Tensor): RPM, Torque envelopes for conditioning [batch, 2, audio_length]

        Returns:
            synth_params (dict): Organized synth parameters ready for synthesis
        """
        # Separate features
        rpm_features = inputs[..., :self.feature_size]
        nm_features = inputs[..., self.feature_size:]

        # Feature embedding
        x_rpm = self.rpm_net(rpm_features)
        x_nm = self.nm_net(nm_features)

        # Concatenate embeddings
        x = torch.cat([x_rpm, x_nm], dim=-1)

        # Temporal embedding
        h0 = self.h0.expand(-1, x.size(0), -1).contiguous()
        x, _ = self.gru(x, h0)
        x = self.gru_norm(x)

        # Skip connection with original features
        x = torch.cat([x, rpm_features, nm_features], dim=-1)

        # Post-GRU MLP stack
        x = self.post_gru(x)

        # Harminc Parameter Head
        amplitudes, freq_mod, cyl_shapes, cyl_offsets = self.cylinder_head(x)
        noise_params = self.noise_head(x)

        # rpm trace
        rpm_placeholder = torch.ones_like(x[:, :, 0:1])

        # Combine all parameters
        synth_params = torch.cat((
            rpm_placeholder,
            amplitudes,
            freq_mod,
            cyl_offsets,
            cyl_shapes,
            noise_params,
        ), dim=-1)

        # reshape to B, H, L for upsampling
        synth_params = synth_params.permute(0, 2, 1)

        # Upsample to audio sample rate
        synth_params = self._upsample(synth_params, target_size=controls.size(2))

        # Scale rpm placeholder with fundamental frequency
        synth_params = self._scale_frequencies(synth_params, controls)

        # Condition Noise Pulse depth with Torque / Throttle
        synth_params = self._scale_noise_balance(synth_params, controls)

        # Sort Params
        params = self._make_param_dict(synth_params)

        # Generate audio
        output = self.synthesizer(**params)

        return output

    @staticmethod
    def _init_weights(module):
        """Xavier (Glorot) initialization for linear layers"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _scale_frequencies(self, synth_params, controls):
        f0_env = self._rpm_to_freq(controls[:, 0]).unsqueeze(1)
        synth_params[:, 0:1, :] = synth_params[:, 0:1, :] * f0_env
        return synth_params

    @staticmethod
    def _scale_noise_balance(synth_params, controls, min_pulse: float = 0.02, min_flow: float = 0.02):
        """
        Conditions Balance between Propulsion Mode Combustion Pulses and Engine Braking Turbulence noise.
        Idle: ~15 - 30 nm
        """

        # Isolate Positive Torque (Throttle)
        propulsion = torch.clamp(controls[:, 1:2], min_pulse, 1) ** 0.7

        # Isolate Negative Torque (Deceleration Fuel Cutoff - DFCO)
        braking = torch.clamp(-controls[:, 1:2], min_flow, 1)

        # Apply Conditioning
        conditioning = torch.concatenate([propulsion, braking], dim=1)
        synth_params[:, -2:, :] = synth_params[:, -2:, :] * conditioning
        return synth_params

    def _make_param_dict(self, outputs):
        """Organize synth params into dict"""

        # Synth parameter structure for output organization
        self.synth_param_structure = {
            "f0": 1,
            "harmonic_amplitudes": self.num_harmonics * self.num_cylinders,
            "harmonic_fm": self.num_cylinders,
            "cylinder_phase_offsets": self.num_cylinders,
            "cylinder_pulse_shapes": 2 * self.num_cylinders,
            "noisebank_amplitudes": self.num_noisebands,
            "noise_pulse_shapes": 2,
            "noise_pulse_gain": 1,
            "flow_noise_gain": 1,
        }
        synth_params = {}
        start_idx = 0
        for param_name, param_length in self.synth_param_structure.items():
            synth_params[param_name] = outputs[:, start_idx:start_idx + param_length, :]
            start_idx += param_length
        return synth_params

    def _rpm_to_freq(self, rpm: torch.Tensor):
        """Scales embedded rpm to frequency"""
        return rpm * self.rpm_range[-1] / 60

    @staticmethod
    def _upsample(x: torch.Tensor, target_size: int = 65536) -> torch.Tensor:
        """Upsample model outputs to audio rate"""
        return F.interpolate(x, size=target_size, mode='linear', align_corners=False)


def create_model(
        config: Dict[str, Any],
        synthesizer: object,
        device: torch.device) -> PulseTrainModel:
    """
    Initialize Model from config.
    """

    synth_cfg = config['synth']
    dataset_cfg = config['dataset']
    model_cfg = config['model']

    return PulseTrainModel(
        synthesizer=synthesizer,
        num_harmonics=synth_cfg["num_harmonics"],
        num_cylinders=synth_cfg["num_cylinders"],
        num_noisebands=synth_cfg["num_noisebands"],
        num_modulators=synth_cfg["num_modulators"],
        feature_size=dataset_cfg["delta_features"],
        rpm_range=dataset_cfg["rpm_range"],
        hidden_size=model_cfg["hidden_size"],
        gru_size=model_cfg["gru_size"],
        gru_layers=model_cfg["gru_layers"]
    ).to(device)
