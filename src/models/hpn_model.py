import torch
import torch.nn as nn
import torch.nn.functional as F

LN10 = torch.log(torch.tensor(10.0))
eps = 1e-8


class HarmonicHead(nn.Module):
    """Specialized head for harmonic synthesis parameters with inharmonicity."""

    def __init__(self, hidden_dim, num_harmonics: int = 128, num_gaussians: int = 4):
        super().__init__()
        self.num_harmonics = num_harmonics

        # Engine order buffer (4 stroke)
        self.register_buffer('engine_orders', torch.arange(1, num_harmonics + 1) * 0.5)

        # Separate networks for different parameter types
        self.dev_amp_logits = nn.Linear(hidden_dim, num_gaussians)
        self.dev_sigma_logits = nn.Linear(hidden_dim, num_gaussians)
        self.dev_mean_logits = nn.Linear(hidden_dim, num_gaussians)
        self.amplitude_logits = nn.Linear(hidden_dim, num_harmonics)
        self.decay_logit = nn.Linear(hidden_dim, 1)
        self.energy_logits = nn.Linear(hidden_dim, 1)

    @staticmethod
    def apply_gaussian_harmonic_shaping(
            amp_logits,
            sigma_logits,
            mean_logits,
            engine_orders,
            order_range: tuple = (1, 24)
    ):
        """
        Apply a limited number of Gaussians to modulate engine_order response.

        Args:
            amp_logits (Tensor): [B, T, G] unscaled amplitude logits
            sigma_logits (Tensor): [B, T, G] unscaled sigma logits
            mean_logits (Tensor): [B, T, G] unscaled mean logits (will be mapped to engine_order space)
            engine_orders (Tensor): [H] vector of harmonic bins, e.g., torch.linspace(0.5, 64, 128)
            order_range (Tupe): Defines Range in which gaussians can move

        Returns:
            shaped (Tensor): [B, T, H] shaped harmonic profile across engine_orders
        """
        H = engine_orders.shape[0]

        # Amplitude scaling to [-0.08, 0.08]
        amplitudes = 0.08 * torch.tanh(amp_logits)

        # Sigma scaling to [0.1, ~5.0]
        sigmas = 0.1 + 4.9 * torch.sigmoid(sigma_logits)

        # Mean scaling to [0.5, 64.0]
        if not order_range:
            mean_min = engine_orders[0].item()
            mean_max = engine_orders[-1].item()
        else:
            mean_min, mean_max = order_range
        means = mean_min + (mean_max - mean_min) * torch.sigmoid(mean_logits)  # [B, T, G]

        # Reshape for broadcasting
        eo = engine_orders.view(1, 1, 1, H)  # [1, 1, 1, H]
        means = means.unsqueeze(-1)  # [B, T, G, 1]
        sigmas = sigmas.unsqueeze(-1)  # [B, T, G, 1]
        amplitudes = amplitudes.unsqueeze(-1)  # [B, T, G, 1]

        # Apply Gaussians
        gaussians = amplitudes * torch.exp(-0.5 * ((eo - means) / sigmas) ** 2)  # [B, T, G, H]

        # Sum over G Gaussians
        shaping = gaussians.sum(dim=2)  # [B, T, H]

        return shaping

    def forward(self, x):

        # Harmonic Deviations
        harmonic_deviations = self.apply_gaussian_harmonic_shaping(
            amp_logits=self.dev_amp_logits(x),
            sigma_logits=self.dev_sigma_logits(x),
            mean_logits=self.dev_mean_logits(x),
            engine_orders=self.engine_orders
        )

        # Engine orders
        harmonic_ratios = self.engine_orders.unsqueeze(0).unsqueeze(0) * (harmonic_deviations + 1)

        # Amplitude processing
        amps_raw = torch.clamp(self.amplitude_logits(x), min=-10, max=10)
        amps = torch.softmax(amps_raw, dim=-1)

        # Apply physics-based harmonic decay
        decay_factor = torch.sigmoid(self.decay_logit(x)) ** 2
        amps = amps * torch.exp(-decay_factor * self.engine_orders)

        # Re-normalize after decay
        amps = amps / torch.sum(amps, dim=-1, keepdim=True).clamp(min=1e-6)

        # Apply energy scaling
        energy_scale = 2 * torch.pow(torch.sigmoid(self.energy_logits(x)), LN10) + eps
        amps = amps * energy_scale

        return harmonic_ratios, amps


class NoiseHead(nn.Module):
    """Specialized head for modulated noise bank parameters"""

    def __init__(self, hidden_dim, num_noisebands=256, num_modulators=4):
        super().__init__()
        self.num_noisebands = num_noisebands
        self.num_modulators = num_modulators

        # Noise band parameters
        self.band_gains = nn.Linear(hidden_dim, num_noisebands)
        self.overall_noise_gain = nn.Linear(hidden_dim, 1)

        # Modulation parameters
        self.mod_exponents = nn.Linear(hidden_dim, num_modulators)
        self.mod_decay = nn.Linear(hidden_dim, 1)
        self.pulse_gain = nn.Linear(hidden_dim, 1)
        self.flow_gain = nn.Linear(hidden_dim, 1)

        # Engine orders of Modulators
        self.register_buffer('engine_orders', torch.arange(num_modulators) * 0.5)

    def forward(self, x):
        # Noise band gains
        noise_amps = 2 * torch.pow(torch.sigmoid(self.band_gains(x)), LN10) + eps

        # Overall energy scaling
        energy_scale = torch.pow(torch.clamp(torch.sigmoid(self.overall_noise_gain(x)), min=1e-6, max=1.0), LN10)
        noise_amps = noise_amps * energy_scale

        # Modulation exponents (1-5 range)
        mod_exponents = torch.tanh(self.mod_exponents(x)) * 2 + 3

        # Decreasing effect of higher orders
        decay_factor = torch.sigmoid(self.mod_decay(x))
        mod_weights = torch.exp(-decay_factor * self.engine_orders)

        # Noise Gains
        pulse_noise_gain = torch.sigmoid(self.pulse_gain(x)) * 0.75 + 0.25
        flow_noise_gain = torch.sigmoid(self.flow_gain(x)) * 0.75 + 0.25

        # Combine parameters
        noise_params = torch.cat((
            noise_amps,
            mod_exponents,
            mod_weights,
            pulse_noise_gain,
            flow_noise_gain
        ), dim=-1)

        return noise_params


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


class HarmonicNoiseModel(nn.Module):
    def __init__(
            self,
            synthesizer: object,
            num_harmonics: int = 128,
            num_noisebands: int = 2048,
            num_modulators: int = 4,
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
        self.num_modulators = num_modulators
        self.rpm_range = rpm_range
        self.mod_idx = 2 * num_harmonics + num_noisebands + 2 * num_modulators

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

        # Specialized parameter heads
        self.harmonic_head = HarmonicHead(
            hidden_size,
            num_harmonics
        )
        self.noise_head = NoiseHead(
            hidden_size,
            num_noisebands,
            num_modulators
        )

        # Setup procedural engines synth
        self.synthesizer = synthesizer

        # Apply Xavier initialization
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        """Xavier (Glorot) initialization for linear layers"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

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
        harmonics, harmonic_amps = self.harmonic_head(x)
        noise_params = self.noise_head(x)

        # Combine all parameters
        synth_params = torch.cat((
            harmonics,
            harmonic_amps,
            noise_params,
        ), dim=-1)

        # reshape to B, H, L for upsampling
        synth_params = synth_params.permute(0, 2, 1)

        # Upsample to audio sample rate
        synth_params = self._upsample(synth_params, target_size=controls.size(2))

        # Scale harmonics with fundamental frequency
        synth_params = self._scale_frequencies(synth_params, controls)

        # Condition Noise Pulse depth with Torque / Throttle
        synth_params = self._scale_noise_mod(synth_params, controls)

        # Sort Params
        params = self._make_param_dict(synth_params)

        # Generate audio
        output = self.synthesizer(**params)

        return output

    def _scale_frequencies(self, synth_params, controls):
        f0_env = self._rpm_to_freq(controls[:, 0]).unsqueeze(1)
        synth_params[:, :self.num_harmonics, :] = synth_params[:, :self.num_harmonics, :] * f0_env
        return synth_params

    @staticmethod
    def _scale_noise_mod(synth_params, controls, min_pulse: float = 0.02, min_flow: float = 0.02):
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
            "harmonic_frequencies": self.num_harmonics,
            "harmonic_amplitudes": self.num_harmonics,
            "noisebank_amplitudes": self.num_noisebands,
            "noisebank_mod_exponents": self.num_modulators,
            "noisebank_mod_weights": self.num_modulators,
            "pulse_noise_gain": 1,
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
