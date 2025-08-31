import torch
from typing import List, Any, Dict


"""
Loss Functions for Differentiable Procedural Engine Synth Model.
"""


class EngineSoundLoss(torch.nn.Module):
    """Combined Loss Functions for Procedural Engines Model optimization."""
    def __init__(
            self,
            harm_loss_params: dict,
            stft_loss_params: dict,
            w_harm: float = 1.0,
            w_stft: float = 1.0):
        super().__init__()

        # Instantiate loss functions
        self.harmonic_loss = HarmonicLoss(**harm_loss_params)
        self.stft_loss = MultiResSpectralLoss(**stft_loss_params)

        # Instantiate weights
        self.w_harm = w_harm
        self.w_stft = w_stft

    def forward(self, x, y, control):

        # Clipping to avoid extreme values going into loss
        x = torch.clamp(x, -1, 1)

        # compute loss terms
        harm_loss = self.harmonic_loss(x, y, control) if self.w_harm > 0 else torch.tensor(0.0, device=y.device)
        stft_loss = self.stft_loss(x, y) if self.w_stft > 0 else torch.tensor(0.0, device=y.device)

        # reduction
        loss = torch.mean(
            torch.stack([
                harm_loss * self.w_harm,
                stft_loss * self.w_stft,
            ])
        )

        return loss, {'harmonic': harm_loss.item(), 'stft': stft_loss.item()}


class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module.

    See [Arik et al., 2018](https://arxiv.org/abs/1808.06719).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class STFTMagnitudeLoss(torch.nn.Module):
    """STFT magnitude loss."""

    def __init__(self, log=True, distance="L1", reduction="mean"):
        super().__init__()
        self.log = log
        if distance == "L1":
            self.distance = torch.nn.L1Loss(reduction=reduction)
        elif distance == "L2":
            self.distance = torch.nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Invalid distance: '{distance}'.")

    def forward(self, x_mag, y_mag):
        if self.log:
            x_mag = torch.log(x_mag)
            y_mag = torch.log(y_mag)
        return self.distance(x_mag, y_mag)


class STFTEnergyLoss(torch.nn.Module):
    def __init__(
            self,
            log: bool = True,
            relative: bool = True,
            eps: float = 1e-8,
            reduction: str = 'mean'):
        super().__init__()

        self.reduction = reduction
        self.relative = relative
        self.log = log
        self.eps = eps
        self.l1_distance = torch.nn.L1Loss(reduction=self.reduction)
        self.mse_distance = torch.nn.MSELoss(reduction=self.reduction)

    def forward(self, x_mag, y_mag):
        """
        Args:
            x_mag, y_mag: Tensor of shape (batch_size, freq_bins, frames)
        """
        # Compute frame-wise energy: shape (batch_size, frames)
        x_energy = torch.sum(x_mag ** 2, dim=1)
        y_energy = torch.sum(y_mag ** 2, dim=1)

        # Relative normalization per batch sample
        if self.relative:
            x_energy = x_energy / (torch.sum(x_energy, dim=1, keepdim=True) + self.eps)
            y_energy = y_energy / (torch.sum(y_energy, dim=1, keepdim=True) + self.eps)

        # Log scaling
        if self.log:
            x_energy = torch.log(x_energy + self.eps)
            y_energy = torch.log(y_energy + self.eps)
            distance = self.l1_distance(x_energy, y_energy)
        # Lin scaling
        else:
            distance = self.mse_distance(x_energy, y_energy)

        return distance


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(
        self,
        fft_size: int = 1024,
        hop_size: int = 256,
        win_length: int = 1024,
        window: str = "blackman",
        w_sc: float = 0.0,
        w_log_mag: float = 0.0,
        w_lin_mag: float = 1.0,
        w_energy: float = 0.0,
        equal_contribution: bool = True,
        scale_invariance: bool = False,
        eps: float = 1e-8,
        output: str = "loss",
        reduction: str = "mean",
        mag_distance: str = "L1",
        device: Any = None,
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.w_sc = w_sc
        self.w_log_mag = w_log_mag
        self.w_lin_mag = w_lin_mag
        self.w_energy = w_energy
        self.scale_invariance = scale_invariance
        self.equal_contribution = equal_contribution
        self.eps = eps
        self.output = output
        self.reduction = reduction
        self.mag_distance = mag_distance
        self.device = device

        # initialise loss terms
        self.spectralconv = SpectralConvergenceLoss()
        self.logstft = STFTMagnitudeLoss(
            log=True,
            reduction=reduction,
            distance=mag_distance,
        )
        self.linstft = STFTMagnitudeLoss(
            log=False,
            reduction=reduction,
            distance=mag_distance,
        )
        self.energy_loss = STFTEnergyLoss(
            reduction=reduction,
            log=True,
            relative=True,
        )

    def stft(self, x):
        x_stft = torch.stft(
            x,
            self.fft_size,
            self.hop_size,
            self.win_length,
            self.window,
            return_complex=True,
        )
        x_mag = torch.sqrt(torch.clamp((x_stft.real ** 2) + (x_stft.imag ** 2), min=self.eps))
        x_phs = torch.atan2(x_stft.imag, x_stft.real)
        return x_mag, x_phs

    def forward(self, input: torch.Tensor, target: torch.Tensor):

        # compute the magnitude and phase spectra of input and target
        self.window = self.window.to(input.device)
        x_mag, x_phs = self.stft(input.view(-1, input.size(-1)))
        y_mag, y_phs = self.stft(target.view(-1, target.size(-1)))

        if self.equal_contribution:
            # normalize by fft size to compensate for different resolutions
            x_mag = x_mag / torch.tensor(self.fft_size / 2, dtype=torch.float)
            y_mag = y_mag / torch.tensor(self.fft_size / 2, dtype=torch.float)

        # Compute Energy Loss before scaling
        energy_loss = self.energy_loss(x_mag, y_mag) * self.w_energy if self.energy_loss else 0.0

        # normalize scales
        if self.scale_invariance:
            alpha = (x_mag * y_mag).sum([-2, -1]) / ((y_mag**2).sum([-2, -1]))
            y_mag = y_mag * alpha[:, None, None]

        # compute loss terms
        sc_loss = self.spectralconv(x_mag, y_mag) * self.w_sc if self.w_sc else 0.0
        log_loss = self.logstft(x_mag, y_mag) * self.w_log_mag if self.w_log_mag else 0.0
        lin_loss = self.linstft(x_mag, y_mag) * self.w_lin_mag if self.w_lin_mag else 0.0

        # combine loss terms
        loss = energy_loss + sc_loss + log_loss + lin_loss

        # reduce loss
        loss = torch.mean(loss) if self.reduction == 'mean' else torch.sum(loss)

        return loss


class MultiResSpectralLoss(torch.nn.Module):
    """Multi resolution STFT loss."""
    def __init__(
        self,
        fft_sizes: List[int] = [16384, 8192, 4096, 1024, 512],
        overlap: float = 0.75,
        window: str = "hanning_window",
        w_sc: float = 0.1,
        w_log_mag: float = 0.1,
        w_lin_mag: float = 0.1,
        w_energy: float = 0.1,
        scale_invariance: bool = False,
        eps: float = 1e-8,
        **kwargs
    ):
        super().__init__()
        self.eps = eps

        # Initialse losses
        self.stft_losses = torch.nn.ModuleList()
        for fft_size, hop in zip(fft_sizes, [int(res * (1 - overlap)) for res in fft_sizes]):
            self.stft_losses += [
                STFTLoss(
                    fft_size=fft_size,
                    hop_size=hop,
                    win_length=fft_size,
                    window=window,
                    w_sc=w_sc,
                    w_log_mag=w_log_mag,
                    w_lin_mag=w_lin_mag,
                    w_energy=w_energy,
                    scale_invariance=scale_invariance,
                    **kwargs,
                )
            ]

    def forward(self, x, y):
        """Compute STFT Losses for multiple resolutions."""
        mrstft_loss = 0.0
        for f in self.stft_losses:
            loss_term = f(x, y)

            # Clip individual loss terms to prevent extreme values
            loss_term = torch.clamp(loss_term, min=-10, max=10)
            mrstft_loss += loss_term

        # Safe division and averaging
        mrstft_loss = mrstft_loss / (len(self.stft_losses) + self.eps)
        return mrstft_loss


class HarmonicLoss(torch.nn.Module):
    """
    Measures Energy Distance at harmonics as vectorized mask.
    """
    def __init__(
            self,
            num_harmonics: int = 128,
            fft_size: int = 65536,
            win_size: int = 16384,
            hop_size: int = 2048,
            sample_rate: int = 16000,
            silhouette: str = 'clip',
            difference: str = 'energy'):
        super().__init__()

        # initialise harmonic and fft params
        self.num_harmonics = num_harmonics
        self.win_size = win_size
        self.fft_size = fft_size
        self.mag_size = self.fft_size // 2 + 1
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.difference = difference
        self.silhouette = silhouette

        # get scaling function
        if self.silhouette == 'clip':
            self.scaling_fn = self._clip_mask
        elif self.silhouette == 'sine':
            self.scaling_fn = self._scale_mask
        elif self.silhouette == 'pow_sine_2':
            self.scaling_fn = self._scale_and_focus_mask

        # allocate buffers
        self.register_buffer('harmonic_ratios', torch.arange(1, num_harmonics + 1) * 0.5)
        self.register_buffer('time', torch.arange(self.mag_size) / self.mag_size)
        self.register_buffer('window', torch.hann_window(win_size))

    def forward(self, inputs, targets, control):
        """Computs distance of energy in harmonic-mask (vectorized).

        Arguments:
            inputs (Tensor): Generated audio from model (batch, channels, samples)
            targets (Tensor): Target audio from dataset (batch, channels, samples)
            control (Tensor): Normalised rpm values (batch, 1, samples)

        Returns:
            Harmonic loss as distance of energy and center of mass around harmonic bins.
        """
        batch, ch, seq = inputs.size()

        # Unfold to frames
        c_frames = control.unfold(size=self.win_size, step=self.hop_size, dimension=-1)

        # Compute Harmonic Mask as sinusoidal shape with peaks at harmonics
        rpm_mean = torch.mean(c_frames, dim=-1).permute(0, 2, 1)
        rpm_bin = self._get_bin(rpm_mean, base=0.5, integer=False)  # 0.5th harmonic as base
        mask_shape_freq = self.mag_size / torch.clamp(rpm_bin, min=1e-7)
        mask_shape_phase = rpm_bin / self.mag_size * torch.pi * 2
        mask_shape = torch.cos(2 * torch.pi * mask_shape_freq * self.time[None, None, :] + mask_shape_phase)

        # Apply scaling function
        mask_shape = self.scaling_fn(mask_shape)

        # Blindfold mask outside expected harmonics
        mask_lower_limits = torch.floor(rpm_bin - rpm_bin / 2).squeeze(-1)
        mask_upper_limits = torch.ceil(rpm_bin * self.num_harmonics + rpm_bin / 2).squeeze(-1)
        mask = self._mask_outside_bounds(
            mask_shape,
            mask_lower_limits.long(),
            mask_upper_limits.long()
        ).unsqueeze(1)

        # Obtain inputs, targets as frames
        xy = torch.cat((inputs, targets), dim=1)
        xy_frames = xy.unfold(size=self.win_size, step=self.hop_size, dimension=-1) * self.window[None, None, None, :]
        padding = torch.zeros([batch, 2, xy_frames.size(2), self.fft_size - self.win_size], device=inputs.device)
        xy_frames = torch.cat((xy_frames, padding), dim=-1)

        # Obtain magnitudes (normalised by fft_size/2)
        xy_spectra = torch.fft.rfft(xy_frames)
        xy_mags = torch.sqrt(torch.clamp(xy_spectra.real ** 2 + xy_spectra.imag ** 2, min=1e-10))
        xy_mags = xy_mags / (self.fft_size / 2)

        # Apply harmonic mask
        xy_masked = xy_mags * mask

        # Exlude frames where mask shape is < nyquist
        valid_frames = (mask_shape_freq < self.mag_size / 2 * 0.95).unsqueeze(1)
        xy_masked = xy_masked * valid_frames

        if self.difference == 'energy':
            # Compute distance based on spectral power of masked spectra
            xy_energy = torch.sum(torch.abs(xy_masked) ** 2, dim=-1)
            xy_energy = torch.log(xy_energy + 1e-18)
            distance = torch.abs(torch.diff(xy_energy, dim=1))
            harmonic_loss = distance.sum() / valid_frames.sum().clamp(min=1)

        elif self.difference == 'spectral':
            # Convert to log domain
            xy_masked_log = torch.log(xy_masked + 1e-8)
            # Compute distance directly between masked spectra
            distance = torch.abs(torch.diff(xy_masked_log, dim=1))
            # Reduce Loss and normalise
            harmonic_loss = torch.sum(distance) / (self.num_harmonics * batch * valid_frames.sum().clamp(min=1))
            # Additional scaling
            harmonic_loss *= 0.1
        else:
            raise ValueError('Difference mode not defined. Expected "spectral" or "energy".')

        return harmonic_loss

    def _get_bin(self, f0_norm, base: float = 1, integer: bool = False):
        """
        Convert normalized f0 to bin index.

        Args:
            f0_norm (Tensor): Normalised f0 value (batch, sequence, values)
        Returns:
            f0_bin_idx (Tensor): f0 bin indices (batch, sequence, values)
        """
        bin_idx = f0_norm * 10000 / 60 / self.sample_rate * self.fft_size * base
        return bin_idx.long() if integer else bin_idx

    @staticmethod
    def _clip_mask(x):
        """
        Remove negative polarity, leaving distance between harmonics.
        """
        return torch.clamp(x, 0, 1)

    @staticmethod
    def _scale_mask(x):
        """
        Scale sinusoid to 0-1, smoothly hiding inter-harmonic material.
        """
        return x * 0.5 + 0.5

    @staticmethod
    def _scale_and_focus_mask(x):
        """
        Scale sinusoid to squared 0-1, narrowing focus on harmonics.
        """
        return (x * 0.5 + 0.5) ** 2

    @staticmethod
    def _mask_outside_bounds(tensor, lower_bounds, upper_bounds):
        """
        Limit mask to harmonic range by setting values to 0 when they exceed the given bounds.

        Args:
            tensor: Tensor of shape (batch, sequence, fft_bins)
            lower_bounds: Tensor of shape (batch, fftbin_idx) - lower bounds for each example in batch
            upper_bounds: Tensor of shape (batch, fftbin_idx) - upper bounds for each example in batch

        Returns:
            Masked tensor of shape (batch, sequence, fft_bins)
        """
        batch_size, seq_len, n_bins = tensor.shape

        # Create an index tensor for the fft_bins dimension
        bin_indices = torch.arange(n_bins, device=tensor.device)

        # Expand the bin indices to match the input tensor shape
        bin_indices = bin_indices.expand(batch_size, seq_len, n_bins)

        # Expand the bounds to match the sequence dimension
        lower_expanded = lower_bounds.unsqueeze(-1).expand(-1, seq_len, n_bins)
        upper_expanded = upper_bounds.unsqueeze(-1).expand(-1, seq_len, n_bins)
        lower_mask = bin_indices >= lower_expanded
        upper_mask = bin_indices <= upper_expanded
        combined_mask = (lower_mask & upper_mask)

        # Apply the mask to the tensor
        masked_tensor = tensor * combined_mask

        return masked_tensor


def create_loss_fn(
        config: Dict[str, Any],
        device: torch.device) -> EngineSoundLoss:

    """
    Initialize loss function from config.
    """

    loss_cfg = config['loss']
    loss_cfg['harm_loss_params']['num_harmonics'] = config['synth']['num_harmonics']
    loss_cfg['harm_loss_params']['sample_rate'] = config['dataset']['sample_rate']
    return EngineSoundLoss(**loss_cfg).to(device)
