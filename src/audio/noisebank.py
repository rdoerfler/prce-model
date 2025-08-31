import torch
import torch.nn as nn
import numpy as np


class NoiseBank(nn.Module):
    """ERB Cosine Noise Band Synthesizer. """
    def __init__(
            self,
            num_noisebands: int = 12,
            chunk_size: int = 8192,
            sample_rate: int = 48000):
        super().__init__()

        # Filterbank Params
        self.chunk_size = chunk_size
        self.half_fft = int(chunk_size // 2 + 1)
        self.sample_rate = sample_rate
        self.num_bands = num_noisebands
        self.freqs = np.linspace(0, sample_rate // 2, self.half_fft)

        # Initialise Noisebank
        filterbank, noisebank = self._initialise_filterbank()
        self.register_buffer('filterbank', filterbank)
        self.register_buffer('noisebank', noisebank)

    @staticmethod
    def hz_to_erb(frequency):
        """Convert Hz to ERB number."""
        return 21.4 * np.log10(1 + 0.00437 * frequency)

    @staticmethod
    def erb_to_hz(erb):
        """Convert ERB number to Hz."""
        return (10 ** (erb / 21.4) - 1) / 0.00437

    def _initialise_filterbank(self):
        """Initialise Filterbank."""
        filterbank = self._construct_filters()
        noisebank = self._construct_noisebank(filterbank)
        return filterbank, noisebank

    def _construct_filters(self):
        """ERB Scaled Cosine Shaped Filterbank."""
        # Create ERB range
        erb_low = self.hz_to_erb(10)
        erb_high = self.hz_to_erb(self.sample_rate // 2)

        # Split in equally distant bands (linear)
        erb_lims = np.linspace(erb_low, erb_high, self.num_bands + 2)

        # Convert back to hz (-> logarithmic spacing)
        cutoffs_hz = self.erb_to_hz(erb_lims)

        # Allocate array for cosine filters
        filterbank = np.zeros([self.num_bands, self.half_fft])

        for erb in range(self.num_bands):

            # Overlap adjacent filters
            low = cutoffs_hz[erb]
            high = cutoffs_hz[erb + 2]

            # Find frequency band limits
            band_start = np.where(self.freqs > low)[0][0]
            band_end = np.where(self.freqs < high)[0][-1]
            center = (self.hz_to_erb(low) + self.hz_to_erb(high)) / 2
            bandwidth = self.hz_to_erb(high) - self.hz_to_erb(low)

            # Construct filters
            filterbank[erb, band_start:band_end + 1] = np.cos(
                (self.hz_to_erb(self.freqs[band_start:band_end + 1]) - center) / bandwidth * np.pi
            )

        # Normalise filterbank
        filterbank = torch.tensor(filterbank, dtype=torch.float32) / (0.3 / self.half_fft) ** 0.5

        return filterbank

    def _construct_noisebank(self, filterbank):
        """Time domain noisebank based on erb filterbank."""

        # Randomize Phases
        phases = 2 * np.pi * torch.rand(1, self.num_bands, self.half_fft)

        # Apply phases
        spectrum = filterbank * torch.exp(1j * phases)

        # Synthesize Noise
        noisebank = torch.fft.irfft(spectrum, n=self.chunk_size, dim=-1)

        return noisebank

    def forward(self, gains: torch.Tensor) -> torch.Tensor:
        """Apply predicted gains (B, H, L) to time domain noisebank."""
        return torch.sum(self.noisebank * gains, dim=1, keepdim=True)
