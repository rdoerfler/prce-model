import soundfile as sf
import scipy.signal
import numpy as np
import torch
import math
import os

""" 
DSP Utils 
"""


def normalize(waveform: torch.Tensor, unipolar: bool = False):
    """ Unipolar or Bipolar Normalization """
    # Normalize the waveform to the range [0, 1]
    waveform = waveform - torch.min(waveform, dim=-1) if unipolar else waveform
    # Normalize the waveform to the range [-1, 1]
    return waveform / torch.max(torch.abs(waveform), dim=-1).values.unsqueeze(1)


def add_duration(static_pred, duration):
    """ Converts Static Outputs to Frame Length """
    return static_pred.unsqueeze(-1).repeat(1, 1, duration)


def apply_pre_emphasis(audio, a=0.97):
    """Pre-Emphasis as simple Highpass filter
    y[n] = x[n] - α * x[n - 1]

    Input shape: [batch, channels, samples]
    """
    # Pad along the sample dimension (dim=2)
    padded = torch.nn.functional.pad(audio, (1, 0))

    # Apply pre-emphasis
    return audio - a * padded[:, :, :-1]


def apply_spectral_pre_emphasis(audio, a=0.97):
    """Apply pre-emphasis in the spectral domain

    Input shape: [batch, channels, samples]
    """
    # Create frequency-domain filter coefficients
    frame_length = audio.shape[-1]
    freqs = torch.fft.rfftfreq(frame_length, d=1.0, device=audio.device)

    # Convert coefficient 'a' to frequency response (1 - a*e^(-jω))
    # Magnitude of this is sqrt(1 + a² - 2a*cos(ω))
    filter_response = torch.sqrt(1 + a ** 2 - 2 * a * torch.cos(2 * math.pi * freqs))

    # Apply filter in frequency domain
    audio_fft = torch.fft.rfft(audio, dim=-1)
    audio_fft = audio_fft * filter_response.view(1, 1, -1)

    # Back to time domain
    return torch.fft.irfft(audio_fft, n=frame_length, dim=-1)


def lin_to_db(spectrum):
    """ Convert linear amplitude spectrum to dB scale. """
    spectrum_db = 20 * torch.log10(torch.abs(spectrum) + 1e-6)
    return spectrum_db


def db_to_lin(spectrum):
    """ Convert dB scale to linear amplitude. """
    spectrum_lin = 10 ** (spectrum / 20.0)
    return spectrum_lin


"""
File Utils
"""


def write_audio_batch(path: str = 'output_audio', data: torch.Tensor = None, sample_rate: int = 48000):
    """
    Writes batch of audio data as single files (batch, channels, samples)
    """
    data = data.cpu().detach().numpy()
    os.makedirs(path, exist_ok=True)
    for i, audio in enumerate(data):
        file_path = os.path.join(path, f"{os.path.basename(path)}_{i+1:02d}.wav")
        sf.write(file_path, data[i].transpose(), samplerate=sample_rate)


def write_audio_tests(name: str = 'test_batch', output_audio: torch.Tensor = None, sample_rate: int = 48000):
    """ Construct directory and write audio files. """
    output_path = os.path.join('..', 'data', 'audiofiles', 'output', 'tests', name)
    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, name)
    write_audio_batch(path=out_file, data=output_audio.cpu().detach().numpy(), sample_rate=sample_rate)


""" 
DDSP Utils
"""


def avg_smoothing(input_tensor, window_size=65):
    """Apply a simple moving average (box filter) smoothing to a 1D tensor."""
    # Ensure the window size is odd
    window_size = window_size if window_size % 2 == 1 else window_size + 1

    # Create a 1D kernel (a box filter)
    kernel = torch.ones(1, 1, window_size, device=input_tensor.device) / window_size

    # Apply padding to handle edges
    padding = (window_size - 1) // 2  # Pad evenly on both sides

    # Perform 1D convolution
    smoothed = torch.nn.functional.conv1d(input_tensor, kernel, padding=padding)

    return smoothed


def calculate_rms(audio, window_size=65):
    # Square the signal
    squared = audio ** 2

    # Proper shape for convolution
    if squared.dim() == 1:
        squared = squared.view(1, 1, -1)

    # Box filter kernel
    window_size = window_size if window_size % 2 == 1 else window_size + 1
    kernel = torch.ones(1, 1, window_size, device=audio.device) / window_size

    # Moving average of squared values
    padding = (window_size - 1) // 2
    mean_squared = torch.nn.functional.conv1d(squared, kernel, padding=padding)

    # Take the square root to get RMS
    rms = torch.sqrt(mean_squared + 1e-10)

    # Reshape back to original dimensions
    if audio.dim() == 1:
        rms = rms.squeeze()

    return rms


def design_lowpass_filter(numtaps=707) -> object:
    """
    Design a low-pass FIR filter using the window method
    Args:
         numtaps: Number of filter taps (should be odd)
    Return:
        taps: taps for lowpass filter application
    """
    # Normalize the cutoff frequency
    normalized_cutoff = 0.303

    # Generate the filter kernel
    taps = torch.sinc(2 * normalized_cutoff * (torch.arange(numtaps) - (numtaps - 1) / 2.))

    # Apply a Hamming window
    window = torch.blackman_window(numtaps, periodic=False)
    taps *= window

    # Normalize the taps
    taps /= torch.sum(taps)

    return taps


def filter_nyquist(self, audio):
    """Design and Apply Nyquist FIR filter to the signal."""

    # Calculate FIR Windowsize
    numtaps = self.filter_coeffs.size(0)
    padding = int((numtaps - 1) // 2)

    # Unsqueeze to match batch dimensions for convolution
    filter_coeffs = self.filter_coeffs.unsqueeze(0).unsqueeze(0)

    # Apply convolution
    filtered_audio = torch.nn.functional.conv1d(audio.unsqueeze(1), filter_coeffs, padding=padding)
    return filtered_audio.squeeze(1)


def mute_above_nyquist(self, frequencies: torch.Tensor, amplitudes: torch.Tensor, cutoff_norm: float = 0.8):
    """ Mute Frequencies above cutoff_norm [0., 1.] """
    cutoff = cutoff_norm * self.SAMPLE_RATE / 2
    if torch.max(frequencies) >= cutoff:
        amplitudes[torch.where(frequencies >= cutoff)] = 0.0
    return frequencies


"""
Network Utils
"""


def gaussian_squash(x, center=3.0, scale=1.0):
    """Biases values toward a center (e.g., 3) with a controllable spread."""

    # Shift and scale input to center around 0
    x_shifted = (x - center) / scale

    # Apply Gaussian-like transformation
    gaussian = torch.exp(-0.5 * x_shifted ** 2)

    # Rescale to favor values near the center
    return center + scale * gaussian * x_shifted


def center_of_mass_real(x, normalized: bool = False):
    """Calculate center of mass in given array."""
    x_axis = torch.linspace(0, 1, x.size(0)) if normalized else torch.arange(0, x.size(0), 1)
    return torch.sum(x_axis * x) / x.sum()


def center_of_mass(x):
    """Calculate normlized center of mass in given array."""
    return torch.sum(torch.linspace(0, 1, x.size(0)) * x) / x.sum()


def plot_spectrum(x, y, markers, fft_size, sample_rate):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('macosx')
    plt.ion()

    t = torch.arange(0, x.shape[-1]).cpu().detach().numpy() / fft_size * sample_rate

    # visualize spectral distances
    plt.figure(figsize=(14, 6))
    plt.plot(t, x.cpu().detach().numpy(), color='teal', label='Prediction')
    plt.plot(t, y.cpu().detach().numpy(), color='coral', label='Target')
    plt.xlim(0, (max(markers).item() + 10) / fft_size * sample_rate)
    [plt.axvline(m / fft_size * sample_rate, color='grey', alpha=0.5) for m in markers.cpu().detach().numpy()]
    plt.axvline(markers[0] / fft_size * sample_rate, color='grey', alpha=0.5, label='Harmonics Positions')
    plt.title('Harmonic Mask applied to X, Y', fontsize='large')
    plt.xlabel('Frequencies [hz]', fontsize='large')
    plt.ylabel('log(Magnitude)', fontsize='large')
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)


"""
Logger utils
"""


def ema(losses, alpha=0.9):
    smoothed_losses = []
    for loss in losses:
        if not smoothed_losses:
            smoothed_losses.append(loss)
        else:
            smoothed = alpha * smoothed_losses[-1] + (1 - alpha) * loss
            smoothed_losses.append(smoothed)
    return smoothed_losses


def setup_device(seed=1234):
    """Returns the best available device (CUDA, MPS or CPU)."""
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        device = "mps"
        print(f'Warning: {device} does not work with stft loss as complex values are not supported!')
        device = "cpu"
        torch.manual_seed(seed)
    else:
        device = "cpu"
        torch.manual_seed(seed)
    print("Using device:", device)
    return torch.device(device)


"""
Inference Post Processing
"""


def apply_high_shelf(audio_tensor, sample_rate=16000, freq=1000, gain_db=4.5):
    """Apply high shelf filter to boost high frequencies"""
    # Convert to numpy for scipy processing
    audio_np = audio_tensor.cpu().numpy()
    original_shape = audio_np.shape

    # Flatten to handle batch processing
    audio_flat = audio_np.reshape(-1, audio_np.shape[-1])

    # Design high shelf filter
    gain_linear = 10 ** (gain_db / 20)  # Convert dB to linear

    # Create high shelf filter using peaking EQ approximation
    nyquist = sample_rate / 2
    normalized_freq = freq / nyquist

    # Design a high shelf using a high-pass filter + gain
    b, a = scipy.signal.butter(2, normalized_freq, btype='high')

    # Apply filter to each channel
    filtered = np.zeros_like(audio_flat)
    for i in range(audio_flat.shape[0]):
        # Apply high-pass
        high_passed = scipy.signal.filtfilt(b, a, audio_flat[i])
        # Blend original + boosted high frequencies
        filtered[i] = audio_flat[i] + (gain_linear - 1) * high_passed

    # Reshape back and convert to torch
    filtered = filtered.reshape(original_shape)
    return torch.from_numpy(filtered).to(audio_tensor.device)


def post_processing(x, fades: bool = True, norm: bool = True, boost_high: bool = False):

    # Apply high shelf boost
    x = apply_high_shelf(x, freq=850, gain_db=14) if boost_high else x

    # normalize
    x = normalize(x) if norm else x

    # Set Fades
    if fades:
        fade_in = int(160 * 2)
        fade_out = int(160 * 2.5)
        x[..., :fade_in] = x[..., :fade_in] * torch.linspace(0, 1, fade_in)[None, :] ** 0.5
        x[..., -fade_out:] = x[..., -fade_out:] * torch.linspace(1, 0, fade_out)[None, :]

    return x
