
import sys
import os
import re
import wave
import tensorflow as tf
import numpy as np
from pathlib import Path
import tensorflow_hub as hub
import librosa 
_my_worksparce = os.path.dirname(os.getcwd())
sys.path.append(_my_worksparce)
from scipy.signal import filtfilt, spectrogram, cheby1, find_peaks, fftconvolve, hilbert, butter
from scipy.fft import fft
import matplotlib.pyplot as plt
import mplcursors


def load_wav(filename):
    """
    Function to load a WAV file

    Args:
        filename (str): path of audio file

    Returns:
        tuple: audio_array, frame_rate
    """
    # Open the WAV file
    with wave.open(filename, 'rb') as wav_file:
        # Extract audio data
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        audio_data = wav_file.readframes(n_frames)


        # Convert audio data to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # If stereo, split into two channels
        if n_channels == 2:
            audio_array = np.reshape(audio_array, (n_frames, 2))

        return audio_array, frame_rate

# ploting FFT function and the original one
def plot_signal_and_fft(signal, sampling_frequency):
    # Perform Fourier Transform
    signal_fft = fft(signal)
    n = len(signal)
    sec = n/sampling_frequency
    t = np.linspace(0, sec, n, endpoint=False)
    # Compute the frequency axis
    freq = np.fft.fftfreq(n, d=1/sampling_frequency)
    
    # Plotting the original signal
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.title('Original Sound Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Plotting the magnitude spectrum of the FFT
    plt.subplot(2, 1, 2)
    plt.plot(freq, np.abs(signal_fft))
    plt.title('Magnitude Spectrum of the FFT')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.xlim(0,1000)  # Limit to positive frequencies
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Return the new audio back to file
def save_wav(filename, data, rate):
    """
    Save a numpy array to a WAV file.

    Args:
        filename (str): Output filename.
        data (array): Audio data to save.
        rate (int): The sample rate of the audio.
    """
    # Ensure the data is in the correct format (16-bit PCM)
    data = np.asarray(data, dtype=np.int16)
    # Open the file
    with wave.open(filename, 'wb') as wav_file:
        # Set the parameters
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes for 16-bit audio
        wav_file.setframerate(rate)
        # Write the data
        wav_file.writeframes(data.tobytes())

def apply_to_all_wav_files(resource_directory, processing_function):
    # Loop through each file in the resource directory
    for filename in os.listdir(resource_directory):
        # Check if the file is a WAV file
        if filename.endswith(".WAV"):
            # Construct the full file path
            filepath = os.path.join(resource_directory, filename)
            # Load and process the audio file
            audio, fs_rate = load_wav(filepath)
            print("Processing file:", filename)
            processing_function(audio, fs_rate)


def plot_fft_log_scale(audio, fs_rate, title):
    # Calculating FFT
    cut_audio = audio[727650:793800]  # Taking specific breathing segment
    n = len(cut_audio)
    fft_audio = np.fft.fft(cut_audio)
    fft_freq = np.fft.fftfreq(n, d=1 / fs_rate)

    # Compute magnitude in dB
    fft_magnitude = np.abs(fft_audio)
    fft_magnitude_db = 20 * np.log10(
        fft_magnitude + 1e-6
    )  # Adding a small constant to avoid log(0)

    # Take only the positive frequencies - FFT output is symmetric
    positive_frequencies = fft_freq > 0
    fft_freq_positive = fft_freq[positive_frequencies]
    fft_magnitude_db_positive = fft_magnitude_db[positive_frequencies]

    # Plotting FFT
    plt.figure(figsize=(10, 5))
    plt.plot(fft_freq_positive, fft_magnitude_db_positive)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.show()


def plotting_spectrogram(audio, fs_rate, title):
    # Plot the signal's spectrogram
    frequencies, times, Sxx = spectrogram(audio, fs_rate)
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(
        times,
        frequencies,
        10 * np.log10(Sxx),
        shading="gouraud",
        cmap="magma",
        vmin=-20,
        vmax=20,
    )
    plt.title(title)
    plt.ylabel("Frequency [Hz]")
    plt.ylim(0, 13000)
    plt.xlabel("Time [sec]")
    plt.xlim(4, 12)
    plt.colorbar(label="Intensity [dB]")
    plt.show()


def apply_fft_and_display(audio, fs_rate):
    plot_fft_log_scale(audio, fs_rate, "Audio FFT")


def apply_spectrogram_and_display(audio, fs_rate):
    plotting_spectrogram(audio, fs_rate, "Audio Spectrogram")


def filter_and_display(audio, fs_rate):
    lowcut = 100  # Low cutoff frequency of the band-pass filter we decided to choose
    highcut = 10000  # High cutoff frequency of the band-pass filter

    normalized_audio = audio / np.max(np.abs(audio))
    length = len(audio)
    time = np.arange(0, length) / fs_rate

    # Apply the band-pass filter to the audio signal
    filter_audio = chebyshev_bandpass_filter(audio, lowcut, highcut, fs_rate)
    filtered_audio = chebyshev_bandpass_filter(filter_audio, lowcut, highcut, fs_rate)

    # Plot the original and filtered signals
    plt.figure(figsize=(12, 6))
    plt.plot(time, normalized_audio)
    plt.xlabel("Time [sec]")
    plt.ylabel("Amplitude")
    plt.title("Original Audio", fontsize=16)
    plt.grid(True)

    plt.figure(figsize=(12, 6))
    plt.plot(time, filtered_audio)
    plt.xlabel("Time [sec]")
    plt.ylabel("Amplitude")
    plt.title("Filtered Audio (band-pass)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def spectrogram_fft_before_and_after_filter(audio, fs_rate):
    # Apply the band-pass filter to the audio signal
    lowcut = 200
    highcut = 10000
    filter_audio = chebyshev_bandpass_filter(audio, lowcut, highcut, fs_rate)
    filtered_audio = chebyshev_bandpass_filter(filter_audio, lowcut, highcut, fs_rate)

    # Plot the FFT and spectrogram signals before and after filter
    plot_fft_log_scale(audio, fs_rate, "Original Audio FFT")
    plot_fft_log_scale(filtered_audio, fs_rate, "Filtered Audio FFT")

    plotting_spectrogram(audio, fs_rate, "Original Audio Spectrogram")
    plotting_spectrogram(filtered_audio, fs_rate, "Filtered Audio Spectrogram")


def chebyshev_bandpass_filter(
    audio_data, lowcut, highcut, fs, order=6, ripple=1
):  # Stable and sharp filter
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Design the Chebyshev Type I filter
    b, a = cheby1(order, ripple, [low, high], btype="band")

    # Apply the filter
    y = filtfilt(b, a, audio_data)

    return y

#? part two
def calculate_envelope(audio):
    # Calculate the analytic signal using the Hilbert transform
    analytic_signal = hilbert(audio)
    # The envelope is the magnitude of the analytic signal
    envelope = np.abs(analytic_signal)
    return envelope


def interval_signal(signal, interval_length, step):
    #num_intervals = (len(signal) - interval_length) // step + 1 # if neccecery
    intervals = np.array(
        [
            signal[i : i + interval_length]
            for i in range(0, len(signal) - interval_length + 1, step)
        ]
    )
    return intervals


def autocorrelation_fft(interval):
    interval = interval - np.mean(interval)  # Remove the mean
    result = fftconvolve(interval, interval[::-1], mode="full")
    result = result[result.size // 2 :]  # Keep only the second half
    # Normalize the fft
    max_val = np.max(np.abs(result)) 
    if max_val == 0:
        return np.zeros_like(result)
    result = result / max_val
    return result

def frame_signal(signal, fs, frame_duration=0.05, overlap=0.5):
    frame_size = int(frame_duration * fs)
    frame_step = int(frame_size * (1 - overlap))
    signal_length = len(signal)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_size)) / frame_step)) + 1

    pad_signal_length = num_frames * frame_step + frame_size
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z)

    indices = np.tile(np.arange(0, frame_size), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_size, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames

def calculate_frame_energy(frames):
    return np.sum(frames**2, axis=1)

def smooth_energy(energy, window_len=21):
    window = np.hanning(window_len)
    smooth_energy = np.convolve(energy, window/window.sum(), mode='same')
    return smooth_energy

def calculate_smoothed_energy(signal, fs, frame_duration=0.05, overlap=0.5, window_len=21):
    frames = frame_signal(signal, fs, frame_duration, overlap)
    energy = calculate_frame_energy(frames)
    smoothed_energy = smooth_energy(energy, window_len)
    
    return smoothed_energy

def plot_signal_and_smoothed_energy(signal, smoothed_energy, fs, frame_duration=0.05, overlap=0.5):
    time_signal = np.arange(len(signal)) / fs
    frame_size = int(frame_duration * fs)
    frame_step = int(frame_size * (1 - overlap))
    time_energy = np.arange(len(smoothed_energy)) * frame_step / fs

    plt.figure(figsize=(12, 6))
    plt.plot(time_signal, signal, label='Original Signal')
    plt.plot(time_energy, smoothed_energy, label='Smoothed Energy', linestyle='--')
    plt.title('Original Signal and Smoothed Energy')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_signal_and_envelope(audio, envelope, fs_rate):
    time = np.arange(len(audio)) / fs_rate

    plt.figure(figsize=(12, 6))
    plt.plot(time, audio, label="Original Signal")
    plt.plot(time, envelope, label="Envelope", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Original Signal and Envelope")
    plt.legend()
    plt.show()


def plot_intervals_and_autocorrelation(intervals, fs_rate):

    for i, interval in enumerate(intervals):
        time = np.arange(len(interval)) / fs_rate
        acorr = autocorrelation_fft(interval)

        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(time, interval)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(f"Interval {i + 1}")

        plt.subplot(2, 1, 2)
        plt.plot(np.arange(len(acorr)) / fs_rate, acorr)
        plt.xlabel("Time (s)")
        plt.ylabel("Autocorrelation")
        plt.title(f"Autocorrelation of Interval {i + 1}")

        plt.tight_layout()
        plt.show()


def calculate_spectral_centroid(audio, fs):
    # Compute the magnitude spectrum of the audio signal
    magnitude_spectrum = np.abs(fft(audio))
    length = len(audio)
    # Compute the corresponding frequencies
    freqs = np.fft.fftfreq(length, 1/fs)
    
    # Only consider the positive half of the spectrum
    positive_freqs = freqs[:length // 2]
    positive_magnitude_spectrum = magnitude_spectrum[:length // 2]
    
    # Calculate the spectral centroid
    spectral_centroid = np.sum(positive_freqs * positive_magnitude_spectrum) / np.sum(positive_magnitude_spectrum)
    
    return spectral_centroid

def calculate_energy_ratio(audio, fs_rate):
    # Divide the signal into inhalation and exhalation segments
    midpoint = len(audio) // 2
    inhale_segment = audio[:midpoint]
    exhale_segment = audio[midpoint:]
    inhale_energy = np.sum(inhale_segment**2)
    exhale_energy = np.sum(exhale_segment**2)
    if inhale_energy > 0:
        energy_ratio = exhale_energy / inhale_energy
    else:
        energy_ratio = None
    return energy_ratio


def process_audio(
    audio, fs_rate, interval_duration=15, overlap=0.5, height=0.2, distance=22100
):
    # Calculate the envelope of the audio signal
    # envelope = calculate_envelope(audio)
    # smooth = smooth_signal(envelope)
    frames = frame_signal(audio, fs_rate)
    energy = calculate_frame_energy(frames)
    smoothed_energy = smooth_energy(energy, fs_rate*0.5)
    
    # Adjust interval length and step for the envelope
    interval_length = int(interval_duration * fs_rate)
    step = int(interval_length * (1 - overlap))

    # interval the envelope
    intervals = interval_signal(smoothed_energy, interval_length, step)

    high_quality_intervals = []
    first_peak_times = []

    print(f"Number of intervals: {len(intervals)}")

    # Process each interval
    for i, interval in enumerate(intervals):
        # Compute autocorrelation of the interval
        acorr = autocorrelation_fft(interval)

        # Find peaks in the autocorrelation function
        peaks, _ = find_peaks(acorr, height=height, distance=distance)
        #print(f"Interval {i+1}: {len(peaks)} peaks found")

        # Select intervals with more than one peak as high-quality intervals
        if len(peaks) > 1:  # Adjust this condition as needed
            high_quality_intervals.append(interval)

            # Find the first peak within the specified range
            valid_peaks = [
                p for p in peaks if 0.64 <= p / fs_rate <= 2
            ]  #! the first peak will be between 0.75 to 2.14 sec
            if valid_peaks:
                first_peak_time = valid_peaks[0] / fs_rate  # Convert index to time
                first_peak_times.append(first_peak_time)
    # Calculate the average first peak time
    if first_peak_times:
        average_first_peak_time = np.mean(first_peak_times)
        print(f"Average first peak time: {average_first_peak_time} seconds")
        breathing_rate = 60 / average_first_peak_time
        print(f"BR = {round(breathing_rate)} breaths in one minute")
    else:
        average_first_peak_time = None
        print("No valid peaks found.")

    # Plot the original signal and envelope
    plot_signal_and_envelope(audio, smoothed_energy, fs_rate)

    # Plot the intervals and their autocorrelation
    plot_intervals_and_autocorrelation(intervals, fs_rate)

    return high_quality_intervals, average_first_peak_time

def process_audio_features(filepath):
    # Load the WAV file
    audio, fs_rate = load_wav(filepath)

    # Apply the Chebyshev bandpass filter
    filtered_audio = chebyshev_bandpass_filter(audio, 100, 10000, fs_rate)

    # Calculate the envelope of the audio signal
    envelope = calculate_envelope(filtered_audio)

    # Normalize the envelope
    normalized_envelope = normalize_signal(envelope)

    # Smooth the envelope to reduce noise
    smoothed_envelope = smooth_signal(normalized_envelope)

    # Calculate Spectral Centroid for the filtered audio
    spectral_centroid = calculate_spectral_centroid(filtered_audio, fs_rate)

    # Calculate Energy Ratio for the filtered audio
    energy_ratio = calculate_energy_ratio(filtered_audio, fs_rate)

    # Plot the original signal, smoothed envelope, and calculated features
    #plot_signal_and_features(filtered_audio, smoothed_envelope, spectral_centroid, energy_ratio, fs_rate)

    return spectral_centroid, energy_ratio

def plot_signal_and_features(audio, envelope, spectral_centroid, energy_ratio, fs_rate):
    time_audio = np.arange(len(audio)) / fs_rate
    time_envelope = np.arange(len(envelope)) / fs_rate

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(time_audio, audio)
    plt.title('Filtered Audio Signal')

    plt.subplot(2, 1, 2)
    plt.plot(time_envelope, envelope)
    plt.title('Smoothed Envelope')

    plt.tight_layout()
    plt.show()

    print(f"Spectral Centroid: {spectral_centroid}")
    print(f"Energy Ratio: {energy_ratio}")


#!

def load_wav_bytes(filepath):
    # Read the WAV file
    wav_bytes = tf.io.read_file(filepath)
    return wav_bytes

def preprocess_audio(audio, fs_rate): # Preprocess audio for the model
    # Resample audio to 16 kHz
    audio_resampled = librosa.resample(audio, orig_sr=fs_rate, target_sr=16000)
    return audio_resampled

# Function to interpret the model's predictions
def interpret_predictions(scores, target_class_indices, threshold=0.15, frame_step=0.48):
    detected_segments = {class_name: [] for class_name in target_class_indices.keys()}

    for i, prob in enumerate(scores):
        for class_name, class_index in target_class_indices.items():
            if prob[class_index] > threshold:
                start_time = i * frame_step
                end_time = start_time + 0.96  # Frame length in seconds
                detected_segments[class_name].append((start_time, end_time))

    # Merge adjacent or overlapping segments
    merged_segments = {class_name: [] for class_name in detected_segments.keys()}
    for class_name, segments in detected_segments.items():
        if not segments:
            continue
        segments.sort()
        merged_start, merged_end = segments[0]
        for start, end in segments[1:]:
            if start <= merged_end:  # Overlapping or contiguous segments
                merged_end = max(merged_end, end)
            else:
                merged_segments[class_name].append((merged_start, merged_end))
                merged_start, merged_end = start, end
        merged_segments[class_name].append((merged_start, merged_end))

    return merged_segments

# Function to visualize waveform, spectrogram, and top classes
def visualize_audio(waveform, spectrogram, scores, class_names):
    plt.figure(figsize=(10, 6))

    # Plot the waveform
    plt.subplot(3, 1, 1)
    plt.plot(waveform)
    plt.xlim([0, len(waveform)])
    plt.title('Waveform')

    # Plot the log-mel spectrogram
    plt.subplot(3, 1, 2)
    plt.imshow(spectrogram.T, aspect='auto', interpolation='nearest', origin='lower')
    plt.title('Spectrogram')

    # Plot and label the model output scores for the top-scoring classes
    mean_scores = np.mean(scores, axis=0)
    top_n = 10
    top_class_indices = np.argsort(mean_scores)[::-1][:top_n]
    plt.subplot(3, 1, 3)
    plt.imshow(scores[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')

    patch_padding = (0.025 / 2) / 0.01
    plt.xlim([-patch_padding-0.5, scores.shape[0] + patch_padding-0.5])
    yticks = range(0, top_n, 1)
    plt.yticks(yticks, [class_names[top_class_indices[x]] for x in yticks])
    plt.ylim(-0.5 + np.array([top_n, 0]))
    plt.title('Top Classes')

    plt.tight_layout()
    plt.show()
   
# Load YAMNet class names
def load_class_names():
    class_map_path = tf.keras.utils.get_file('yamnet_class_map.csv',
                                             'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv')
    class_names = []
    with open(class_map_path) as f:
        for line in f.readlines()[1:]:
            class_names.append(line.strip().split(',')[2])
    return class_names

class_names = load_class_names()

#!


def bandpass_filter(signal, fs_rate, lowcut=300, highcut=3000, order=5):
    """
    Apply a bandpass filter to the signal.
    """
    nyquist = 0.5 * fs_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def normalize_signal(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

def smooth_signal(signal, window_len=882):
    """
    Apply a Hanning window smoothing filter to the signal.
    """
    window = np.hanning(window_len)
    smooth_signal = np.convolve(signal, window / window.sum(), mode='same')
    return smooth_signal

def detect_breaths(envelope, fs_rate, height_factor=0.52, distance_factor=1.2):
    """
    Detect peaks and troughs in the envelope of the signal.
    """
    height = height_factor * np.max(envelope)
    distance = int(distance_factor * fs_rate)
    peaks, _ = find_peaks(envelope, height=height, distance=distance)
    troughs, _ = find_peaks(-envelope, height=height, distance=distance)
    return peaks, troughs

def find_breath_intervals(peaks, troughs):
    """
    Pair peaks and troughs to determine the start and end of each breath.
    """
    breath_intervals = []
    peak_idx = 0
    trough_idx = 0

    while peak_idx < len(peaks) and trough_idx < len(troughs):
        if troughs[trough_idx] < peaks[peak_idx]:
            start_idx = troughs[trough_idx]
            end_idx = peaks[peak_idx]
            breath_intervals.append((start_idx, end_idx))
            trough_idx += 1
        peak_idx += 1

    return breath_intervals

def validate_breath_intervals(breath_intervals, fs_rate, min_breath_duration=0.6, max_breath_duration=2.0):
    """
    Validate breath intervals by filtering out intervals that are too short or too long.
    """
    valid_intervals = []
    for start, end in breath_intervals:
        duration = (end - start) / fs_rate
        if min_breath_duration <= duration <= max_breath_duration:
            valid_intervals.append((start, end))
    return valid_intervals

def plot_signal_envelope_breaths(audio, envelope, smoothed_envelope, breath_intervals, peaks, troughs, fs_rate):
    """
    Plot the original signal, smoothed envelope, and detected breath intervals.
    """
    time_envelope = np.arange(len(envelope)) / fs_rate

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(time_envelope, envelope)
    plt.title('Envelope of the Signal')

    plt.subplot(2, 1, 2)
    plt.plot(time_envelope, smoothed_envelope)
    plt.plot(peaks / fs_rate, smoothed_envelope[peaks], "x")
    plt.plot(troughs / fs_rate, smoothed_envelope[troughs], "o")
    for start, end in breath_intervals:
        plt.axvline(x=start / fs_rate, color='g', linestyle='--')
        plt.axvline(x=end / fs_rate, color='r', linestyle='--')
    plt.title('Smoothed Envelope of the Signal with Detected Breaths')

    plt.tight_layout()
    plt.show()

def process_audio_with_breath_detection(filtered_audio, fs_rate):
    """
    Process the audio file to find breath intervals and plot the results.
    """

    # Normalize the filtered audio
    normalized_audio = normalize_signal(filtered_audio)
    
    # Calculate the envelope of the audio signal
    envelope = calculate_envelope(normalized_audio)

    # Smooth the envelope to reduce noise
    smoothed_envelope = smooth_signal(envelope)

    # Detect peaks and troughs in the smoothed envelope
    peaks, troughs = detect_breaths(smoothed_envelope, fs_rate)

    # Find breath intervals (start and end indexes)
    breath_intervals = find_breath_intervals(peaks, troughs)

    # Validate breath intervals
    valid_intervals = validate_breath_intervals(breath_intervals, fs_rate)

    # Plot the original signal, envelope, smoothed envelope, and detected breaths
    plot_signal_envelope_breaths(normalized_audio, envelope, smoothed_envelope, valid_intervals, peaks, troughs, fs_rate)

    return valid_intervals


