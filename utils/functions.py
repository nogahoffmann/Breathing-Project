
import sys
import os
import re
import wave
import numpy as np
from pathlib import Path
import tensorflow_hub as hub
import librosa 
_my_worksparce = os.path.dirname(os.getcwd())
sys.path.append(_my_worksparce)
from scipy.signal import filtfilt, spectrogram, cheby1, find_peaks, fftconvolve, hilbert, stft,butter
from scipy.fft import fft
import matplotlib.pyplot as plt
import mplcursors
import librosa.display

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
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        
        # Normalize audio data to the range [-1, 1]
        audio_array /= np.iinfo(np.int16).max

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


def chebyshev_bandpass_filter(audio_data, lowcut, highcut, fs, order=6, ripple=1):  # Stable and sharp filter
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Design the Chebyshev Type I filter
    b, a = cheby1(order, ripple, [low, high], btype="band")

    # Apply the filter
    y = filtfilt(b, a, audio_data)

    return y


#? part two functions - feature extraction
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


def smooth_signal(signal, window_len=4410):
    # Apply a Hanning window smoothing filter
    window = np.hanning(window_len)
    smooth_signal = np.convolve(signal, window/window.sum(), mode='same')
    return smooth_signal

def plot_signal_and_envelope(audio, envelope, fs_rate):
    time = np.arange(len(audio[441000:882000])) / fs_rate

    plt.figure(figsize=(12, 6))
    plt.plot(time, audio[441000:882000], label="Original Signal")
    plt.plot(time, envelope[441000:882000], label="Envelope")
    plt.xlabel("Time (s)", fontsize=15)
    plt.ylabel("Amplitude", fontsize=15)
    plt.title("Original Signal and Envelope", fontsize=17)
    plt.legend()
    plt.show()


def plot_intervals_and_autocorrelation(intervals, fs_rate):

    for i, interval in enumerate(intervals):
        time = np.arange(len(interval)) / fs_rate
        acorr = autocorrelation_fft(interval)

        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(time, interval)
        plt.xlabel("Time (s)", fontsize=15)
        plt.ylabel("Amplitude", fontsize=15)
        plt.title(f"Interval {i + 1}", fontsize=18)

        plt.subplot(2, 1, 2)
        plt.plot(np.arange(len(acorr)) / fs_rate, acorr)
        plt.xlabel("Time (s)", fontsize=15)
        plt.ylabel("Autocorrelation", fontsize=15)
        plt.title(f"Autocorrelation of Interval {i + 1}", fontsize=18)

        plt.tight_layout()
        plt.show()


def process_audio_breathing_rate( audio, fs_rate, interval_duration=10, overlap=0.5, height=0.3, distance=22100):
    # Calculate the envelope of the audio signal
    envelope = calculate_envelope(audio)
    smooth = smooth_signal(envelope)
    
    # Adjust interval length and step for the envelope
    interval_length = int(interval_duration * fs_rate)
    step = int(interval_length * (1 - overlap))

    # interval the envelope
    intervals = interval_signal(smooth, interval_length, step)

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
        if len(peaks) >= 1:  
        
            # Find the first peak within the specified range
            valid_peaks = [
                p for p in peaks if 0.64 <= p / fs_rate <= 2.75
            ]  #! the first peak will be between 0.75 to 2.5 sec
            if valid_peaks:
                high_quality_intervals.append(interval)
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
        #breathing_rate = 40
        print("No valid peaks found.")

    # Plot the original signal and envelope
    #plot_signal_and_envelope(audio, smooth, fs_rate)

    # Plot the intervals and their autocorrelation
    #plot_intervals_and_autocorrelation(high_quality_intervals, fs_rate)

    return high_quality_intervals, average_first_peak_time, breathing_rate

def classify_inhalation_exhalation(intervals, fs_rate):
    inhalation_exhalation_segments = []
    for interval in intervals:
        sorted_interval = np.sort(interval)
        cumulative_sum = np.cumsum(sorted_interval)
        total_sum = cumulative_sum[-1]
        threshold_index = np.searchsorted(cumulative_sum, 0.75 * total_sum)
        threshold = sorted_interval[threshold_index]
        inhalation = np.where(interval < threshold)[0]
        exhalation = np.where(interval >= threshold)[0]
        inhalation_time = len(inhalation) / fs_rate
        exhalation_time = len(exhalation) / fs_rate
        inhalation_exhalation_segments.append((inhalation_time, exhalation_time))
    return inhalation_exhalation_segments

def plot_intervals_with_threshold(intervals, fs_rate):
    for i, interval in enumerate(intervals):
        sorted_interval = np.sort(interval)
        cumulative_sum = np.cumsum(sorted_interval)
        total_sum = cumulative_sum[-1]
        threshold_index = np.searchsorted(cumulative_sum, 0.75 * total_sum)
        threshold = sorted_interval[threshold_index]
        time = np.arange(len(interval)) / fs_rate
        plt.figure(figsize=(12, 6))
        plt.plot(time, interval, label="Interval")
        plt.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(f"Interval {i + 1} with Threshold")
        plt.legend()
        plt.show()


def calculate_spectral_centroid(interval, fs_rate, lowcut=300, highcut=19000):
    # Normalize the interval
    #interval = (interval - np.mean(interval)) / np.std(interval)
    
    # Compute the Short-Time Fourier Transform (STFT)
    f, t, Zxx = stft(interval, fs_rate, nperseg=256, noverlap=128)
    
    # Calculate the magnitude spectrum
    magnitude_spectrum = np.abs(Zxx)
    
    # Frequency mask to include only the desired range
    freq_mask = (f >= lowcut) & (f <= highcut)
    
    # Apply mask to frequencies and magnitude spectrum
    filtered_frequencies = f[freq_mask]
    filtered_spectrum = magnitude_spectrum[freq_mask, :]
    
    # Initialize an empty list to store spectral centroids
    spectral_centroids = []
    
    # Calculate spectral centroids for each frame
    for i in range(filtered_spectrum.shape[1]):
        numerator = np.sum(filtered_frequencies * filtered_spectrum[:, i])
        denominator = np.sum(filtered_spectrum[:, i])
        spectral_centroid = numerator / denominator if denominator != 0 else 0
        spectral_centroids.append(spectral_centroid)
    
    return spectral_centroids

def calculate_average_spectral_centroid(high_quality_intervals, fs_rate):
    all_spectral_centroids = []
    for interval in high_quality_intervals:
        centroids = calculate_spectral_centroid(interval, fs_rate)
        all_spectral_centroids.extend(centroids)
    
    average_spectral_centroid = np.mean(all_spectral_centroids) if all_spectral_centroids else None
    return average_spectral_centroid
        

#!
def calculate_average_breath_times(label_file):
    inhalation_times = []
    exhalation_times = []
    inhalation_segments = []
    exhalation_segments = []

    with open(label_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            columns = line.strip().split()
            if len(columns) == 3:
                start_time = float(columns[0])
                end_time = float(columns[1])
                breath_type = columns[2]

                if breath_type == 'i':
                    inhalation_times.append(end_time - start_time)
                    inhalation_segments.append((start_time, end_time))
                elif breath_type == 'e':
                    exhalation_times.append(end_time - start_time)
                    exhalation_segments.append((start_time, end_time))
                else: 
                    continue  # Skip lines that are neither 'i' nor 'e'

    avg_inhalation_time = np.mean(inhalation_times) if inhalation_times else 0
    avg_exhalation_time = np.mean(exhalation_times) if exhalation_times else 0

    return inhalation_segments, exhalation_segments, avg_inhalation_time, avg_exhalation_time


def calculate_segment_energy(audio, fs, segments):
    energies = []
    for start_time, end_time in segments:
        start_sample = int(start_time * fs)
        end_sample = int(end_time * fs)
        segment = audio[start_sample:end_sample]
        
        # Ensure the segment length is greater than zero to avoid division by zero
        if len(segment) > 0:
            energy = np.sum(segment ** 2) / len(segment)
            energies.append(energy)
    
    if energies:
        # Convert to decibels, add a small constant to avoid log(0)
        energies_db = 10 * np.log10(np.array(energies) + 1e-10)
        return np.mean(energies_db)
    else:
        return float('-inf')  # Return a very small number if no energies are calculated
