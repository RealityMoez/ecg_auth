import wfdb
import numpy as np
from scipy.signal import butter, filtfilt, detrend
from biosppy.signals import ecg

def load_ecg_data(record_path):
    record = wfdb.rdrecord(record_path)
    ecg_data = record.p_signal
    return ecg_data

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def preprocess_ecg_data(ecg_data, fs=1000.0, order=3, lowcut=0.5, highcut=50.0):
    # Extract the required lead (e.g., lead II)
    lead_ii = ecg_data[:, 1]

    # Filter the ECG data using a bandpass filter
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_ecg_data = filtfilt(b, a, lead_ii)

    # Apply baseline correction using detrend function
    corrected_ecg_data = detrend(filtered_ecg_data)

    return corrected_ecg_data

def segment_ecg_data(ecg_data, fs=1000.0):
    out = ecg.hamilton_segmenter(signal=ecg_data, sampling_rate=fs)
    qrs_peaks = out['rpeaks']

    # Segment the ECG signal into individual heartbeats
    heartbeats = []
    for i in range(len(qrs_peaks) - 1):
        start = qrs_peaks[i]
        end = qrs_peaks[i + 1]
        heartbeat = ecg_data[start:end]
        heartbeats.append(heartbeat)
    
    return np.array(heartbeats)