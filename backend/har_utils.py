
import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft
import json
import os

# =============================================================================
# CONSTANTS
# =============================================================================
ORIGINAL_COLUMNS = [
    'chest_acc_x', 'chest_acc_y', 'chest_acc_z',
    'ecg_1', 'ecg_2',
    'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z',
    'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z',
    'ankle_mag_x', 'ankle_mag_y', 'ankle_mag_z',
    'arm_acc_x', 'arm_acc_y', 'arm_acc_z',
    'arm_gyro_x', 'arm_gyro_y', 'arm_gyro_z',
    'arm_mag_x', 'arm_mag_y', 'arm_mag_z',
    'label'  # Input might or might not have label
]

CHANNEL_NAMES = [
    'chest_acc_x', 'chest_acc_y', 'chest_acc_z',
    'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z',
    'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z',
    'ankle_mag_x', 'ankle_mag_y', 'ankle_mag_z',
    'arm_acc_x', 'arm_acc_y', 'arm_acc_z',
    'arm_gyro_x', 'arm_gyro_y', 'arm_gyro_z',
    'arm_mag_x', 'arm_mag_y', 'arm_mag_z',
    'ankle_acc_mag', 'ankle_gyro_mag', 'arm_acc_mag', 'arm_gyro_mag',
    'chest_acc_mag'
]

ACTIVITY_LABELS = {
    1: 'Standing still',
    2: 'Sitting and relaxing',
    3: 'Lying down',
    4: 'Walking',
    5: 'Climbing stairs',
    6: 'Waist bends forward',
    7: 'Frontal elevation of arms',
    8: 'Knees bending (crouching)',
    9: 'Cycling',
    10: 'Jogging',
    11: 'Running',
    12: 'Jump front & back'
}

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def compute_vector_magnitude(df, x_col, y_col, z_col):
    return np.sqrt(df[x_col]**2 + df[y_col]**2 + df[z_col]**2)

def add_all_magnitude_features(df):
    df = df.copy()
    
    # Magnitudes
    df['chest_acc_mag'] = compute_vector_magnitude(df, 'chest_acc_x', 'chest_acc_y', 'chest_acc_z')
    df['ankle_acc_mag'] = compute_vector_magnitude(df, 'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z')
    df['ankle_gyro_mag'] = compute_vector_magnitude(df, 'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z')
    df['arm_acc_mag'] = compute_vector_magnitude(df, 'arm_acc_x', 'arm_acc_y', 'arm_acc_z')
    df['arm_gyro_mag'] = compute_vector_magnitude(df, 'arm_gyro_x', 'arm_gyro_y', 'arm_gyro_z')
    
    return df

def compute_energy_fft(signal):
    fft_vals = fft(signal)
    energy = np.sum(np.abs(fft_vals[1:len(fft_vals)//2])**2)
    return energy

def extract_window_features(window):
    """
    Extract 7 stats from each channel in window.
    window shape: (100, 26)
    """
    n_channels = window.shape[1]
    features = []
    
    for ch in range(n_channels):
        signal = window[:, ch]
        
        features.extend([
            np.mean(signal),
            np.std(signal),
            np.min(signal),
            np.max(signal),
            np.median(signal),
            stats.skew(signal),
            compute_energy_fft(signal)
        ])
    
    return np.array(features)

def standardize_features(features, scaler_params):
    mean = np.array(scaler_params['mean'])
    scale = np.array(scaler_params['scale'])
    return (features - mean) / scale

# =============================================================================
# DATA LOADING (For Simulation)
# =============================================================================
def load_subject_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = pd.read_csv(file_path, sep='\t', header=None, names=ORIGINAL_COLUMNS)
    return data

def preprocess_data(data):
    # Remove null class
    data = data[data['label'] != 0].copy()
    # Add magnitudes
    data = add_all_magnitude_features(data)
    return data

def create_windows(data, window_size=100, step_size=50):
    windows = []
    labels = []
    
    # Assuming data is from one subject, but let's handle activity blocks
    unique_labels = data['label'].unique()
    
    for label in unique_labels:
        label_data = data[data['label'] == label]
        X = label_data[CHANNEL_NAMES].values
        
        for start in range(0, len(X) - window_size + 1, step_size):
            end = start + window_size
            window = X[start:end]
            if len(window) == window_size:
                windows.append(window)
                labels.append(label)
                
    return np.array(windows), np.array(labels)

def create_sequential_windows(data, window_size=100, step_size=50):
    """
    Create windows sequentially to preserve time order for visualization.
    """
    windows = []
    labels = []
    f_indices = [] # keep track of indices to reconstruct time
    
    # We assume 'label' column exists. If not, we handle it.
    has_label = 'label' in data.columns
    
    X = data[CHANNEL_NAMES].values
    y = data['label'].values if has_label else np.zeros(len(data))
    
    for start in range(0, len(X) - window_size + 1, step_size):
        end = start + window_size
        window = X[start:end]
        
        if len(window) == window_size:
            windows.append(window)
            # Use the label from the middle of the window or max voting
            # For simplicity & timeline, let's take the mode (most common)
            window_labels = y[start:end]
            if len(window_labels) > 0:
                mode_label = stats.mode(window_labels, keepdims=False)[0]
                labels.append(mode_label)
            else:
                labels.append(0)
            
            f_indices.append(start)
            
    return np.array(windows), np.array(labels), np.array(f_indices)
