"""
Human Activity Recognition (HAR) Pipeline using MHEALTH Dataset
================================================================
Complete ML pipeline for activity recognition with feature engineering,
model training, and ONNX export.

Author: Data Science Expert
Date: December 2025
"""

import os
import json
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00319/MHEALTHDATASET.zip"
DATA_DIR = "MHEALTHDATASET"
ZIP_FILE = "MHEALTHDATASET.zip"

TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6]
VAL_SUBJECTS = [7, 8]

WINDOW_SIZE = 100
OVERLAP = 0.5
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))

# Original column names from MHEALTH dataset
ORIGINAL_COLUMNS = [
    'chest_acc_x', 'chest_acc_y', 'chest_acc_z',           # 0-2: Chest accelerometer
    'ecg_1', 'ecg_2',                                       # 3-4: ECG signals
    'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z',           # 5-7: Left ankle accelerometer
    'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z',        # 8-10: Left ankle gyroscope
    'ankle_mag_x', 'ankle_mag_y', 'ankle_mag_z',           # 11-13: Left ankle magnetometer
    'arm_acc_x', 'arm_acc_y', 'arm_acc_z',                 # 14-16: Right arm accelerometer
    'arm_gyro_x', 'arm_gyro_y', 'arm_gyro_z',              # 17-19: Right arm gyroscope
    'arm_mag_x', 'arm_mag_y', 'arm_mag_z',                 # 20-22: Right arm magnetometer
    'label'                                                 # 23: Activity label
]

# Activity labels
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
# 1. DATA DOWNLOAD AND EXTRACTION
# =============================================================================
def download_and_extract_dataset():
    """Download and extract MHEALTH dataset if not exists."""
    if os.path.exists(DATA_DIR):
        print(f"[OK] Dataset directory '{DATA_DIR}' already exists.")
        return True
    
    print("Downloading MHEALTH dataset...")
    try:
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            print(f"\r  Progress: {percent:.1f}%", end='')
        
        urllib.request.urlretrieve(DATA_URL, ZIP_FILE, report_progress)
        print("\n[OK] Download complete.")
        
        # Extract
        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall('.')
        print("[OK] Extraction complete.")
        
        # Clean up zip file
        os.remove(ZIP_FILE)
        
        print("[OK] Cleaned up zip file.")
        
        return True
    except Exception as e:
        print(f"[ERROR] Error downloading dataset: {e}")
        return False

# =============================================================================
# 2. DATA LOADING
# =============================================================================
def load_subject_data(subject_id):
    """Load data for a specific subject."""
    file_path = os.path.join(DATA_DIR, f"mHealth_subject{subject_id}.log")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Subject file not found: {file_path}")
    
    # Load data (space-separated values)
    data = pd.read_csv(file_path, sep='\t', header=None, names=ORIGINAL_COLUMNS)
    
    return data


def load_subjects(subject_list):
    """Load and concatenate data from multiple subjects."""
    all_data = []
    
    for subject_id in subject_list:
        print(f"  Loading subject {subject_id}...")
        subject_data = load_subject_data(subject_id)
        subject_data['subject'] = subject_id
        all_data.append(subject_data)
    
    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data


# =============================================================================
# 3. FEATURE ENGINEERING - VECTOR MAGNITUDES
# =============================================================================
def compute_vector_magnitude(df, x_col, y_col, z_col):
    """Compute vector magnitude from x, y, z components."""
    return np.sqrt(df[x_col]**2 + df[y_col]**2 + df[z_col]**2)


def add_magnitude_features(df):
    """
    Add vector magnitude features for accelerometers and gyroscopes.
    
    Original: 23 columns (excluding label)
    After: 26 channels
    
    New channels:
    - ankle_acc_mag: magnitude of ankle accelerometer
    - ankle_gyro_mag: magnitude of ankle gyroscope
    - arm_acc_mag: magnitude of arm accelerometer
    - arm_gyro_mag: magnitude of arm gyroscope
    """
    df = df.copy()
    
    # Ankle accelerometer magnitude
    df['ankle_acc_mag'] = compute_vector_magnitude(
        df, 'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z'
    )
    
    # Ankle gyroscope magnitude
    df['ankle_gyro_mag'] = compute_vector_magnitude(
        df, 'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z'
    )
    
    # Arm accelerometer magnitude
    df['arm_acc_mag'] = compute_vector_magnitude(
        df, 'arm_acc_x', 'arm_acc_y', 'arm_acc_z'
    )
    
    # Arm gyroscope magnitude
    df['arm_gyro_mag'] = compute_vector_magnitude(
        df, 'arm_gyro_x', 'arm_gyro_y', 'arm_gyro_z'
    )
    
    return df


def get_feature_columns():
    """Get the 26 feature channel names (excluding label and subject)."""
    # Original 23 sensor columns (excluding label)
    original_sensor_cols = ORIGINAL_COLUMNS[:-1]  # Remove 'label'
    
    # Add 4 magnitude columns (total: 23 + 4 = 27, but we keep all)
    # Actually the original has 23 columns, we add 4 magnitudes = 27
    # But we need 26 channels - let's check the assignment
    
    # Looking at original columns:
    # 0-2: chest acc (3)
    # 3-4: ecg (2)  
    # 5-7: ankle acc (3)
    # 8-10: ankle gyro (3)
    # 11-13: ankle mag (3)
    # 14-16: arm acc (3)
    # 17-19: arm gyro (3)
    # 20-22: arm mag (3)
    # Total: 23 columns + label = 24
    
    # We add: ankle_acc_mag, ankle_gyro_mag, arm_acc_mag, arm_gyro_mag = 4
    # But to get 26, we might exclude ECG or magnetometer?
    # Let's include all 23 + 4 magnitudes = 27, but user said 26
    # Perhaps excluding one column... Let me include the most relevant
    
    # For HAR, we'll use: accelerometers, gyroscopes, and their magnitudes
    # Excluding ECG (not directly related to motion) to get 26 channels
    
    feature_cols = [
        # Chest accelerometer (3)
        'chest_acc_x', 'chest_acc_y', 'chest_acc_z',
        # Left ankle accelerometer (3)
        'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z',
        # Left ankle gyroscope (3)
        'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z',
        # Left ankle magnetometer (3)
        'ankle_mag_x', 'ankle_mag_y', 'ankle_mag_z',
        # Right arm accelerometer (3)
        'arm_acc_x', 'arm_acc_y', 'arm_acc_z',
        # Right arm gyroscope (3)
        'arm_gyro_x', 'arm_gyro_y', 'arm_gyro_z',
        # Right arm magnetometer (3)
        'arm_mag_x', 'arm_mag_y', 'arm_mag_z',
        # Magnitudes (4)
        'ankle_acc_mag', 'ankle_gyro_mag', 'arm_acc_mag', 'arm_gyro_mag'
    ]
    
    # Total: 21 + 4 = 25... still not 26
    # Let's add chest acc magnitude too
    # Actually, let me include ECG but add chest magnitude
    
    feature_cols = [
        # Chest accelerometer (3)
        'chest_acc_x', 'chest_acc_y', 'chest_acc_z',
        # ECG (2) - can be useful for activity intensity
        'ecg_1', 'ecg_2',
        # Left ankle accelerometer (3)
        'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z',
        # Left ankle gyroscope (3)
        'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z',
        # Left ankle magnetometer (3)
        'ankle_mag_x', 'ankle_mag_y', 'ankle_mag_z',
        # Right arm accelerometer (3)
        'arm_acc_x', 'arm_acc_y', 'arm_acc_z',
        # Right arm gyroscope (3)
        'arm_gyro_x', 'arm_gyro_y', 'arm_gyro_z',
        # Right arm magnetometer (3) - but we'll exclude to get 26
        # Magnitudes (4)
        'ankle_acc_mag', 'ankle_gyro_mag', 'arm_acc_mag', 'arm_gyro_mag'
    ]
    
    # 3 + 2 + 3 + 3 + 3 + 3 + 3 + 3 + 4 = 27 with arm_mag
    # Without arm_mag: 3 + 2 + 3 + 3 + 3 + 3 + 3 + 4 = 24
    # Without ECG: 3 + 3 + 3 + 3 + 3 + 3 + 3 + 4 = 25
    
    # Let's use all 23 original + 3 magnitudes (excluding arm_gyro_mag) = 26
    # Or all motion-related: exclude ECG, include all magnitudes = 25
    
    # Final decision: Use the original sensors + 4 magnitudes but exclude ECG
    # 21 + 4 = 25... Add chest_acc_mag to get 26!
    
    return feature_cols


def get_26_channel_columns():
    """
    Define the 26 channels for feature extraction.
    Original 23 columns (excluding ECG) + 4 magnitudes + chest magnitude = 26
    """
    channels = [
        # Chest accelerometer (3)
        'chest_acc_x', 'chest_acc_y', 'chest_acc_z',
        # Left ankle accelerometer (3)
        'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z',
        # Left ankle gyroscope (3)
        'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z',
        # Left ankle magnetometer (3)
        'ankle_mag_x', 'ankle_mag_y', 'ankle_mag_z',
        # Right arm accelerometer (3)
        'arm_acc_x', 'arm_acc_y', 'arm_acc_z',
        # Right arm gyroscope (3)
        'arm_gyro_x', 'arm_gyro_y', 'arm_gyro_z',
        # Right arm magnetometer (3)
        'arm_mag_x', 'arm_mag_y', 'arm_mag_z',
        # Magnitudes (4) - ankle acc, ankle gyro, arm acc, arm gyro
        'ankle_acc_mag', 'ankle_gyro_mag', 'arm_acc_mag', 'arm_gyro_mag',
        # Chest accelerometer magnitude (1) - to reach 26
        'chest_acc_mag'
    ]
    
    # 3 + 3 + 3 + 3 + 3 + 3 + 3 + 4 + 1 = 26 ✓
    return channels


def add_all_magnitude_features(df):
    """Add all vector magnitude features including chest."""
    df = df.copy()
    
    # Chest accelerometer magnitude
    df['chest_acc_mag'] = compute_vector_magnitude(
        df, 'chest_acc_x', 'chest_acc_y', 'chest_acc_z'
    )
    
    # Ankle accelerometer magnitude
    df['ankle_acc_mag'] = compute_vector_magnitude(
        df, 'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z'
    )
    
    # Ankle gyroscope magnitude
    df['ankle_gyro_mag'] = compute_vector_magnitude(
        df, 'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z'
    )
    
    # Arm accelerometer magnitude
    df['arm_acc_mag'] = compute_vector_magnitude(
        df, 'arm_acc_x', 'arm_acc_y', 'arm_acc_z'
    )
    
    # Arm gyroscope magnitude
    df['arm_gyro_mag'] = compute_vector_magnitude(
        df, 'arm_gyro_x', 'arm_gyro_y', 'arm_gyro_z'
    )
    
    return df


# =============================================================================
# 4. PREPROCESSING - REMOVE NULL CLASS
# =============================================================================
def remove_null_class(df):
    """Remove samples with label 0 (Null/No activity)."""
    original_count = len(df)
    df_filtered = df[df['label'] != 0].copy()
    removed_count = original_count - len(df_filtered)
    print(f"  Removed {removed_count} samples with Null class (label=0)")
    return df_filtered


# =============================================================================
# 5. WINDOWING
# =============================================================================
def create_windows(data, feature_cols, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """
    Create sliding windows from continuous sensor data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data with sensor readings
    feature_cols : list
        List of column names to use as features
    window_size : int
        Number of samples per window
    step_size : int
        Step size between windows (overlap = window_size - step_size)
    
    Returns:
    --------
    windows : np.ndarray
        Shape: (n_windows, window_size, n_channels)
    labels : np.ndarray
        Shape: (n_windows,)
    """
    windows = []
    labels = []
    
    # Get unique subjects to window within each subject's data
    subjects = data['subject'].unique()
    
    for subject in subjects:
        subject_data = data[data['subject'] == subject]
        
        # Get unique labels to window within each activity
        unique_labels = subject_data['label'].unique()
        
        for label in unique_labels:
            label_data = subject_data[subject_data['label'] == label]
            
            # Get feature values
            X = label_data[feature_cols].values
            
            # Create windows
            for start in range(0, len(X) - window_size + 1, step_size):
                end = start + window_size
                window = X[start:end]
                
                if len(window) == window_size:
                    windows.append(window)
                    labels.append(label)
    
    return np.array(windows), np.array(labels)


# =============================================================================
# 6. FEATURE EXTRACTION
# =============================================================================
def compute_energy_fft(signal):
    """Compute energy from FFT of signal."""
    fft_vals = fft(signal)
    # Energy is sum of squared magnitudes (excluding DC component)
    energy = np.sum(np.abs(fft_vals[1:len(fft_vals)//2])**2)
    return energy


def extract_window_features(window):
    """
    Extract 7 statistical features from each channel in a window.
    
    Features per channel:
    1. Mean
    2. Standard Deviation
    3. Minimum
    4. Maximum
    5. Median
    6. Skewness
    7. Energy (FFT)
    
    Parameters:
    -----------
    window : np.ndarray
        Shape: (window_size, n_channels)
    
    Returns:
    --------
    features : np.ndarray
        Shape: (n_channels * 7,)
    """
    n_channels = window.shape[1]
    features = []
    
    for ch in range(n_channels):
        signal = window[:, ch]
        
        # 1. Mean
        feat_mean = np.mean(signal)
        
        # 2. Standard Deviation
        feat_std = np.std(signal)
        
        # 3. Minimum
        feat_min = np.min(signal)
        
        # 4. Maximum
        feat_max = np.max(signal)
        
        # 5. Median
        feat_median = np.median(signal)
        
        # 6. Skewness
        feat_skewness = stats.skew(signal)
        
        # 7. Energy (FFT)
        feat_energy = compute_energy_fft(signal)
        
        features.extend([
            feat_mean, feat_std, feat_min, feat_max, 
            feat_median, feat_skewness, feat_energy
        ])
    
    return np.array(features)


def extract_all_features(windows):
    """
    Extract features from all windows.
    
    Parameters:
    -----------
    windows : np.ndarray
        Shape: (n_windows, window_size, n_channels)
    
    Returns:
    --------
    features : np.ndarray
        Shape: (n_windows, n_channels * 7)
    """
    print(f"  Extracting features from {len(windows)} windows...")
    features = []
    
    for i, window in enumerate(windows):
        if (i + 1) % 1000 == 0:
            print(f"    Processed {i + 1}/{len(windows)} windows")
        
        window_features = extract_window_features(window)
        features.append(window_features)
    
    return np.array(features)


def get_feature_names(channel_names):
    """Generate feature names for all channels and statistics."""
    stat_names = ['mean', 'std', 'min', 'max', 'median', 'skewness', 'energy']
    feature_names = []
    
    for channel in channel_names:
        for stat in stat_names:
            feature_names.append(f"{channel}_{stat}")
    
    return feature_names


# =============================================================================
# 7. STANDARDIZATION
# =============================================================================
def standardize_features(X_train, X_val, save_path='scaler_params.json'):
    """
    Standardize features using StandardScaler and save parameters.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    X_val : np.ndarray
        Validation features
    save_path : str
        Path to save scaler parameters as JSON
    
    Returns:
    --------
    X_train_scaled : np.ndarray
        Scaled training features
    X_val_scaled : np.ndarray
        Scaled validation features
    scaler : StandardScaler
        Fitted scaler object
    """
    scaler = StandardScaler()
    
    # Fit on training data
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform validation data
    X_val_scaled = scaler.transform(X_val)
    
    # Save scaler parameters to JSON
    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'var': scaler.var_.tolist(),
        'n_features_in': int(scaler.n_features_in_),
        'n_samples_seen': int(scaler.n_samples_seen_)
    }
    
    with open(save_path, 'w') as f:
        json.dump(scaler_params, f, indent=2)
    
    print(f"[OK] Scaler parameters saved to '{save_path}'")
    
    return X_train_scaled, X_val_scaled, scaler


# =============================================================================
# 8. MODEL TRAINING
# =============================================================================
def train_random_forest(X_train, y_train):
    """
    Train a RandomForestClassifier with balanced class weights.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    
    Returns:
    --------
    model : RandomForestClassifier
        Trained model
    """
    print("Training Random Forest Classifier...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    print("[OK] Model training complete.")
    
    return model


# =============================================================================
# 9. EVALUATION
# =============================================================================
def evaluate_model(model, X_val, y_val, class_names=None):
    """Evaluate model and print metrics."""
    y_pred = model.predict(X_val)
    
    # Accuracy
    accuracy = accuracy_score(y_val, y_pred)
    print(f"[OK] Validation Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=class_names, zero_division=0))
    
    return y_pred


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """
    Generate and save confusion matrix visualization.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    class_names : list
        Names of classes
    save_path : str
        Path to save the plot
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=(14, 12))
    
    # Plot heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar_kws={'label': 'Proportion'}
    )
    
    plt.title('Confusion Matrix - HAR MHEALTH\n(Normalized by True Labels)', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Confusion matrix saved to '{save_path}'")
    
    plt.show()


# =============================================================================
# 10. ONNX EXPORT
# =============================================================================
def export_to_onnx(model, n_features, save_path='har_model.onnx'):
    """
    Export trained model to ONNX format.
    
    Parameters:
    -----------
    model : RandomForestClassifier
        Trained sklearn model
    n_features : int
        Number of input features (26 channels * 7 stats = 182)
    save_path : str
        Path to save ONNX model
    """
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        print("[ERROR] skl2onnx not installed. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'skl2onnx'])
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    
    # Define input type with correct dimensions
    # Input: (batch_size, n_features) where n_features = 26 * 7 = 182
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    
    # Convert to ONNX
    print(f"Exporting model to ONNX format...")
    print(f"  Input dimensions: (batch_size, {n_features})")
    
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=12,
        options={id(model): {'zipmap': False}}  # Return arrays instead of dict
    )
    
    # Save ONNX model
    with open(save_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    
    print(f"[OK] ONNX model saved to '{save_path}'")
    
    # Verify ONNX model
    try:
        import onnx
        onnx_model_loaded = onnx.load(save_path)
        onnx.checker.check_model(onnx_model_loaded)
        print("[OK] ONNX model validation passed.")
        
        # Print model info
        print(f"\nONNX Model Info:")
        print(f"  IR Version: {onnx_model_loaded.ir_version}")
        print(f"  Opset Version: {onnx_model_loaded.opset_import[0].version}")
        for input_tensor in onnx_model_loaded.graph.input:
            print(f"  Input: {input_tensor.name}, Shape: {[d.dim_value if d.dim_value else 'batch' for d in input_tensor.type.tensor_type.shape.dim]}")
        for output_tensor in onnx_model_loaded.graph.output:
            print(f"  Output: {output_tensor.name}")
            
    except ImportError:
        print("  (Install 'onnx' package for model validation)")
    
    return onnx_model


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    """Main pipeline execution."""
    print("=" * 70)
    print("  HUMAN ACTIVITY RECOGNITION (HAR) PIPELINE - MHEALTH DATASET")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Step 1: Download and Extract Dataset
    # -------------------------------------------------------------------------
    print("\n[1/8] DOWNLOADING DATASET")
    print("-" * 40)
    if not download_and_extract_dataset():
        print("Failed to download dataset. Exiting.")
        return
    
    # -------------------------------------------------------------------------
    # Step 2: Load Data
    # -------------------------------------------------------------------------
    print("\n[2/8] LOADING DATA")
    print("-" * 40)
    print("Loading training subjects (1-6)...")
    train_data = load_subjects(TRAIN_SUBJECTS)
    print(f"[OK] Training data shape: {train_data.shape}")
    
    print("\nLoading validation subjects (7-8)...")
    val_data = load_subjects(VAL_SUBJECTS)
    print(f"[OK] Validation data shape: {val_data.shape}")
    
    # -------------------------------------------------------------------------
    # Step 3: Remove Null Class
    # -------------------------------------------------------------------------
    print("\n[3/8] REMOVING NULL CLASS")
    print("-" * 40)
    train_data = remove_null_class(train_data)
    val_data = remove_null_class(val_data)
    
    print(f"  Training samples after filtering: {len(train_data)}")
    print(f"  Validation samples after filtering: {len(val_data)}")
    
    # -------------------------------------------------------------------------
    # Step 4: Feature Engineering - Add Magnitudes
    # -------------------------------------------------------------------------
    print("\n[4/8] FEATURE ENGINEERING - ADDING MAGNITUDES")
    print("-" * 40)
    train_data = add_all_magnitude_features(train_data)
    val_data = add_all_magnitude_features(val_data)
    
    # Get 26 channel names
    channel_names = get_26_channel_columns()
    print(f"[OK] Expanded to {len(channel_names)} channels:")
    for i, ch in enumerate(channel_names, 1):
        print(f"    {i:2d}. {ch}")
    
    # -------------------------------------------------------------------------
    # Step 5: Create Windows
    # -------------------------------------------------------------------------
    print("\n[5/8] CREATING WINDOWS")
    print("-" * 40)
    print(f"  Window size: {WINDOW_SIZE} samples")
    print(f"  Overlap: {OVERLAP * 100:.0f}%")
    print(f"  Step size: {STEP_SIZE} samples")
    
    print("\nCreating training windows...")
    train_windows, train_labels = create_windows(train_data, channel_names)
    print(f"[OK] Training windows shape: {train_windows.shape}")
    
    print("\nCreating validation windows...")
    val_windows, val_labels = create_windows(val_data, channel_names)
    print(f"[OK] Validation windows shape: {val_windows.shape}")
    
    # -------------------------------------------------------------------------
    # Step 6: Extract Features
    # -------------------------------------------------------------------------
    print("\n[6/8] EXTRACTING FEATURES")
    print("-" * 40)
    print("Extracting 7 statistics per channel (26 x 7 = 182 features)...")
    
    X_train = extract_all_features(train_windows)
    X_val = extract_all_features(val_windows)
    
    print(f"[OK] Training features shape: {X_train.shape}")
    print(f"[OK] Validation features shape: {X_val.shape}")
    
    # Get feature names
    feature_names = get_feature_names(channel_names)
    print(f"[OK] Total features: {len(feature_names)}")
    
    # Handle any NaN or Inf values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    
    # -------------------------------------------------------------------------
    # Step 7: Standardization
    # -------------------------------------------------------------------------
    print("\n[7/8] STANDARDIZING FEATURES")
    print("-" * 40)
    X_train_scaled, X_val_scaled, scaler = standardize_features(
        X_train, X_val, save_path='scaler_params.json'
    )
    
    # -------------------------------------------------------------------------
    # Step 8: Train Model
    # -------------------------------------------------------------------------
    print("\n[8/8] TRAINING MODEL")
    print("-" * 40)
    model = train_random_forest(X_train_scaled, train_labels)
    
    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)
    
    # Get class names for labels present in the data
    unique_labels = sorted(np.unique(np.concatenate([train_labels, val_labels])))
    class_names = [ACTIVITY_LABELS.get(l, f"Class_{l}") for l in unique_labels]
    
    # Evaluate on validation set
    y_pred = evaluate_model(model, X_val_scaled, val_labels, class_names)
    
    # -------------------------------------------------------------------------
    # Generate Confusion Matrix
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  GENERATING CONFUSION MATRIX")
    print("=" * 70)
    
    # Map labels to indices for confusion matrix
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    y_true_idx = np.array([label_to_idx[l] for l in val_labels])
    y_pred_idx = np.array([label_to_idx[l] for l in y_pred])
    
    plot_confusion_matrix(y_true_idx, y_pred_idx, class_names)
    
    # -------------------------------------------------------------------------
    # Export to ONNX
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  EXPORTING MODEL TO ONNX")
    print("=" * 70)
    
    n_features = X_train_scaled.shape[1]  # Should be 182 (26 * 7)
    print(f"Number of features: {n_features}")
    
    export_to_onnx(model, n_features, save_path='har_mhealth_model.onnx')
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"""
    Dataset:          MHEALTH
    Train Subjects:   {TRAIN_SUBJECTS}
    Val Subjects:     {VAL_SUBJECTS}
    
    Preprocessing:
      - Removed Null class (label 0)
      - Added 5 magnitude features
      - Final channels: 26
    
    Windowing:
      - Window size: {WINDOW_SIZE} samples
      - Overlap: {OVERLAP * 100:.0f}%
      - Training windows: {len(train_windows)}
      - Validation windows: {len(val_windows)}
    
    Feature Extraction:
      - Statistics per channel: 7 (Mean, Std, Min, Max, Median, Skewness, Energy)
      - Total features: {n_features}
    
    Model:
      - Type: RandomForestClassifier
      - Class weights: Balanced
      - Estimators: 100
    
    Outputs:
      - Scaler parameters: scaler_params.json
      - Confusion matrix: confusion_matrix.png
      - ONNX model: har_mhealth_model.onnx
    """)
    
    return model, scaler


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    model, scaler = main()
