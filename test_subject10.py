"""
Test HAR Model on Subject 10
============================
Script to evaluate the trained ONNX model on unseen subject (Subject 10).

This script:
1. Loads the trained ONNX model
2. Loads scaler parameters from JSON
3. Processes Subject 10 data with the same pipeline
4. Evaluates and visualizes results
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION (Must match training pipeline)
# =============================================================================
DATA_DIR = "MHEALTHDATASET"
MODEL_PATH = "har_mhealth_model.onnx"
SCALER_PATH = "scaler_params.json"

WINDOW_SIZE = 100
OVERLAP = 0.5
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))

TEST_SUBJECT = 10

# Original column names from MHEALTH dataset
ORIGINAL_COLUMNS = [
    'chest_acc_x', 'chest_acc_y', 'chest_acc_z',
    'ecg_1', 'ecg_2',
    'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z',
    'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z',
    'ankle_mag_x', 'ankle_mag_y', 'ankle_mag_z',
    'arm_acc_x', 'arm_acc_y', 'arm_acc_z',
    'arm_gyro_x', 'arm_gyro_y', 'arm_gyro_z',
    'arm_mag_x', 'arm_mag_y', 'arm_mag_z',
    'label'
]

# 26 channel names (must match training)
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
# LOAD FUNCTIONS
# =============================================================================
def load_onnx_model(model_path):
    """Load ONNX model for inference."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("Installing onnxruntime...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'onnxruntime'])
        import onnxruntime as ort
    
    print(f"Loading ONNX model from '{model_path}'...")
    session = ort.InferenceSession(model_path)
    
    # Print model info
    print(f"[OK] Model loaded successfully")
    print(f"  Input name: {session.get_inputs()[0].name}")
    print(f"  Input shape: {session.get_inputs()[0].shape}")
    print(f"  Output names: {[o.name for o in session.get_outputs()]}")
    
    return session


def load_scaler_params(scaler_path):
    """Load StandardScaler parameters from JSON."""
    print(f"Loading scaler parameters from '{scaler_path}'...")
    
    with open(scaler_path, 'r') as f:
        params = json.load(f)
    
    print(f"[OK] Scaler parameters loaded")
    print(f"  Number of features: {params['n_features_in']}")
    
    return params


def load_subject_data(subject_id):
    """Load data for a specific subject."""
    file_path = os.path.join(DATA_DIR, f"mHealth_subject{subject_id}.log")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Subject file not found: {file_path}")
    
    data = pd.read_csv(file_path, sep='\t', header=None, names=ORIGINAL_COLUMNS)
    data['subject'] = subject_id
    
    return data


# =============================================================================
# PREPROCESSING FUNCTIONS (Same as training)
# =============================================================================
def compute_vector_magnitude(df, x_col, y_col, z_col):
    """Compute vector magnitude from x, y, z components."""
    return np.sqrt(df[x_col]**2 + df[y_col]**2 + df[z_col]**2)


def add_all_magnitude_features(df):
    """Add all vector magnitude features."""
    df = df.copy()
    
    df['chest_acc_mag'] = compute_vector_magnitude(
        df, 'chest_acc_x', 'chest_acc_y', 'chest_acc_z'
    )
    df['ankle_acc_mag'] = compute_vector_magnitude(
        df, 'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z'
    )
    df['ankle_gyro_mag'] = compute_vector_magnitude(
        df, 'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z'
    )
    df['arm_acc_mag'] = compute_vector_magnitude(
        df, 'arm_acc_x', 'arm_acc_y', 'arm_acc_z'
    )
    df['arm_gyro_mag'] = compute_vector_magnitude(
        df, 'arm_gyro_x', 'arm_gyro_y', 'arm_gyro_z'
    )
    
    return df


def remove_null_class(df):
    """Remove samples with label 0 (Null/No activity)."""
    original_count = len(df)
    df_filtered = df[df['label'] != 0].copy()
    removed_count = original_count - len(df_filtered)
    print(f"  Removed {removed_count} samples with Null class (label=0)")
    return df_filtered


def create_windows(data, feature_cols, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """Create sliding windows from continuous sensor data."""
    windows = []
    labels = []
    
    subjects = data['subject'].unique()
    
    for subject in subjects:
        subject_data = data[data['subject'] == subject]
        unique_labels = subject_data['label'].unique()
        
        for label in unique_labels:
            label_data = subject_data[subject_data['label'] == label]
            X = label_data[feature_cols].values
            
            for start in range(0, len(X) - window_size + 1, step_size):
                end = start + window_size
                window = X[start:end]
                
                if len(window) == window_size:
                    windows.append(window)
                    labels.append(label)
    
    return np.array(windows), np.array(labels)


# =============================================================================
# FEATURE EXTRACTION (Same as training)
# =============================================================================
def compute_energy_fft(signal):
    """Compute energy from FFT of signal."""
    fft_vals = fft(signal)
    energy = np.sum(np.abs(fft_vals[1:len(fft_vals)//2])**2)
    return energy


def extract_window_features(window):
    """Extract 7 statistical features from each channel."""
    n_channels = window.shape[1]
    features = []
    
    for ch in range(n_channels):
        signal = window[:, ch]
        
        feat_mean = np.mean(signal)
        feat_std = np.std(signal)
        feat_min = np.min(signal)
        feat_max = np.max(signal)
        feat_median = np.median(signal)
        feat_skewness = stats.skew(signal)
        feat_energy = compute_energy_fft(signal)
        
        features.extend([
            feat_mean, feat_std, feat_min, feat_max,
            feat_median, feat_skewness, feat_energy
        ])
    
    return np.array(features)


def extract_all_features(windows):
    """Extract features from all windows."""
    print(f"  Extracting features from {len(windows)} windows...")
    features = []
    
    for i, window in enumerate(windows):
        if (i + 1) % 500 == 0:
            print(f"    Processed {i + 1}/{len(windows)} windows")
        window_features = extract_window_features(window)
        features.append(window_features)
    
    return np.array(features)


# =============================================================================
# STANDARDIZATION WITH LOADED PARAMS
# =============================================================================
def standardize_with_params(X, scaler_params):
    """Apply standardization using saved parameters."""
    mean = np.array(scaler_params['mean'])
    scale = np.array(scaler_params['scale'])
    
    X_scaled = (X - mean) / scale
    return X_scaled


# =============================================================================
# ONNX INFERENCE
# =============================================================================
def predict_with_onnx(session, X):
    """Run inference with ONNX model."""
    input_name = session.get_inputs()[0].name
    
    # Ensure correct dtype
    X = X.astype(np.float32)
    
    # Run inference
    outputs = session.run(None, {input_name: X})
    
    # First output is predictions (labels)
    predictions = outputs[0]
    
    return predictions


# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix_subject10.png'):
    """Generate and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(14, 12))
    
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Greens',
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar_kws={'label': 'Proportion'}
    )
    
    plt.title(f'Confusion Matrix - Subject {TEST_SUBJECT} Test\n(Normalized by True Labels)', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Confusion matrix saved to '{save_path}'")
    plt.show()


def plot_prediction_distribution(y_true, y_pred, class_names):
    """Plot distribution of predictions vs true labels."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # True label distribution
    unique_true, counts_true = np.unique(y_true, return_counts=True)
    axes[0].bar(range(len(unique_true)), counts_true, color='steelblue', alpha=0.7)
    axes[0].set_xticks(range(len(unique_true)))
    axes[0].set_xticklabels([class_names[i] for i in range(len(unique_true))], rotation=45, ha='right')
    axes[0].set_title('True Label Distribution')
    axes[0].set_ylabel('Count')
    
    # Predicted label distribution
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    axes[1].bar(range(len(unique_pred)), counts_pred, color='forestgreen', alpha=0.7)
    axes[1].set_xticks(range(len(unique_pred)))
    axes[1].set_xticklabels([class_names[i] for i in range(len(unique_pred))], rotation=45, ha='right')
    axes[1].set_title('Predicted Label Distribution')
    axes[1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('prediction_distribution_subject10.png', dpi=150, bbox_inches='tight')
    print("[OK] Prediction distribution saved to 'prediction_distribution_subject10.png'")
    plt.show()


# =============================================================================
# MAIN TEST PIPELINE
# =============================================================================
def main():
    """Main test pipeline for Subject 10."""
    print("=" * 70)
    print(f"  HAR MODEL TEST - SUBJECT {TEST_SUBJECT}")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Step 1: Check Required Files
    # -------------------------------------------------------------------------
    print("\n[1/6] CHECKING REQUIRED FILES")
    print("-" * 40)
    
    required_files = [MODEL_PATH, SCALER_PATH, DATA_DIR]
    for f in required_files:
        if os.path.exists(f):
            print(f"  [OK] Found: {f}")
        else:
            print(f"  [MISSING] Missing: {f}")
            print("\nPlease run the training pipeline first (har_mhealth_pipeline.py)")
            return
    
    # -------------------------------------------------------------------------
    # Step 2: Load Model and Scaler
    # -------------------------------------------------------------------------
    print("\n[2/6] LOADING MODEL AND SCALER")
    print("-" * 40)
    
    onnx_session = load_onnx_model(MODEL_PATH)
    scaler_params = load_scaler_params(SCALER_PATH)
    
    # -------------------------------------------------------------------------
    # Step 3: Load and Preprocess Subject 10 Data
    # -------------------------------------------------------------------------
    print(f"\n[3/6] LOADING SUBJECT {TEST_SUBJECT} DATA")
    print("-" * 40)
    
    test_data = load_subject_data(TEST_SUBJECT)
    print(f"[OK] Loaded {len(test_data)} samples")
    
    # Remove null class
    test_data = remove_null_class(test_data)
    
    # Add magnitude features
    test_data = add_all_magnitude_features(test_data)
    print(f"[OK] Added magnitude features")
    
    # -------------------------------------------------------------------------
    # Step 4: Create Windows and Extract Features
    # -------------------------------------------------------------------------
    print(f"\n[4/6] CREATING WINDOWS")
    print("-" * 40)
    
    test_windows, test_labels = create_windows(test_data, CHANNEL_NAMES)
    print(f"[OK] Created {len(test_windows)} windows")
    
    print(f"\n[5/6] EXTRACTING FEATURES")
    print("-" * 40)
    
    X_test = extract_all_features(test_windows)
    print(f"[OK] Feature shape: {X_test.shape}")
    
    # Handle NaN/Inf values
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Standardize using saved parameters
    X_test_scaled = standardize_with_params(X_test, scaler_params)
    print("[OK] Features standardized with training parameters")
    
    # -------------------------------------------------------------------------
    # Step 5: Run Inference
    # -------------------------------------------------------------------------
    print(f"\n[6/6] RUNNING INFERENCE")
    print("-" * 40)
    
    y_pred = predict_with_onnx(onnx_session, X_test_scaled)
    print(f"[OK] Predictions complete: {len(y_pred)} windows")
    
    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)
    
    # Get class names
    unique_labels = sorted(np.unique(test_labels))
    class_names = [ACTIVITY_LABELS.get(l, f"Class_{l}") for l in unique_labels]
    
    # Accuracy
    accuracy = accuracy_score(test_labels, y_pred)
    print(f"\n[OK] Test Accuracy (Subject {TEST_SUBJECT}): {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\nClassification Report:")
    print("-" * 60)
    print(classification_report(test_labels, y_pred, target_names=class_names, zero_division=0))
    
    # -------------------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    # Map labels to indices
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    y_true_idx = np.array([label_to_idx[l] for l in test_labels])
    y_pred_idx = np.array([label_to_idx[l] for l in y_pred])
    
    # Confusion matrix
    plot_confusion_matrix(y_true_idx, y_pred_idx, class_names)
    
    # Prediction distribution
    plot_prediction_distribution(y_true_idx, y_pred_idx, class_names)
    
    # -------------------------------------------------------------------------
    # Per-Activity Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PER-ACTIVITY ACCURACY")
    print("=" * 70)
    
    print(f"\n{'Activity':<30} {'Correct':>10} {'Total':>10} {'Accuracy':>12}")
    print("-" * 62)
    
    for label in unique_labels:
        mask = test_labels == label
        correct = np.sum(y_pred[mask] == label)
        total = np.sum(mask)
        acc = correct / total if total > 0 else 0
        activity_name = ACTIVITY_LABELS.get(label, f"Class_{label}")
        print(f"{activity_name:<30} {correct:>10} {total:>10} {acc:>11.2%}")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  TEST COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"""
    Test Subject:     {TEST_SUBJECT}
    Total Windows:    {len(test_windows)}
    Total Features:   {X_test.shape[1]}
    
    Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
    
    Output Files:
      - confusion_matrix_subject10.png
      - prediction_distribution_subject10.png
    """)
    
    return y_pred, test_labels, accuracy


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    predictions, true_labels, accuracy = main()
