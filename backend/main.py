
import sys
import os
import json
import io
import numpy as np
import pandas as pd
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from har_utils import (
    extract_window_features, standardize_features, 
    load_subject_data, preprocess_data, create_windows,
    create_sequential_windows,
    ACTIVITY_LABELS, CHANNEL_NAMES, ORIGINAL_COLUMNS
)

app = FastAPI(title="HAR MHEALTH API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
model_session = None
scaler_params = None
simulation_data = {
    "windows": None,
    "labels": None,
    "index": 0
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "har_mhealth_model.onnx")
SCALER_PATH = os.path.join(BASE_DIR, "scaler_params.json")
DATA_PATH = os.path.join(BASE_DIR, "MHEALTHDATASET", "mHealth_subject10.log")

@app.on_event("startup")
async def startup_event():
    global model_session, scaler_params, simulation_data
    
    # Load Model
    if os.path.exists(MODEL_PATH):
        try:
            model_session = ort.InferenceSession(MODEL_PATH)
            print(f"Loaded ONNX model from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")

    # Load Scaler
    if os.path.exists(SCALER_PATH):
        try:
            with open(SCALER_PATH, 'r') as f:
                scaler_params = json.load(f)
            print(f"Loaded Scaler params from {SCALER_PATH}")
        except Exception as e:
             print(f"Error loading scaler: {e}")
    else:
        print(f"Warning: Scaler not found at {SCALER_PATH}")

    # Load Simulation Data
    if os.path.exists(DATA_PATH):
        try:
            print(f"Loading simulation data from {DATA_PATH}...")
            raw_data = load_subject_data(DATA_PATH)
            processed_data = preprocess_data(raw_data)
            windows, labels = create_windows(processed_data)
            
            simulation_data["windows"] = windows
            simulation_data["labels"] = labels
            print(f"Loaded {len(windows)} windows for simulation.")
        except Exception as e:
             print(f"Error loading data: {e}")
    else:
        print(f"Warning: Data not found at {DATA_PATH}")

@app.post("/predict/log")
async def predict_log_file(file: UploadFile = File(...)):
    """
    Upload a log file, process it, and return a sequence of predictions for timeline visualization.
    """
    try:
        content = await file.read()
        # Assume tab-separated values like original MHEALTH dataset
        df = pd.read_csv(io.BytesIO(content), sep='\t', header=None, names=ORIGINAL_COLUMNS)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")
    
    # Preprocess (remove nulls, add magnitudes)
    # Note: This might create time gaps if nulls are removed, but ensures valid predictions
    df_processed = preprocess_data(df)
    
    if len(df_processed) < 100:
        raise HTTPException(status_code=400, detail="Not enough valid data in file (min 100 samples)")
        
    # Create sequential windows
    # step_size=50 means 50% overlap (100 window size) -> 1s resolution at 50Hz (approx)
    windows, labels, indices = create_sequential_windows(df_processed, step_size=50)
    
    timeline = []
    
    for i in range(len(windows)):
        window = windows[i]
        true_lbl = int(labels[i])
        
        # Feature extraction
        features = extract_window_features(window)
        features = np.nan_to_num(features, nan=0.0)
        
        # Standardize
        if scaler_params:
            try:
                features_scaled = standardize_features(features, scaler_params)
            except:
                features_scaled = features
        else:
            features_scaled = features
            
        # Predict
        pred_lbl = -1
        if model_session:
            try:
                input_name = model_session.get_inputs()[0].name
                input_data = features_scaled.reshape(1, -1).astype(np.float32)
                outputs = model_session.run(None, {input_name: input_data})
                pred_lbl = int(outputs[0][0])
            except Exception as e:
                print(f"Prediction error: {e}")
        
        timeline.append({
            "segment_id": i,
            "start_time_idx": int(indices[i]),
            "end_time_idx": int(indices[i] + 100),
            "true_label": ACTIVITY_LABELS.get(true_lbl, "Unknown"),
            "true_id": true_lbl,
            "predicted_label": ACTIVITY_LABELS.get(pred_lbl, "Unknown"),
            "predicted_id": pred_lbl,
            "features": features_scaled.tolist() # Send standardized features for frontend inspection
        })
        
    return {"filename": file.filename, "timeline": timeline}

@app.get("/")
def read_root():
    return {"status": "ok", "service": "HAR Prediction API"}

@app.get("/simulation/next")
def get_next_simulation_window():
    if simulation_data["windows"] is None:
        raise HTTPException(status_code=404, detail="Simulation data not loaded")
    
    idx = simulation_data["index"]
    total = len(simulation_data["windows"])
    
    if idx >= total:
        simulation_data["index"] = 0 # Loop back
        idx = 0
    
    window = simulation_data["windows"][idx]
    true_label = int(simulation_data["labels"][idx])
    
    # Predict
    features = extract_window_features(window)
    # Handle NaN
    features = np.nan_to_num(features, nan=0.0)
    
    # Standardize
    if scaler_params:
        features_scaled = standardize_features(features, scaler_params)
    else:
        features_scaled = features # Fallback (will likely fail model)
    
    # Reshape for ONNX (1, 182)
    if model_session:
        input_name = model_session.get_inputs()[0].name
        input_data = features_scaled.reshape(1, -1).astype(np.float32)
        
        outputs = model_session.run(None, {input_name: input_data})
        pred_label = int(outputs[0][0])
        # Probabilities often in outputs[1] which is a list of maps for sklearn-onnx
        # But for ClassLabel output it might be different. 
        # Usually outputs[1] is a list of dictionaries {label: prob}
        
        probabilities = outputs[1][0] if len(outputs) > 1 else {}
    else:
        pred_label = -1
        probabilities = {}
    
    # Increment
    simulation_data["index"] += 1
    
    # Prepare response data (raw sensor data for visualization)
    # Window shape (100, 26). 
    # CHANNEL_NAMES indices:
    # 0-2: chest_acc
    # 3-5: ankle_acc
    # 6-8: ankle_gyro (wait, no. let's check CHANNEL_NAMES in har_utils)
    # CHANNEL_NAMES = [chest_acc_x... (0,1,2), ankle_acc_x... (3,4,5)]
    
    sensor_data = {
        "ankle_acc_x": window[:, 3].tolist(),
        "ankle_acc_y": window[:, 4].tolist(),
        "ankle_acc_z": window[:, 5].tolist(),
    }
    
    return {
        "index": idx,
        "true_label": ACTIVITY_LABELS.get(true_label, "Unknown"),
        "predicted_label": ACTIVITY_LABELS.get(pred_label, "Unknown"),
        "is_correct": true_label == pred_label,
        "probabilities": probabilities,
        "sensor_data": sensor_data
    }
