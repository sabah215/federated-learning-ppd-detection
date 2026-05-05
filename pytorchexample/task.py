"""
task.py: Core logic for Data Loading, Preprocessing, Federated Training, and Differential Privacy.
"""

import os
import logging
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_sample_weight

# --- Configuration ---
DATA_FILENAME = "PPD_analysis.csv"
TARGET_COLUMN = "EPDS Result"
DROP_COLUMNS = ["sr", "PHQ9 Score", "PHQ9 Result", "EPDS Score"]

USE_SHAP_SELECTION = True
TOP_K_FEATURES = 30
NUM_SELECTION_CLIENTS = 4

# --- DIFFERENTIAL PRIVACY CONFIG ---
# Set to True to enable Privacy
ENABLE_DIFFERENTIAL_PRIVACY = True 
# Epsilon (ε): Lower = More Privacy/More Noise. Higher = Less Privacy/Less Noise.
# Recommended Tests: 0.1 (Strict), 1.0 (Standard), 10.0 (Loose)
DP_EPSILON = 50.0  # Increased from 5.0 (Less noise)
# Sensitivity: The clipping bound for features (StandardScaled data usually falls in -3 to 3)
DP_SENSITIVITY = 1.5 # Lowered from 5.0 (Tighter clipping bound)


XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,              # DOWN from 6: Stops the trees from getting too deep
    "eta": 0.055,                 # DOWN from 0.1: Slows down the learning process
    "subsample": 0.8,            
    "colsample_bytree": 0.8,     
    "min_child_weight": 3,       # UP from 1: Puts a mild restriction back on the leaves
    "device": "cpu",
    "tree_method": "hist"
}
NUM_BOOST_ROUND = 5  

_CACHED_DATA: Dict[str, Any] = {}
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def apply_differential_privacy(X: np.ndarray, epsilon: float, sensitivity: float = 5.0) -> np.ndarray:
    """
    Applies Local Differential Privacy (LDP) via Input Perturbation (Laplace Mechanism).
    
    Theory:
    X_priv = Clip(X, -S, S) + Laplace(0, S/epsilon)
    """
    logger.info(f"🛡️ Applying Differential Privacy (Epsilon={epsilon}, Sensitivity={sensitivity})")
    
    # 1. Clipping: Bound the influence of any single data point
    # We clip data to range [-sensitivity, sensitivity]
    X_clipped = np.clip(X, -sensitivity, sensitivity)
    
    # 2. Noise Scale Calculation (Laplace Mechanism)
    # scale = sensitivity / epsilon
    scale = sensitivity / epsilon
    
    # 3. Generate Noise
    noise = np.random.laplace(loc=0.0, scale=scale, size=X_clipped.shape)
    
    # 4. Add Noise
    X_noisy = X_clipped + noise
    
    return X_noisy


def get_federated_feature_selection(X: np.ndarray, y: np.ndarray, num_clients: int = 4, top_k: int = 30) -> np.ndarray:
    """Simulates Federated Feature Selection using SHAP."""
    logger.info("--- Starting Federated Feature Selection (SHAP Methodology) ---")
    n_samples, n_features = X.shape
    part_size = n_samples // num_clients
    global_shap_importance = np.zeros(n_features)
    selection_config = XGB_PARAMS.copy()
    selection_config.update({"n_estimators": 10, "max_depth": 3})

    for i in range(num_clients):
        start = i * part_size
        end = (i + 1) * part_size if i < num_clients - 1 else n_samples
        X_local = X[start:end]
        y_local = y[start:end]

        model = xgb.XGBClassifier(**selection_config)
        model.fit(X_local, y_local)

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_local)
            if isinstance(shap_values, list):
                mean_shap = np.mean(np.abs(shap_values[0]), axis=0)
            else:
                mean_shap = np.mean(np.abs(shap_values), axis=0)
                if len(mean_shap.shape) > 1:
                    mean_shap = np.mean(mean_shap, axis=-1)
            global_shap_importance += mean_shap
        except Exception as e:
            logger.warning(f"Client {i + 1} SHAP calculation failed: {e}")

    top_indices = np.argsort(global_shap_importance)[::-1][:top_k]
    logger.info(f"--- Global Selection Complete: Kept Top {top_k} Features ---")
    return top_indices

def _load_raw_data() -> Tuple[pd.DataFrame, np.ndarray]:
    """Helper: Loads CSV and maps target."""
    current_dir = Path(__file__).parent.absolute()
    file_path = current_dir / DATA_FILENAME
    if not file_path.exists():
        file_path = Path(DATA_FILENAME)
        if not file_path.exists():
            raise FileNotFoundError(f"Could not find {DATA_FILENAME}")

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    cols_to_drop = [c for c in DROP_COLUMNS + [TARGET_COLUMN] if c in df.columns]
    X_raw = df.drop(columns=cols_to_drop)
    y_raw = df[TARGET_COLUMN].astype(str).str.lower()
    y = y_raw.apply(lambda x: 0 if 'low' in x else 1).values
    return X_raw, y

def _preprocess_data(X_raw: pd.DataFrame, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Helper: OneHot, Scaling, Selection."""
    cat_cols = X_raw.select_dtypes(include=['object']).columns.tolist()
    num_cols = X_raw.select_dtypes(exclude=['object']).columns.tolist()
    for col in cat_cols: X_raw[col] = X_raw[col].astype(str)

    ct = ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ("scaler", StandardScaler(), num_cols)
    ])
    
    X_processed = ct.fit_transform(X_raw)
    feature_names = ct.get_feature_names_out()

    if USE_SHAP_SELECTION:
        selected_indices = get_federated_feature_selection(X_processed, y, NUM_SELECTION_CLIENTS, TOP_K_FEATURES)
        X_final = X_processed[:, selected_indices]
        final_feature_names = feature_names[selected_indices]
    else:
        X_final = X_processed
        final_feature_names = feature_names

    _CACHED_DATA['feature_names'] = final_feature_names
    return X_final, y

def load_ppd_data(partition_id: int, num_partitions: int) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    global _CACHED_DATA
    if 'X' not in _CACHED_DATA:
        X_raw, y_binary = _load_raw_data()
        X_final, y_final = _preprocess_data(X_raw, y_binary)
        _CACHED_DATA['X'] = X_final
        _CACHED_DATA['y'] = y_final

    X_global = _CACHED_DATA['X']
    y_global = _CACHED_DATA['y']
    
    total_samples = len(X_global)
    indices = np.arange(total_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    part_size = total_samples // num_partitions
    start = partition_id * part_size
    end = start + part_size
    X_local = X_global[indices][start:end]
    y_local = y_global[indices][start:end]

    X_train, X_test, y_train, y_test = train_test_split(
        X_local, y_local, test_size=0.2, random_state=42, stratify=y_local
    )
    return (X_train, y_train), (X_test, y_test)

def get_feature_names() -> Optional[List[str]]:
    return _CACHED_DATA.get('feature_names')

def train_xgb(X_train: np.ndarray, y_train: np.ndarray, params: Optional[Dict] = None) -> xgb.Booster:
    """
    Trains a local XGBoost model.
    Applies Differential Privacy to X_train if enabled.
    """
    
    # --- 🛡️ APPLY DIFFERENTIAL PRIVACY HERE ---
    if ENABLE_DIFFERENTIAL_PRIVACY:
        # We only add noise to Training data. Testing data must remain clean for valid evaluation.
        X_train_to_use = apply_differential_privacy(
            X_train, 
            epsilon=DP_EPSILON, 
            sensitivity=DP_SENSITIVITY
        )
    else:
        X_train_to_use = X_train

    # Handle Class Imbalance
    weights = compute_sample_weight(class_weight='balanced', y=y_train)
    dtrain = xgb.DMatrix(X_train_to_use, label=y_train, weight=weights)

    train_config = XGB_PARAMS.copy()
    if params: train_config.update(params)

    bst = xgb.train(train_config, dtrain, num_boost_round=NUM_BOOST_ROUND)
    return bst