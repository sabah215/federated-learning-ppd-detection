import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_sample_weight
import shap

# Global cache
_CACHED_DATA = {}

# Toggle: set to False initially to debug FL without feature selection
USE_SHAP_FEATURE_SELECTION = False
TOP_K_FEATURES = 10


def get_federated_feature_selection(X, y, num_clients: int = 4, top_k: int = 20):
    """Federated SHAP feature selection (optional; can be turned off)."""
    print("\n--- Starting Federated Feature Selection (SHAP Methodology) ---")
    n_samples = len(X)
    part_size = n_samples // num_clients

    n_features = X.shape[1]
    global_shap_importance = np.zeros(n_features)

    for i in range(num_clients):
        start = i * part_size
        end = (i + 1) * part_size if i < num_clients - 1 else n_samples
        X_local = X[start:end]
        y_local = y[start:end]

        model = xgb.XGBClassifier(
            n_estimators=10, max_depth=3, eval_metric="logloss",
            objective="binary:logistic", # <--- Ensure this matches
            n_jobs=1, device="cpu", tree_method="hist"  
        )
        model.fit(X_local, y_local)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_local)

        if isinstance(shap_values, list):
            class_means = [np.mean(np.abs(sv), axis=0) for sv in shap_values]
            mean_shap = np.mean(class_means, axis=0)
        else:
            mean_shap = np.mean(np.abs(shap_values), axis=0)
            if len(mean_shap.shape) > 1:
                mean_shap = np.mean(mean_shap, axis=-1)

        global_shap_importance += mean_shap
        print(f"Client {i + 1}: Calculated SHAP feature importance.")

    top_indices = np.argsort(global_shap_importance)[::-1][:top_k]
    print(f"--- Global Selection Complete: Kept Top {top_k} Features ---\n")
    return top_indices

# --- NEW HELPER FUNCTION ---
def get_feature_names():
    """Returns the list of feature names corresponding to the processed data."""
    if 'feature_names' in _CACHED_DATA:
        return _CACHED_DATA['feature_names']
    return None

def load_ppd_data(partition_id, num_partitions):
    global _CACHED_DATA
    if not _CACHED_DATA:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, "PPD_analysis.csv")
        if not os.path.exists(path): path = "PPD_analysis.csv"
        
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        target = 'EPDS Result'
        
        # Drop useless columns
        cols_to_drop = [target, 'sr', 'PHQ9 Score', 'PHQ9 Result', 'EPDS Score']
        drop_cols = [c for c in cols_to_drop if c in df.columns]
        X_raw = df.drop(columns=drop_cols)
        
        # --- CRITICAL FIX: BINARY MAPPING ---
        # Map "Low risk" to 0 (Healthy)
        # Map "Medium risk" and "High risk" to 1 (At Risk)
        y_raw = df[target].astype(str).str.lower()
        y = y_raw.apply(lambda x: 0 if 'low' in x else 1).values
        
        # Preprocessing
        cat_cols = X_raw.select_dtypes(include=['object']).columns.tolist()
        for col in cat_cols: X_raw[col] = X_raw[col].astype(str)
        num_cols = X_raw.select_dtypes(exclude=['object']).columns.tolist()
        
        ct = ColumnTransformer([
            ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
            ("scaler", StandardScaler(), num_cols)
        ])
        X_processed = ct.fit_transform(X_raw)
        
        # --- CAPTURE FEATURE NAMES ---
        all_feature_names = ct.get_feature_names_out()

        selected_indices = get_federated_feature_selection(X_processed, y, num_clients=4, top_k=30)
        X_final = X_processed[:, selected_indices]

        # Filter names using the same indices
        final_feature_names = all_feature_names[selected_indices]
        
        _CACHED_DATA['X'] = X_final
        _CACHED_DATA['y'] = y
        _CACHED_DATA['feature_names'] = final_feature_names


    X_global = _CACHED_DATA['X']
    y_global = _CACHED_DATA['y']
    
    # Shuffle and Partition
    total_samples = len(X_global)
    indices = np.arange(total_samples)
    np.random.seed(42) 
    np.random.shuffle(indices)
    X_global = X_global[indices]
    y_global = y_global[indices]
    
    part_size = total_samples // num_partitions
    start = partition_id * part_size
    end = start + part_size
    X_local = X_global[start:end]
    y_local = y_global[start:end]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_local, y_local, test_size=0.2, random_state=42, stratify=y_local)
    return (X_train, y_train), (X_test, y_test)

def train_xgb(X_train, y_train, params):
    # Weights for binary imbalance
    weights = compute_sample_weight(class_weight='balanced', y=y_train)
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)
    
    # --- BINARY CONFIGURATION ---
    config = {
        "objective": "binary:logistic", # <--- Correct objective
        "eval_metric": "logloss",       # <--- Correct metric
        "max_depth": 4,       
        "eta": 0.02,
        "subsample": 0.8,
        "colsample_bytree": 0.8, 
        "min_child_weight": 3, 
        "device": "cpu",
        "tree_method": "hist" 
    }
    
    return xgb.train(config, dtrain, num_boost_round=10)

