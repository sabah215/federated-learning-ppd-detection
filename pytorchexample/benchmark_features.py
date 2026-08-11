import time
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

# Import the original base data loader from task.py
from task import load_ppd_data

def simulate_converged_fl_client(X_train, y_train):
    """
    Simulates a fully converged FL client. 
    By training for 50 trees, we mimic the accumulated knowledge of a 
    10-round FL simulation (10 rounds * 5 trees = 50 trees).
    """
    weights = compute_sample_weight(class_weight='balanced', y=y_train)
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)
    
    config = {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 6,       
        "eta": 0.02,  # From 0.1         
        "subsample": 0.8,
        "colsample_bytree": 0.8, 
        "min_child_weight": 2, # from 5
        "gamma": 0.2, #from 1.0
        "lambda": 1.0,         
        "eval_metric": "mlogloss",
        "device": "cpu",
        "tree_method": "hist" 
    }

    return xgb.train(config, dtrain, num_boost_round=100) # from 50

def run_feature_benchmark():
    # Test feature subsets from N=5 to N=20
    N_values = [5, 10, 15, 20,25,30]
    
    avg_training_times = []
    fl_model_sizes_kb = []
    fl_accuracies = []
    num_clients = 4

    print("🚀 Starting Federated Feature Selection Benchmark...\n")

    for N in N_values:
        print(f"--- Testing N = {N} Features ---")
        
        client_times = []
        client_models = []
        
        # A. SIMULATE LOCAL FL CLIENT TRAINING
        for client_id in range(num_clients):
            # Load partitioned data using the standard signature (no keyword argument conflict)
            (X_train_full, y_train), _ = load_ppd_data(partition_id=client_id, num_partitions=num_clients)
            
            # SAFEGUARD: Manually truncate columns to N features right here
            X_train = X_train_full[:, :N]
            
            start_time = time.time()
            bst = simulate_converged_fl_client(X_train, y_train)
            train_time = time.time() - start_time
            
            client_times.append(train_time)
            client_models.append(bst)
            
        # Record Average local training speed
        avg_time = np.mean(client_times)
        avg_training_times.append(avg_time)
        
        # Measure network payload/model footprint size
        total_size_bytes = sum([len(bst.save_raw("json")) for bst in client_models])
        size_kb = total_size_bytes / 1024.0
        fl_model_sizes_kb.append(size_kb)
        
        # B. SIMULATE GLOBAL AGGREGATION & EVALUATION
        _, (X_test_full, y_test) = load_ppd_data(partition_id=0, num_partitions=num_clients)
        X_test = X_test_full[:, :N] # Truncate test features to match
        dtest = xgb.DMatrix(X_test)
        
        global_preds = np.zeros((len(y_test), 3))
        for bst in client_models:
            preds = bst.predict(dtest)
            
            # Safe-fallback check to prevent broadcasting shape mismatch errors
            if preds.ndim == 1:
                one_hot = np.zeros((len(y_test), 3))
                one_hot[np.arange(len(y_test)), preds.astype(int)] = 1.0
                preds = one_hot
                
            global_preds += preds
            
        global_preds /= num_clients
        y_pred = np.argmax(global_preds, axis=1)
        acc = accuracy_score(y_test, y_pred)
        fl_accuracies.append(acc)
        
        print(f"✅ Result for N={N} -> Time: {avg_time:.4f}s | Size: {size_kb:.1f} KB | Acc: {acc:.4f}\n")

    # C. GENERATE MULTI-PANEL VISUALIZATION
    plot_benchmark_results(N_values, avg_training_times, fl_model_sizes_kb, fl_accuracies)

def plot_benchmark_results(N_values, times, sizes, accuracies):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Impact of Selected Features (N) on Federated Learning Efficiency', fontsize=16, y=1.05)

    # Panel 1: Execution Latency
    axes[0].plot(N_values, times, marker='o', color='#1f77b4', linewidth=2.5, markersize=8)
    axes[0].set_title('Avg. Training Time (Converged)', fontsize=12)
    axes[0].set_xlabel('Number of Selected Features (N)')
    axes[0].set_ylabel('Time (Seconds)')

    # Panel 2: Network Transfer Footprint
    axes[1].plot(N_values, sizes, marker='s', color='#ff7f0e', linewidth=2.5, markersize=8)
    axes[1].set_title('Federated Model Size', fontsize=12)
    axes[1].set_xlabel('Number of Selected Features (N)')
    axes[1].set_ylabel('Model Size (KB)')

    # Panel 3: Global Predictive Accuracy
    axes[2].plot(N_values, accuracies, marker='^', color='#2ca02c', linewidth=2.5, markersize=8)
    axes[2].set_title('Federated Model Accuracy', fontsize=12)
    axes[2].set_xlabel('Number of Selected Features (N)')
    axes[2].set_ylabel('Accuracy')
    
    # Enable automatic dynamic scaling to avoid clipping valid performance data
    axes[2].set_ylim(auto=True) 

    plt.tight_layout()
    save_path = os.path.join(os.path.abspath(os.getcwd()), "feature_selection_benchmark.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"📊 Benchmark Plot saved successfully to: {save_path}")

if __name__ == "__main__":
    run_feature_benchmark()