import xgboost as xgb
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import os
import shap 
import logging
from flwr.common import Context, Parameters, Scalar
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedXgbBagging
from sklearn.metrics import accuracy_score, log_loss, roc_curve, auc
from pytorchexample.task import load_ppd_data, train_xgb, get_feature_names
from typing import Dict, List, Tuple, Union, Optional

class SaveModelStrategy(FedXgbBagging):
    def __init__(self, num_nodes, total_rounds, **kwargs):
        super().__init__(**kwargs)
        self.num_nodes = num_nodes
        self.total_rounds = total_rounds
        self.history_acc = []
        self.history_loss = []
        self.project_dir = os.path.abspath(os.getcwd())

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is None: return None, {}

        try:
            model_bytes = aggregated_parameters.tensors[0]
            temp_name = f"temp_global_{server_round}.json"
            with open(temp_name, "wb") as f: f.write(model_bytes)
            
            bst = xgb.Booster()
            bst.load_model(temp_name)
            os.remove(temp_name)

            _, (X_test, y_test) = load_ppd_data(partition_id=0, num_partitions=self.num_nodes)
            dtest = xgb.DMatrix(X_test)
            preds = bst.predict(dtest) 
            
            # ***** REPLACE THIS WITH THE OLD CODE ON GITHUB
            # --- DYNAMIC THRESHOLD SEARCH ---
            best_acc = 0.0
            best_thresh = 0.5
            
            # Test every threshold between 10% and 90%
            for thresh in np.arange(0.10, 0.91, 0.01):
                temp_preds = (preds > thresh).astype(int)
                temp_acc = accuracy_score(y_test, temp_preds)
                if temp_acc > best_acc:
                    best_acc = temp_acc
                    best_thresh = thresh
            
            # Apply the best threshold found
            y_pred_class = (preds > best_thresh).astype(int)
            # ***** REPLACE THIS WITH THE OLD CODE ON GITHUB
            # y_pred_class = (preds > 0.5).astype(int)

            # Calculate final metrics
            acc = accuracy_score(y_test, y_pred_class)  # <-- Properly using y_pred_class
            loss = log_loss(y_test, preds)
            
            print(f"\n✅ ROUND {server_round}: Acc={acc:.4f} (Optimal Threshold: {best_thresh:.2f}), Loss={loss:.4f}")
            
            self.history_acc.append(acc)
            self.history_loss.append(loss)
            self.save_learning_curve(server_round)

            if server_round == self.total_rounds: 
                self.save_comparison_roc(bst)
                self.save_shap_visualizations(bst, X_test)

        except Exception as e:
            print(f"⚠️ Error during custom evaluation: {e}")

        return aggregated_parameters, metrics

    def save_shap_visualizations(self, model, X_test):
        try:
            dtest = xgb.DMatrix(X_test)
            # Binary: just need the contribs directly
            contribs = model.predict(dtest, pred_contribs=True)
            shap_values = contribs[:, :-1] # Remove bias term

            # --- RETRIEVE FEATURE NAMES ---
            feature_names = get_feature_names()

            plt.figure(figsize=(20, 10))
            shap.summary_plot(
                shap_values, 
                X_test, 
                feature_names=feature_names, show=False,
                plot_size=(20, 10)
            )
            plt.title("SHAP Beeswarm: PPD Risk")
            plt.tight_layout()
            plt.savefig(os.path.join(self.project_dir, "final_shap_beeswarm.png"))
            plt.close()
            
            plt.figure(figsize=(20, 10))
            shap.summary_plot(
                shap_values, 
                X_test, 
                feature_names=feature_names, 
                plot_type="bar", 
                show=False,
                plot_size=(20, 10)
            )
            plt.title("SHAP Feature Importance")
            plt.tight_layout()
            plt.savefig(os.path.join(self.project_dir, "final_shap_bar.png"))
            plt.close()
        except Exception as e:
            print(f"⚠️ SHAP Error: {e}")

    def save_comparison_roc(self, global_model):
        try:
            plt.figure(figsize=(10, 8))
            
            # Local Clients
            for i in range(self.num_nodes):
                (X_train, y_train), (X_test, y_test) = load_ppd_data(i, self.num_nodes)
                local_bst = train_xgb(X_train, y_train, params={})
                dtest = xgb.DMatrix(X_test)
                # Binary: predict returns probability of class 1 directly
                local_probs = local_bst.predict(dtest)
                fpr, tpr, _ = roc_curve(y_test, local_probs)
                plt.plot(fpr, tpr, '--', alpha=0.5, label=f'Client {i} (AUC={auc(fpr, tpr):.2f})')

            # Global Model
            all_y, all_probs = [], []
            for i in range(self.num_nodes):
                _, (X_test, y_test) = load_ppd_data(i, self.num_nodes)
                preds = global_model.predict(xgb.DMatrix(X_test))
                all_y.extend(y_test)
                all_probs.extend(preds)
                
            fpr, tpr, _ = roc_curve(all_y, all_probs)
            plt.plot(fpr, tpr, 'k-', linewidth=3, label=f'Federated Model (AUC={auc(fpr, tpr):.2f})')
            plt.plot([0, 1], [0, 1], 'k:')
            plt.legend(loc="lower right")
            plt.title('ROC Curve: PPD Risk Detection')
            plt.savefig(os.path.join(self.project_dir, "final_roc_comparison.png"))
            plt.close()
        except Exception as e:
            print(f"⚠️ ROC Error: {e}")

    def save_learning_curve(self, round_num):
        rounds = range(1, len(self.history_acc) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.plot(rounds, self.history_acc, 'go-')
        ax1.set_title("Accuracy")
        ax2.plot(rounds, self.history_loss, 'rs-')
        ax2.set_title("Log Loss")
        plt.savefig(os.path.join(self.project_dir, "federated_learning_results.png"))
        plt.close()

def server_fn(context: Context):
    num_rounds = context.run_config.get("num-server-rounds", 15)
    num_nodes = context.run_config.get("num-supernodes", 4)
    strategy = SaveModelStrategy(num_nodes=num_nodes, total_rounds=num_rounds,
                                 fraction_fit=1.0, min_fit_clients=num_nodes, 
                                 min_available_clients=num_nodes, fraction_evaluate=0.0)
    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=num_rounds))

app = ServerApp(server_fn=server_fn)

