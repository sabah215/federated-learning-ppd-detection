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
from sklearn.metrics import accuracy_score, log_loss, roc_curve, auc, precision_score, recall_score, f1_score, roc_auc_score
from pytorchexample.task import load_ppd_data, train_xgb, get_feature_names
from typing import Dict, List, Tuple, Union, Optional

class SaveModelStrategy(FedXgbBagging):
    def __init__(self, num_nodes, total_rounds, **kwargs):
        super().__init__(**kwargs)
        self.num_nodes = num_nodes
        self.total_rounds = total_rounds
        self.history_acc = []
        self.history_loss = []
        self.history_val_acc = []
        self.history_val_loss = []
        # --- NEW TRACKERS FOR CLASSIFICATION METRICS ---
        self.history_f1 = []
        self.history_auc = []
        self.history_precision = []
        self.history_recall = []

        self.project_dir = os.path.abspath(os.getcwd())

    def save_client_metrics(self, bst, round_num):
        """Evaluates the global model on each individual client's test set."""
        try:
            client_names = []
            client_auc = []
            client_f1 = []
            client_prec = []
            client_rec = []

            # Lists to hold the combined test data for the overall Federated metrics
            all_y = []
            all_probs = []

            # --- 1. Loop through every client for individual metrics ---
            for i in range(self.num_nodes):
                _, (X_test, y_test) = load_ppd_data(partition_id=i, num_partitions=self.num_nodes)
                
                dtest = xgb.DMatrix(X_test)
                preds_test = bst.predict(dtest)
                y_pred_class = (preds_test > 0.5).astype(int)

                # Store for global calculation later
                all_y.extend(y_test)
                all_probs.extend(preds_test)

                # Calculate individual client metrics
                try:
                    auc_val = roc_auc_score(y_test, preds_test)
                except ValueError:
                    auc_val = 0.5 
                
                f1 = f1_score(y_test, y_pred_class, zero_division=0)
                prec = precision_score(y_test, y_pred_class, zero_division=0)
                rec = recall_score(y_test, y_pred_class, zero_division=0)

                # Store metrics
                client_names.append(f"Client {i}")
                client_auc.append(auc_val)
                client_f1.append(f1)
                client_prec.append(prec)
                client_rec.append(rec)

            # --- 2. Calculate OVERALL FEDERATED metrics ---
            all_y = np.array(all_y)
            all_probs = np.array(all_probs)
            all_pred_class = (all_probs > 0.5).astype(int)

            try:
                global_auc = roc_auc_score(all_y, all_probs)
            except ValueError:
                global_auc = 0.5
            
            global_f1 = f1_score(all_y, all_pred_class, zero_division=0)
            global_prec = precision_score(all_y, all_pred_class, zero_division=0)
            global_rec = recall_score(all_y, all_pred_class, zero_division=0)

            # Append the Federated overall scores to the lists
            client_names.append("Federated\n(Overall)")
            client_auc.append(global_auc)
            client_f1.append(global_f1)
            client_prec.append(global_prec)
            client_rec.append(global_rec)

            # --- 3. PLOT GROUPED BAR CHART ---
            x = np.arange(len(client_names))  # Automatically adjusts to fit the 5 groups
            width = 0.2

            # Increased figure width slightly to accommodate the 5th column
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Draw bars for each metric side-by-side
            rects1 = ax.bar(x - 1.5*width, client_auc, width, label='AUC', color='purple', edgecolor='black')
            rects2 = ax.bar(x - 0.5*width, client_f1, width, label='F1 Score', color='brown', edgecolor='black')
            rects3 = ax.bar(x + 0.5*width, client_prec, width, label='Precision', color='teal', edgecolor='black')
            rects4 = ax.bar(x + 1.5*width, client_rec, width, label='Recall', color='darkorange', edgecolor='black')

            # Formatting
            ax.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax.set_title(f'Model Performance: Individual Clients vs. Overall Federated (Round {round_num})', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(client_names, fontsize=12, fontweight='bold')
            ax.set_ylim([0.0, 1.05])
            ax.legend(loc='lower right')
            ax.grid(axis='y', linestyle='--', alpha=0.5)

            # Save plot
            path = os.path.join(self.project_dir, "federated_client_comparison.png")
            plt.tight_layout()
            plt.savefig(path, dpi=300)
            plt.close()
            
            print(f"   📸 Client vs Global Comparison Chart Saved: {path}")

        except Exception as e:
            print(f"⚠️ Client Metrics Plotting Error: {e}")

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

            # Extract both Train (Validation) and Test data
            (X_train, y_train), (X_test, y_test) = load_ppd_data(partition_id=0, num_partitions=self.num_nodes)
            
            # 1. TEST Metrics (Unseen Data)
            dtest = xgb.DMatrix(X_test)
            preds_test = bst.predict(dtest) 
            y_pred_class = (preds_test > 0.5).astype(int)
            
            test_acc = accuracy_score(y_test, y_pred_class)
            test_loss = log_loss(y_test, preds_test)
            
            # --- CALCULATE NEW METRICS ---
            precision = precision_score(y_test, y_pred_class, zero_division=0)
            recall = recall_score(y_test, y_pred_class, zero_division=0)
            f1 = f1_score(y_test, y_pred_class, zero_division=0)
            try:
                auc_val = roc_auc_score(y_test, preds_test)
            except ValueError:
                auc_val = 0.5 # Fallback if test set only has 1 class by chance
            
            # 2. VALIDATION Metrics (Training Data)
            dtrain = xgb.DMatrix(X_train)
            preds_val = bst.predict(dtrain)
            val_acc = accuracy_score(y_train, (preds_val > 0.5).astype(int))
            val_loss = log_loss(y_train, preds_val)
            
            # 3. Store ALL metrics
            self.history_acc.append(test_acc)
            self.history_loss.append(test_loss)
            self.history_val_acc.append(val_acc)
            self.history_val_loss.append(val_loss)
            self.history_precision.append(precision)
            self.history_recall.append(recall)
            self.history_f1.append(f1)
            self.history_auc.append(auc_val)
            
            print(f"\n✅ ROUND {server_round}: Acc={test_acc:.4f} | AUC={auc_val:.4f} | F1={f1:.4f} | Prec={precision:.4f} | Rec={recall:.4f}")
            
            # Generate plots
            self.save_learning_curve(server_round)
            self.save_classification_metrics(server_round) # <--- NEW PLOT

            # --- END OF TRAINING PLOTS ---
            if server_round == self.total_rounds: 
                self.save_comparison_roc(bst)
                self.save_shap_visualizations(bst, X_test)
                self.save_final_bar_chart() 
                self.save_client_metrics(bst, server_round) # <--- ADD THIS LINE

        except Exception as e:
            print(f"⚠️ Error during custom evaluation: {e}")

        return aggregated_parameters, metrics
    
    # --- NEW BAR CHART FUNCTION ---
    def save_final_bar_chart(self):
        try:
            # Dynamically grab the last recorded value from round 20
            metrics = ['AUC', 'F1 Score', 'Precision', 'Recall']
            scores = [
                self.history_auc[-1], 
                self.history_f1[-1], 
                self.history_precision[-1], 
                self.history_recall[-1]
            ]
            
            plt.figure(figsize=(8, 6))
            colors = ['#800080', '#A52A2A', '#008080', '#FF8C00'] 
            bars = plt.bar(metrics, scores, color=colors, width=0.5, edgecolor='black', linewidth=1)
            
            # Write exact numbers above the bars
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval + 0.015, 
                         f'{yval:.3f}', ha='center', va='bottom', 
                         fontsize=12, fontweight='bold')
            
            plt.ylim([0.0, 1.05]) 
            plt.ylabel('Score', fontsize=14, fontweight='bold')
            plt.title(f'Final Federated Model Performance (Round {self.total_rounds})', fontsize=16, fontweight='bold')
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            
            path = os.path.join(self.project_dir, "federated_final_barchart.png")
            plt.tight_layout()
            plt.savefig(path, dpi=300) 
            plt.close()
            print(f"   📸 Final Bar Chart Saved: {path}")
        except Exception as e:
             print(f"⚠️ Bar Chart Plotting Error: {e}")

    def save_classification_metrics(self, round_num):
        try:
            rounds = range(1, len(self.history_f1) + 1)
            plt.figure(figsize=(10, 6))
            
            plt.plot(rounds, self.history_auc, color='purple', marker='^', linestyle='-', linewidth=2.5, label='AUC')
            plt.plot(rounds, self.history_f1, color='brown', marker='s', linestyle='-', linewidth=2.5, label='F1 Score')
            plt.plot(rounds, self.history_precision, color='teal', marker='o', linestyle='--', linewidth=2, label='Precision')
            plt.plot(rounds, self.history_recall, color='darkorange', marker='d', linestyle='--', linewidth=2, label='Recall')
            
            plt.title(f"Classification Metrics over Federated Rounds (Round {round_num})", fontsize=15, fontweight='bold')
            plt.xlabel('Federated Rounds', fontsize=12)
            plt.ylabel('Score', fontsize=12)
            plt.ylim([0.70, 0.95]) # Zoom in to show the variance clearly
            plt.xticks(rounds)
            plt.legend(loc="lower right", framealpha=0.9)
            plt.grid(True, linestyle=':', alpha=0.7)
            
            path = os.path.join(self.project_dir, "federated_classification_metrics.png")
            plt.tight_layout()
            plt.savefig(path, dpi=300) 
            plt.close()
        except Exception as e:
             print(f"⚠️ Metrics Plotting Error: {e}")

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
        try:
            rounds = range(1, len(self.history_acc) + 1)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # --- ACCURACY PLOT ---
            ax1.plot(rounds, self.history_acc, color='green', marker='s', 
                     linestyle='-', linewidth=2, label='Test Accuracy')
            ax1.plot(rounds, self.history_val_acc, color='blue', marker='o', 
                     linestyle='--', linewidth=2, label='Validation Accuracy')
            
            ax1.set_title(f"Accuracy (Round {round_num})")
            ax1.set_xlabel('Federated Rounds')
            ax1.set_ylabel('Accuracy')
            ax1.legend(loc="lower right")
            ax1.grid(True, alpha=0.3)
            
            # --- LOSS PLOT ---
            ax2.plot(rounds, self.history_loss, color='red', marker='s', 
                     linestyle='-', linewidth=2, label='Training Loss')
            ax2.plot(rounds, self.history_val_loss, color='orange', marker='o', 
                     linestyle='--', linewidth=2, label='Validation Loss')
            ax2.set_title(f"Log Loss (Round {round_num})")
            ax2.set_xlabel('Federated Rounds')
            ax2.set_ylabel('Loss')
            ax2.legend(loc="upper right")
            ax2.grid(True, alpha=0.3)
            
            path = os.path.join(self.project_dir, "federated_learning_results.png")
            plt.savefig(path, dpi=300) 
            plt.close()
        except Exception as e:
             print(f"⚠️ Plotting Error: {e}")

def server_fn(context: Context):
    num_rounds = context.run_config.get("num-server-rounds", 21)
    num_nodes = context.run_config.get("num-supernodes", 4)
    strategy = SaveModelStrategy(num_nodes=num_nodes, total_rounds=num_rounds,
                                 fraction_fit=1.0, min_fit_clients=num_nodes, 
                                 min_available_clients=num_nodes, fraction_evaluate=0.0)
    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=num_rounds))

app = ServerApp(server_fn=server_fn)

