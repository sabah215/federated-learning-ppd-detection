"""client_app.py: Defines the Client behavior for federated XGBoost."""
import os
import uuid

import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from flwr.client import Client, ClientApp
from flwr.common import (
    Context,
    FitRes,
    EvaluateRes,
    Status,
    Code,
    Parameters,
)

from pytorchexample.task import load_ppd_data, train_xgb

print(f"DEBUG: Client is running XGBoost Version: {xgb.__version__}")


class FlowerXGBClient(Client):
    def __init__(self, train_set, test_set):
        self.X_train, self.y_train = train_set
        self.X_test, self.y_test = test_set

    def fit(self, ins) -> FitRes:
        # Local training, ignoring ins.parameters (bagging adds new trees)
        bst = train_xgb(self.X_train, self.y_train, params={})

        # Serialize model to JSON bytes
        unique_name = f"model_{uuid.uuid4().hex}.json"
        model_bytes = b""
        try:
            bst.save_model(unique_name)
            with open(unique_name, "rb") as f:
                model_bytes = f.read()
        finally:
            if os.path.exists(unique_name):
                os.remove(unique_name)

        params = Parameters(tensors=[model_bytes], tensor_type="bytes")

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=params,
            num_examples=len(self.X_train),
            metrics={},
        )

    def evaluate(self, ins) -> EvaluateRes:
        """Evaluate the global model on local test data for debugging."""
        # If no global model yet (e.g., round 0), return dummy
        if not ins.parameters.tensors:
            return EvaluateRes(
                status=Status(code=Code.OK, message="No global model"),
                loss=1.0,
                num_examples=len(self.X_test),
                metrics={"accuracy": 0.0},
            )

        try:
            bst = xgb.Booster()
            # parameters.tensors[0] contains the JSON bytes
            bst.load_model(bytearray(ins.parameters.tensors[0]))
            dtest = xgb.DMatrix(self.X_test)
            preds_probs = bst.predict(dtest)

            preds_labels = np.argmax(preds_probs, axis=1)
            loss = log_loss(self.y_test, preds_probs, labels=[0, 1, 2])
            acc = accuracy_score(self.y_test, preds_labels)

            return EvaluateRes(
                status=Status(code=Code.OK, message="OK"),
                loss=float(loss),
                num_examples=len(self.X_test),
                metrics={"accuracy": float(acc), "loss": float(loss)},
            )
        except Exception as e:
            # Fallback if anything goes wrong
            return EvaluateRes(
                status=Status(code=Code.ERROR, message=f"Eval error: {e}"),
                loss=1.0,
                num_examples=len(self.X_test),
                metrics={"accuracy": 0.0},
            )


def client_fn(context: Context) -> Client:
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.run_config["num-supernodes"])

    train_set, test_set = load_ppd_data(partition_id, num_partitions)
    return FlowerXGBClient(train_set, test_set).to_client()


app = ClientApp(client_fn=client_fn)
