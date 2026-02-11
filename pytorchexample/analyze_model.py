import xgboost as xgb

# Use the same absolute path here
MODEL_PATH = "/home/learner/Desktop/Projects/Postparum_Dep/federated-learning/quickstart-pytorch/final_ppd_model.json"

with open(MODEL_PATH, "rb") as f:
    model_bytes = f.read()

bst = xgb.Booster()
bst.load_model(bytearray(model_bytes))
print("✅ Model loaded successfully for analysis.")