# test_transformer_model.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

EMB_DIM_PCA = 64
MODEL_PATH = "models/model_transformer.pt"
Y_SCALER_PATH = "models/y_scaler_transformer.pkl"
PCA_PATH = "models/pca_transformer.pkl"
TEST_FILE = "data/test1_embeddings.pkl"

class PairwiseTransformer(nn.Module):
    def __init__(self, input_dim, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        x = self.input_proj(x)
        x = x.transpose(0,1)
        x = self.transformer(x)
        x = x.mean(dim=0)
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pca = joblib.load(PCA_PATH)
y_scaler = joblib.load(Y_SCALER_PATH)

model = PairwiseTransformer(input_dim=EMB_DIM_PCA).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

df_test = pd.read_pickle(TEST_FILE)
embA = np.stack(df_test["embA"].values)
embB = np.stack(df_test["embB"].values)

embA_pca = pca.transform(embA)
embB_pca = pca.transform(embB)

X_seq_test = np.stack([embA_pca, embB_pca], axis=1)
X_test_tensor = torch.tensor(X_seq_test, dtype=torch.float32).to(device)

with torch.no_grad():
    preds_scaled = model(X_test_tensor).cpu().numpy()

preds_real = y_scaler.inverse_transform(preds_scaled)
df_test["pred_affinity"] = preds_real

if "pK" in df_test.columns:
    y_true = df_test["pK"].values
    y_pred = df_test["pred_affinity"].values

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]

    print("=" * 60)
    print(f"MSE = {mse:.4f}")
    print(f"MAE = {mae:.4f}")
    print(f"R²  = {r2:.4f}")
    print(f"PCC = {corr:.4f}")
    print("=" * 60)

    plt.figure(figsize=(6,6))
    sns.scatterplot(x=y_true, y=y_pred, s=40, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label="y = x")
    plt.xlabel("True pK")
    plt.ylabel("Predicted pK")
    plt.title("Transformer: Prediction vs True pK")
    plt.legend()
    plt.grid(True)
    plt.show()

else:
    plt.figure(figsize=(6,4))
    sns.histplot(df_test["pred_affinity"], bins=30, kde=True)
    plt.title("Transformer: Predicted pK Distribution")
    plt.xlabel("Predicted pK")
    plt.ylabel("Count")
    plt.show()

output_path = TEST_FILE.replace(".pkl", "_predicted_transformer.pkl")
df_test.to_pickle(output_path)

if "pK" in df_test.columns:
    y_true_pk = df_test["pK"].values 
    y_pred_pk = df_test["pred_affinity"].values  

    mse_pk = mean_squared_error(y_true_pk, y_pred_pk)
    mae_pk = mean_absolute_error(y_true_pk, y_pred_pk)
    r2_pk = r2_score(y_true_pk, y_pred_pk)
    pcc_pk = np.corrcoef(y_true_pk.flatten(), y_pred_pk.flatten())[0, 1]

    print("=" * 60)
    print(f"MSE (pK) = {mse_pk:.4f}")
    print(f"MAE (pK) = {mae_pk:.4f}")
    print(f"R²  (pK) = {r2_pk:.4f}")
    print(f"PCC (pK) = {pcc_pk:.4f}")
    print("=" * 60)

    R = 1.987  
    T = 298    
    conversion_factor = 1000  

    # 1. pK → Kd（mol/L）
    y_true_kd = 10 ** (-y_true_pk)
    y_pred_kd = 10 ** (-y_pred_pk)

    # 2. Kd → ΔG（kcal/mol）
    y_true_dg = (R * T * np.log(y_true_kd)) / conversion_factor
    y_pred_dg = (R * T * np.log(y_pred_kd)) / conversion_factor

    mae_dg = mean_absolute_error(y_true_dg, y_pred_dg)  
    r_dg = np.corrcoef(y_true_dg.flatten(), y_pred_dg.flatten())[0, 1]  


    print("=" * 60)
    print(f"MAE (ΔG, kcal/mol) = {mae_dg:.4f}") 
    print(f"R (ΔG) = {r_dg:.4f}")               
    print("=" * 60)

    plt.figure(figsize=(6,6))
    sns.scatterplot(x=y_true_dg, y=y_pred_dg, s=40, alpha=0.7)
    plt.plot([y_true_dg.min(), y_true_dg.max()], [y_true_dg.min(), y_true_dg.max()], 'r--', label="y = x")
    plt.xlabel("True ΔG (kcal/mol)")
    plt.ylabel("Predicted ΔG (kcal/mol)")
    plt.title("Transformer: Prediction vs True ΔG (Consistent with Literature)")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=y_true_pk, y=y_pred_pk, s=40, alpha=0.7)
    plt.plot([y_true_pk.min(), y_true_pk.max()], [y_true_pk.min(), y_true_pk.max()], 'r--', label="y = x")
    plt.xlabel("True pK")
    plt.ylabel("Predicted pK")
    plt.title("Transformer: Prediction vs True pK")
    plt.legend()
    plt.grid(True)
    plt.show()

else:
    plt.figure(figsize=(6,4))
    sns.histplot(df_test["pred_affinity"], bins=30, kde=True)
    plt.title("Transformer: Predicted pK Distribution")
    plt.xlabel("Predicted pK")
    plt.ylabel("Count")
    plt.show()

if "pK" in df_test.columns:
    df_test["true_ΔG"] = y_true_dg
    df_test["pred_ΔG"] = y_pred_dg

output_path = TEST_FILE.replace(".pkl", "_predicted_transformer.pkl")
df_test.to_pickle(output_path)

