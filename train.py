import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


EMB_DIM_PCA = 64          
LR = 1e-4
EPOCHS = 200
BATCH_SIZE = 16
DROPOUT = 0.3
TEST_SIZE = 0.2
EARLY_STOPPING_PATIENCE = 15
WEIGHT_DECAY = 1e-4
NHEAD = 4
DELTA_HUBER = 2.0

DATA_PATH = "train_embeddings.pkl"
SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)


df = pd.read_pickle(DATA_PATH)
embA = np.stack(df['embA'].values)
embB = np.stack(df['embB'].values)
y = df['pK'].values.reshape(-1,1)

y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y)


if EMB_DIM_PCA is not None:
    pcaA = PCA(n_components=EMB_DIM_PCA)
    pcaB = PCA(n_components=EMB_DIM_PCA)
    if embA.ndim == 3:  
        N,L,D = embA.shape
        embA_flat = embA.reshape(N,L*D)
        embB_flat = embB.reshape(N,L*D)
        embA_pca = pcaA.fit_transform(embA_flat).reshape(N,L,EMB_DIM_PCA)
        embB_pca = pcaB.fit_transform(embB_flat).reshape(N,L,EMB_DIM_PCA)
    else:  
        embA_pca = pcaA.fit_transform(embA)
        embB_pca = pcaB.fit_transform(embB)
    embA, embB = embA_pca, embB_pca

X_A = torch.tensor(embA, dtype=torch.float32)
X_B = torch.tensor(embB, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

if X_A.ndim == 2:  # (N, D) -> (N,1,D)
    X_A = X_A.unsqueeze(1)
    X_B = X_B.unsqueeze(1)


mu, sigma = np.mean(y), np.std(y)
weights = np.ones_like(y)
tail_mask = np.abs(y - mu) > sigma
weights[tail_mask] = 1 + np.log1p(np.abs(y[tail_mask]-mu)/sigma)
weights = np.minimum(weights, 5.0)
weights = torch.tensor(weights.flatten(), dtype=torch.float32)


X_A_train, X_A_test, X_B_train, X_B_test, y_train, y_test, w_train, w_test = train_test_split(
    X_A, X_B, y_tensor, weights, test_size=TEST_SIZE, random_state=42
)

print(f"train: {len(y_train)}, test: {len(y_test)}")

class PairwiseCrossAttention(nn.Module):
    def __init__(self, embed_dim, nhead=4, dropout=0.3):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nhead, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim*2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64,1)
        )

    def forward(self, embA, embB, return_attn=False):
        # Cross-attention
        out_A2B, attn_A2B = self.cross_attn(query=embA, key=embB, value=embB)
        out_B2A, attn_B2A = self.cross_attn(query=embB, key=embA, value=embA)

        pooled_A = out_A2B.mean(dim=1)
        pooled_B = out_B2A.mean(dim=1)

        final_feat = torch.cat([pooled_A, pooled_B], dim=1)
        pred = self.fc(final_feat)

        if return_attn:
            attn_matrix = attn_A2B.mean(dim=0)  
            return pred, attn_matrix.detach().cpu().numpy()
        return pred


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_dim = EMB_DIM_PCA if EMB_DIM_PCA else (embA.shape[2] if embA.ndim==3 else embA.shape[1])
model = PairwiseCrossAttention(embed_dim=embed_dim, nhead=NHEAD, dropout=DROPOUT).to(device)

X_A_train, X_A_test = X_A_train.to(device), X_A_test.to(device)
X_B_train, X_B_test = X_B_train.to(device), X_B_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)
w_train, w_test = w_train.to(device), w_test.to(device)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
huber_loss = nn.SmoothL1Loss(reduction='none', beta=DELTA_HUBER)

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(1, EPOCHS+1):
    model.train()
    permutation = torch.randperm(X_A_train.size(0))
    epoch_loss = 0.0
    for i in range(0, X_A_train.size(0), BATCH_SIZE):
        idx = permutation[i:i+BATCH_SIZE]
        batch_A, batch_B, batch_y, batch_w = X_A_train[idx], X_B_train[idx], y_train[idx], w_train[idx]
        optimizer.zero_grad()
        outputs = model(batch_A, batch_B)
        loss_sample = huber_loss(outputs, batch_y)
        loss = (loss_sample.flatten()*batch_w).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        epoch_loss += loss.item()*batch_A.size(0)
    epoch_loss /= X_A_train.size(0)

    model.eval()
    with torch.no_grad():
        val_pred = model(X_A_test,X_B_test)
        val_loss_sample = huber_loss(val_pred,y_test)
        val_loss = (val_loss_sample.flatten()*w_test).mean().item()
    scheduler.step(val_loss)

    if epoch%10==0:
        print(f"Epoch {epoch:03d} | TrainLoss={epoch_loss:.4f} | ValLoss={val_loss:.4f}")

    # EarlyStopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(SAVE_DIR,"affinity_cross_attn.pt"))
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(" Early stopping triggered")
            break

model.load_state_dict(torch.load(os.path.join(SAVE_DIR,"affinity_cross_attn.pt")))
model.eval()
with torch.no_grad():
    pred_train = model(X_A_train,X_B_train).cpu().numpy()
    pred_test = model(X_A_test,X_B_test).cpu().numpy()

pred_train_real = y_scaler.inverse_transform(pred_train)
pred_test_real = y_scaler.inverse_transform(pred_test)
y_train_real = y_scaler.inverse_transform(y_train.cpu().numpy())
y_test_real = y_scaler.inverse_transform(y_test.cpu().numpy())

def calc_metrics(y_true,y_pred):
    mse = mean_squared_error(y_true,y_pred)
    r2 = r2_score(y_true,y_pred)
    mae = mean_absolute_error(y_true,y_pred)
    corr = np.corrcoef(y_true.flatten(),y_pred.flatten())[0,1]
    return mse,r2,mae,corr

train_mse, train_r2, train_mae, train_corr = calc_metrics(y_train_real,pred_train_real)
test_mse, test_r2, test_mae, test_corr = calc_metrics(y_test_real,pred_test_real)

print("="*50)
print("Cross-Attention Transformer结果:")
print(f"train_set: MSE={train_mse:.4f}, R2={train_r2:.3f}, MAE={train_mae:.3f}, Corr={train_corr:.3f}")
print(f"test_set: MSE={test_mse:.4f}, R2={test_r2:.3f}, MAE={test_mae:.3f}, Corr={test_corr:.3f}")
print("="*50)


joblib.dump(y_scaler, os.path.join(SAVE_DIR,"y_scaler_cross_attn.pkl"))
if EMB_DIM_PCA is not None:
    joblib.dump(pcaA, os.path.join(SAVE_DIR,"pcaA_cross_attn.pkl"))
    joblib.dump(pcaB, os.path.join(SAVE_DIR,"pcaB_cross_attn.pkl"))

