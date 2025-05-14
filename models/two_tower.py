import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchmetrics
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from torchmetrics.functional import recall


# --- 1. Load and preprocess the data ---
df = pd.read_csv("../data/fnb_dataset/raw/dq_recsys_challenge_2025(in).csv")
df['int_date'] = pd.to_datetime(df['int_date'], format='%d-%b-%y')
df = df[df['interaction'].isin(['CLICK', 'CHECKOUT'])].copy()
df['item_descrip'] = df['item_descrip'].fillna("UNKNOWN")

df['user_id'] = df['idcol'].astype(str)
df['item_id'] = df['item'].astype(str)

user_encoder = LabelEncoder()
item_encoder = LabelEncoder()
df['user_idx'] = user_encoder.fit_transform(df['user_id'])
df['item_idx'] = item_encoder.fit_transform(df['item_id'])

num_users = df['user_idx'].nunique()
num_items = df['item_idx'].nunique()

print(f"Num users: {num_users}, Num items: {num_items}")

# --- 2. Dataset and DataLoader ---
class RecSysDataset(Dataset):
    def __init__(self, users, items):
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx]

dataset = RecSysDataset(df['user_idx'].values, df['item_idx'].values)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# --- 3. Two-Tower Model ---
class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_ids, item_ids):
        user_vecs = self.user_emb(user_ids)
        item_vecs = self.item_emb(item_ids)
        return (user_vecs * item_vecs).sum(dim=1)  # Dot product

    def get_user_embedding(self, user_ids):
        return self.user_emb(user_ids)

    def get_item_embedding(self):
        return self.item_emb.weight.data  # [num_items, dim]

# --- 4. Training ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TwoTowerModel(num_users, num_items).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

# --- 5. Train Loop ---
EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for user_ids, item_ids in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        user_ids, item_ids = user_ids.to(device), item_ids.to(device)
        optimizer.zero_grad()
        logits = model(user_ids, item_ids)
        labels = torch.ones_like(logits)  # All observed are positive
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

# --- 6. Evaluation with TorchMetrics ---
print("Evaluating...")

# Build recall@10


top_k = 10
all_recalls = []

# Generate predictions for each user
model.eval()
user_embs = model.get_user_embedding(torch.arange(num_users).to(device))  # [U, D]
item_embs = model.get_item_embedding().to(device)                         # [I, D]
scores = torch.matmul(user_embs, item_embs.T)                             # [U, I]

user_pos_items = df.groupby("user_idx")["item_idx"].apply(set)

for user_id in range(num_users):
    user_scores = scores[user_id]
    topk_items = torch.topk(user_scores, k=top_k).indices.cpu().numpy()

    actual_items = user_pos_items.get(user_id, set())
    if not actual_items:
        continue  # skip users with no positives

    y_pred = torch.tensor([1 if item in actual_items else 0 for item in topk_items], dtype=torch.float32)
    y_true = torch.ones_like(y_pred)  # All top-K should ideally be positive

    r = recall(preds=y_pred, target=y_true, task="binary", threshold=0.5)
    all_recalls.append(r.item())

mean_recall_at_10 = sum(all_recalls) / len(all_recalls)
print(f"Recall@{top_k}: {mean_recall_at_10:.4f}")

