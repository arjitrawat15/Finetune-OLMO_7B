""" BASELINE: Random Forest with TRUE RDKit Scaffold Split """

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

df = pd.read_csv("clintox.csv")
df = df[['smiles','CT_TOX']].dropna().reset_index(drop=True)

y = df['CT_TOX'].astype(int)

print(f"Total samples: {len(df)}")
print(f"Positive: {y.sum()} ({y.mean()*100:.1f}%)")
print(f"Negative: {(1-y).sum()} ({(1-y).mean()*100:.1f}%)")

def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

df['scaffold'] = df['smiles'].apply(get_scaffold)
df = df.dropna(subset=['scaffold']).reset_index(drop=True)

scaffold_to_indices = defaultdict(list)
for i, scaf in enumerate(df['scaffold']):
    scaffold_to_indices[scaf].append(i)

# sort scaffolds by size (largest first)
scaffold_sets = sorted(scaffold_to_indices.values(), key=len, reverse=True)

# ===================== SPLIT =====================
train_idx, valid_idx, test_idx = [], [], []

n_total = len(df)
train_cutoff = int(0.7 * n_total)
valid_cutoff = int(0.85 * n_total)

for group in scaffold_sets:
    if len(train_idx) + len(group) <= train_cutoff:
        train_idx.extend(group)
    elif len(valid_idx) + len(group) <= (valid_cutoff - train_cutoff):
        valid_idx.extend(group)
    else:
        test_idx.extend(group)

print(f"\nSplit sizes:")
print(f"  Train: {len(train_idx)}")
print(f"  Valid: {len(valid_idx)}")
print(f"  Test:  {len(test_idx)}")

# ===================== FEATURES =====================
tokens = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789[]()=#+-@/\\")

def smiles_to_vector(smiles):
    counts = Counter(smiles)
    return np.array([counts.get(t, 0) for t in tokens], dtype=np.float32)

X = np.stack([smiles_to_vector(s) for s in df['smiles']])

X_train, y_train = X[train_idx], y.iloc[train_idx].values
X_valid, y_valid = X[valid_idx], y.iloc[valid_idx].values
X_test,  y_test  = X[test_idx],  y.iloc[test_idx].values

# ===================== MODEL =====================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

rf.fit(X_train, y_train)

train_probs = rf.predict_proba(X_train)[:,1]
valid_probs = rf.predict_proba(X_valid)[:,1]
test_probs  = rf.predict_proba(X_test)[:,1]

# ===================== METRICS =====================
def print_metrics(name, y_true, y_pred):
    roc = roc_auc_score(y_true, y_pred)
    pr  = average_precision_score(y_true, y_pred)
    print(f"{name:<8} ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f}")

print("\nRESULTS:")
print_metrics("Train", y_train, train_probs)
print_metrics("Valid", y_valid, valid_probs)
print_metrics("Test",  y_test,  test_probs)

print(f"\nBaseline Test ROC-AUC: {roc_auc_score(y_test, test_probs):.4f}")
