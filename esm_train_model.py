import numpy as np
import pandas as pd
import torch
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc

# CONFIG
EMBEDDINGS_FILE = "bacillales_1385_3families_embeddings.pt"

TRAIN_PAIRS = "train_pairs.csv"
VAL_PAIRS   = "val_pairs.csv"
TEST_PAIRS  = "test_pairs.csv"

COL_A  = "gene1_id"
COL_B  = "gene2_id"
TARGET = "label"

RANDOM_SEED = 42



def plot_roc_used_threshold(y_true, y_prob, used_threshold, title, out_png):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    idx = np.argmin(np.abs(thresholds - used_threshold))

    y_pred = (y_prob >= used_threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")

    plt.scatter(
        fpr[idx], tpr[idx],
        s=120, marker='x', linewidths=3,
        label=f"Threshold = {used_threshold:.3f}, Acc = {acc:.4f}"
    )

    plt.annotate(
        f"thr={used_threshold:.3f}\nacc={acc:.4f}",
        (fpr[idx], tpr[idx]),
        textcoords="offset points",
        xytext=(10, -15),
        fontsize=9
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Saved {out_png}")
    plt.show()


def load_embeddings(path):
    print(f"Loading embeddings from {path} ...")
    emb = torch.load(path, map_location="cpu")

    # If saved as (dict, meta) or [dict, ...], extract dict
    if isinstance(emb, (list, tuple)) and len(emb) >= 1 and isinstance(emb[0], dict):
        emb = emb[0]

    if not isinstance(emb, dict):
        raise ValueError("Embeddings file is not a dict {gene_id: tensor}.")

    cleaned = {}
    for k, v in emb.items():
        gid = str(k)
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
        if not torch.is_tensor(v):
            continue
        v = v.detach().cpu()
        if v.ndim == 2 and v.shape[0] == 1:
            v = v.squeeze(0)
        if v.ndim != 1:
            continue
        cleaned[gid] = v.float()

    if len(cleaned) == 0:
        raise ValueError("No valid embeddings found after cleaning.")

    d0 = next(iter(cleaned.values())).shape[0]
    print(f"Loaded {len(cleaned)} embeddings. Dim = {d0}")
    return cleaned


def cosine_distance(e1, e2, eps= 1e-12):
    """
    Cosine distance = 1 - cosine_similarity
    """
    u = e1.numpy()
    v = e2.numpy()
    num = float(np.dot(u, v))
    den = float(np.linalg.norm(u) * np.linalg.norm(v)) + eps
    cos_sim = num / den
    return 1.0 - cos_sim


def load_pairs_make_Xy(pairs_csv, emb_dict, drop_missing=True):
    """
    Reads pairs and builds X (cosine distance) and y.
    X shape = (N, 1)
    """
    print(f"\nLoading pairs from {pairs_csv} ...")
    df = pd.read_csv(pairs_csv)

    for c in [COL_A, COL_B, TARGET]:
        if c not in df.columns:
            raise ValueError(f"Pairs file {pairs_csv} missing column '{c}'. Found: {list(df.columns)}")

    df[COL_A] = df[COL_A].astype(str)
    df[COL_B] = df[COL_B].astype(str)

    X_list, y_list = [], []
    missing = 0

    for a, b, y in zip(df[COL_A].values, df[COL_B].values, df[TARGET].values):
        e1 = emb_dict.get(a)
        e2 = emb_dict.get(b)
        if e1 is None or e2 is None:
            missing += 1
            if drop_missing:
                continue
            else:
                continue

        cd = cosine_distance(e1, e2)
        X_list.append([cd])          
        y_list.append(int(y))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    print(f"Built X,y: X shape={X.shape}, y shape={y.shape}")
    if missing > 0:
        print(f"WARNING: {missing} pairs skipped due to missing embeddings (gene_id not found).")
    return X, y


def plot_confusion_matrix(y_true, y_pred, title, out_png):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Label 0", "Label 1 (Ortholog)"],
        yticklabels=["Label 0", "Label 1 (Ortholog)"]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Saved {out_png}")
    plt.show()


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)

    emb = load_embeddings(EMBEDDINGS_FILE)

    X_train, y_train = load_pairs_make_Xy(TRAIN_PAIRS, emb)
    X_val, y_val     = load_pairs_make_Xy(VAL_PAIRS, emb)
    X_test, y_test   = load_pairs_make_Xy(TEST_PAIRS, emb)

    print(f"\nTrain: {len(y_train)}  Val: {len(y_val)}  Test: {len(y_test)}")
    print("Feature = cosine_distance (1 - cosine_similarity)")

    # Logistic Regression
    print("\n------------ Logistic Regression ------------")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    log_model = LogisticRegression(
        solver="liblinear",
        random_state=RANDOM_SEED,
        max_iter=2000
    )
    print("Training Logistic Regression...")
    log_model.fit(X_train_s, y_train)

    print("\n--- VAL RESULTS ---")
    val_pred = log_model.predict(X_val_s)
    val_prob = log_model.predict_proba(X_val_s)[:, 1]
    print(f"Val Accuracy: {accuracy_score(y_val, val_pred):.4f}")
    print(f"Val ROC-AUC : {roc_auc_score(y_val, val_prob):.4f}")

    print("\n--- TEST RESULTS ---")
    test_pred = log_model.predict(X_test_s)
    test_prob = log_model.predict_proba(X_test_s)[:, 1]
    print(f"Test Accuracy: {accuracy_score(y_test, test_pred):.4f}")
    print(f"Test ROC-AUC : {roc_auc_score(y_test, test_prob):.4f}")

    plot_confusion_matrix(
        y_test, test_pred,
        title="Logistic Regression Confusion Matrix",
        out_png="confusion_matrix_logreg_cosine.png"
    )


    plot_roc_used_threshold(
    y_test,
    test_prob,
    used_threshold=0.5,
    title="Logistic Regression ROC Curve",
    out_png="roc_logreg_used_threshold.png"
    )

    # Save the Logistic Regression Pipeline
    print("\nSaving Logistic Regression model and scaler...")
    joblib.dump(log_model, "log_model_orthology.pkl")
    joblib.dump(scaler, "scaler_orthology.pkl")
    print("Logistic Regression model saved successfully!")

    # Random Forest
    print("\n------------ Random Forest ------------")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_SEED,
        max_depth=10,
        n_jobs=-1
    )
    print("Training Random Forest...")
    rf_model.fit(X_train, y_train)

    print("\n--- VAL RESULTS ---")
    val_pred = rf_model.predict(X_val)
    val_prob = rf_model.predict_proba(X_val)[:, 1]
    print(f"Val Accuracy: {accuracy_score(y_val, val_pred):.4f}")
    print(f"Val ROC-AUC : {roc_auc_score(y_val, val_prob):.4f}")

    print("\n--- TEST RESULTS ---")
    test_pred = rf_model.predict(X_test)
    test_prob = rf_model.predict_proba(X_test)[:, 1]
    print(f"Test Accuracy: {accuracy_score(y_test, test_pred):.4f}")
    print(f"Test ROC-AUC : {roc_auc_score(y_test, test_prob):.4f}")

    plot_confusion_matrix(
        y_test, test_pred,
        title="Random Forest Confusion Matrix",
        out_png="confusion_matrix_rf_cosine.png"
    )

 
    plot_roc_used_threshold(
        y_test,
        test_prob,
        used_threshold=0.5,
        title="Random Forest ROC Curve",
        out_png="roc_rf_cosine.png"
    )

    # Save the Random Forest
    print("Saving Random Forest model...")
    joblib.dump(rf_model, "rf_model_orthology.pkl")