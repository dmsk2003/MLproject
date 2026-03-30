import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

TRAIN_FILE = "train_pairs_features.csv"
VAL_FILE   = "val_pairs_features.csv"
TEST_FILE  = "test_pairs_features.csv"

FEATURES = ["lev_dist", "hellinger_dist", "ncd_dist"]
TARGET = "label"

def load_data(filepath):
    """Loads CSV and returns X (features) and y (labels)."""
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath)
    
    # Drop rows with NaN
    initial_len = len(df)
    df = df.dropna()
    if len(df) < initial_len:
        print(f"  Dropped {initial_len - len(df)} rows with missing values.")
        
    X = df[FEATURES]
    y = df[TARGET]
    return X, y

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=["Different (0)", "Ortholog (1)"],
                yticklabels=["Different (0)", "Ortholog (1)"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Saved confusion_matrix.png")
    plt.show()


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    X_train, y_train = load_data(TRAIN_FILE)
    X_val, y_val     = load_data(VAL_FILE) 
    X_test, y_test   = load_data(TEST_FILE)

    print(f"\nTraining Samples: {len(X_train)}")
    print(f"Testing Samples:  {len(X_test)}")

    print("\n------------ Random Forest ------------")
    print("\nTraining Random Forest Classifier...")
    # max_depth=15: Prevents overfitting on noise
    rf_model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        max_depth=15
    )
    
    rf_model.fit(X_train, y_train)
    print("Training complete.")
    print("\n--- VAL RESULTS ---")
    val_pred = rf_model.predict(X_val)
    val_accuracy = accuracy_score(y_val,val_pred)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    #print("Validation Report:")
    #print(classification_report(y_val, val_pred))

    print("\n--- TEST RESULTS ---")
    y_pred = rf_model.predict(X_test)
    y_accuracy = accuracy_score(y_test,y_pred)
    print(f"Test Accuracy: {y_accuracy:.4f}")
    y_prob = rf_model.predict_proba(X_test)[:, 1] 

    rf_auc = roc_auc_score(y_test, y_prob)
    print(f"RANDOM FOREST ROC-AUC:  {rf_auc:.4f}")
    
    plot_confusion_matrix(y_test, y_pred, "Random Forest Confusion Matrix")

    # Logistic Regression
    print("\n------------ Logistic Regression ------------")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("Training Logistic Regression...")
    log_model = LogisticRegression(solver='liblinear', random_state=42)
    log_model.fit(X_train_scaled, y_train)
    print("Training complete.")
    print("\n--- VAL RESULTS ---")
    val_predictions = log_model.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, val_predictions)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    #print("Validation Report:")
    #print(classification_report(y_val, val_predictions))
    test_predictions = log_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print("--- TEST RESULTS ---")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    #print("\nTest Report:")
    #print(classification_report(y_test, test_predictions))
    test_prob = log_model.predict_proba(X_test_scaled)[:, 1] 

    rf_auc = roc_auc_score(y_test, test_prob)
    print(f"ROC-AUC:  {rf_auc:.4f}")
    plot_confusion_matrix(y_test, test_predictions, "Logistic Regression Confusion Matrix")
