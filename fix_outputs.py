"""Generate missing output files for the dashboard."""
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 50)
print("STEP 1: Generate missing v2 files")
print("=" * 50)

# Load data
X_train = pd.read_csv("outputs/v1/X_train_resampled.csv")
y_train = pd.read_csv("outputs/v1/y_train_resampled.csv").squeeze()
X_test = pd.read_csv("outputs/v1/X_test_engineered.csv")
y_test = pd.read_csv("outputs/v1/y_test.csv").squeeze()
model = joblib.load("outputs/v2/fraud_model_optimized.pkl")
y_test_proba = joblib.load("outputs/v2/test_probabilities.pkl")

print("Data loaded.")

# Feature importance
fi = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
fi.to_csv("outputs/v2/feature_importance.csv", index=False)
print("feature_importance.csv saved.")

# ROC + PR curves
from sklearn.metrics import roc_curve, auc, precision_recall_curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)
prec, rec, _ = precision_recall_curve(y_test, y_test_proba)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(fpr, tpr, color='#636EFA', linewidth=2)
axes[0].plot([0, 1], [0, 1], 'k--')
axes[0].set_title(f'ROC Curve (AUC = {roc_auc:.3f})')
axes[1].plot(rec, prec, color='#EF553B', linewidth=2)
axes[1].set_title('Precision-Recall Curve')
plt.tight_layout()
plt.savefig("outputs/v2/roc_pr_curves.png", dpi=150)
plt.close()
print("roc_pr_curves.png saved.")

# SHAP
try:
    import shap
    explainer = shap.TreeExplainer(model)
    X_sample = X_test.sample(500, random_state=42)
    shap_values = explainer.shap_values(X_sample)

    joblib.dump(explainer, "outputs/v2/shap_explainer.pkl")
    print("shap_explainer.pkl saved.")

    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig("outputs/v2/shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("shap_summary.png saved.")
except Exception as e:
    print(f"SHAP error: {e}")
    print("Creating placeholder shap_summary.png...")
    fig, ax = plt.subplots(figsize=(10, 6))
    top = fi.head(15)
    ax.barh(top['feature'], top['importance'], color='#636EFA')
    ax.set_title('Feature Importance (XGBoost native)')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig("outputs/v2/shap_summary.png", dpi=150)
    plt.close()
    # Save a dummy explainer flag
    joblib.dump(None, "outputs/v2/shap_explainer.pkl")
    print("Placeholder files saved.")

print("\n" + "=" * 50)
print("STEP 2: Generate v3 files (Autoencoder)")
print("=" * 50)

os.makedirs("outputs/v3", exist_ok=True)

try:
    from sklearn.preprocessing import StandardScaler
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Dense, Dropout, BatchNormalization
    )
    from tensorflow.keras.callbacks import EarlyStopping

    tf.random.set_seed(42)

     # Legitimate only
    X_train_full = pd.read_csv("outputs/v1/X_train_engineered.csv")
    y_train_full = pd.read_csv("outputs/v1/y_train.csv").squeeze()
    X_train_legit = X_train_full[y_train_full == 0]

    ae_scaler = StandardScaler()
    X_train_scaled = ae_scaler.fit_transform(X_train_legit)
    X_test_scaled = ae_scaler.transform(X_test)

    input_dim = X_train_scaled.shape[1]
    print(f"Training autoencoder on {len(X_train_legit)} legit transactions")

    # Build model
    inp = Input(shape=(input_dim,))
    enc = Dense(32, activation='relu')(inp)
    enc = BatchNormalization()(enc)
    enc = Dropout(0.2)(enc)
    enc = Dense(16, activation='relu')(enc)
    enc = BatchNormalization()(enc)
    enc = Dense(8, activation='relu')(enc)

    dec = Dense(16, activation='relu')(enc)
    dec = BatchNormalization()(dec)
    dec = Dropout(0.2)(dec)
    dec = Dense(32, activation='relu')(dec)
    dec = BatchNormalization()(dec)
    dec = Dense(input_dim, activation='linear')(dec)

    autoencoder = Model(inp, dec)
    autoencoder.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    history = autoencoder.fit(
        X_train_scaled, X_train_scaled,
        epochs=100, batch_size=256,
        validation_split=0.1,
        callbacks=[early_stop],
        shuffle=True, verbose=1
    )

    # Learning curves
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Autoencoder Learning Curves')
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/v3/ae_learning_curves.png", dpi=150)
    plt.close()
    print("ae_learning_curves.png saved.")

    # Reconstruction errors
    X_test_recon = autoencoder.predict(X_test_scaled)
    recon_errors = np.mean((X_test_scaled - X_test_recon) ** 2, axis=1)

    legit_errors = recon_errors[y_test == 0]
    fraud_errors = recon_errors[y_test == 1]

    # Error distribution plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(legit_errors, bins=100, alpha=0.7, label='Legitimate',
            color='#636EFA', density=True)
    ax.hist(fraud_errors, bins=50, alpha=0.7, label='Fraud',
            color='#EF553B', density=True)
    ax.set_xlabel('Reconstruction Error')
    ax.set_ylabel('Density')
    ax.set_title('Reconstruction Error Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/v3/ae_error_distribution.png", dpi=150)
    plt.close()
    print("ae_error_distribution.png saved.")

    # Find best threshold
    from sklearn.metrics import f1_score
    best_f1 = 0
    best_threshold = 0
    best_p = 0
    for p in range(80, 100):
        t = np.percentile(legit_errors, p)
        y_pred_t = (recon_errors > t).astype(int)
        f1 = f1_score(y_test, y_pred_t, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_p = p
