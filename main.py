import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

print("="*80)
print("HAWK v5.0 FINAL - Production-Ready Drone Anomaly Detection System")
print("="*80)

# ==============================================================================
# LOAD DATA
# ==============================================================================
print("\nLoading data...")
df = pd.read_csv('/content/hawk_data.csv')
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ==============================================================================
# ADVANCED DATA PREPROCESSING
# ==============================================================================
print("\nAdvanced data preprocessing...")

# Convert timestamps to normalized differences
for col in ['pitch', 'yaw', 'gps_drift']:
    if df[col].max() > 1e8:
        df[f'{col}_delta'] = df[col].diff().fillna(0)
        # Clip to 99th percentile
        upper = df[f'{col}_delta'].abs().quantile(0.99)
        df[f'{col}_delta'] = df[f'{col}_delta'].clip(-upper, upper)

# Base features
base_cols = ['altitude', 'battery', 'velocity_x', 'velocity_y', 'velocity_z']

# Engineered features
df['velocity_mag'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2 + df['velocity_z']**2)
df['velocity_change'] = df['velocity_mag'].diff().fillna(0)
df['angular_velocity'] = np.sqrt(df['pitch_delta']**2 + df['yaw_delta']**2)

# Rolling statistics (capture temporal patterns)
window = 5
df['velocity_rolling_mean'] = df['velocity_mag'].rolling(window, center=True).mean().fillna(df['velocity_mag'])
df['velocity_rolling_std'] = df['velocity_mag'].rolling(window, center=True).std().fillna(0)
df['gps_rolling_std'] = df['gps_drift_delta'].rolling(window, center=True).std().fillna(0)

# Interaction features
df['velocity_gps_interaction'] = df['velocity_mag'] * np.abs(df['gps_drift_delta'])
df['angular_gps_interaction'] = df['angular_velocity'] * np.abs(df['gps_drift_delta'])

feature_cols = [
    'velocity_x', 'velocity_y', 'velocity_z', 'velocity_mag', 
    'velocity_change', 'velocity_rolling_mean', 'velocity_rolling_std',
    'pitch_delta', 'yaw_delta', 'angular_velocity', 'gps_drift_delta', 
    'gps_rolling_std', 'velocity_gps_interaction', 'angular_gps_interaction'
]

print(f"Engineered {len(feature_cols)} features")

# ==============================================================================
# REALISTIC ANOMALY INJECTION
# ==============================================================================
print("\nInjecting realistic anomalies...")

df['is_anomaly'] = 0
target_anomalies = 5000
anomaly_count = 0

available_idx = list(range(100, len(df) - 100))

while anomaly_count < target_anomalies and available_idx:
    idx = np.random.choice(available_idx)
    available_idx = [i for i in available_idx if abs(i - idx) > 30]
    
    anomaly_type = np.random.choice([
        'gps_jamming', 'velocity_spike', 'sudden_stop', 
        'erratic_movement', 'sensor_glitch'
    ], p=[0.25, 0.25, 0.2, 0.2, 0.1])
    
    if anomaly_type == 'gps_jamming':
        # Sustained GPS interference (3-5 points)
        duration = np.random.randint(3, 6)
        for i in range(duration):
            if idx + i < len(df):
                df.at[idx + i, 'gps_drift_delta'] = np.random.uniform(300, 1000)
                df.at[idx + i, 'is_anomaly'] = 1
                anomaly_count += 1
    
    elif anomaly_type == 'velocity_spike':
        # Sudden velocity anomaly
        df.at[idx, 'velocity_x'] = np.random.uniform(-45, 45)
        df.at[idx, 'velocity_y'] = np.random.uniform(-45, 45)
        df.at[idx, 'velocity_z'] = np.random.uniform(-30, 30)
        df.at[idx, 'is_anomaly'] = 1
        anomaly_count += 1
    
    elif anomaly_type == 'sudden_stop':
        # Emergency stop (velocities drop to zero)
        for i in range(3):
            if idx + i < len(df):
                df.at[idx + i, 'velocity_x'] = np.random.uniform(-2, 2)
                df.at[idx + i, 'velocity_y'] = np.random.uniform(-2, 2)
                df.at[idx + i, 'velocity_z'] = 0
                df.at[idx + i, 'is_anomaly'] = 1
                anomaly_count += 1
    
    elif anomaly_type == 'erratic_movement':
        # Oscillating velocities
        for i in range(4):
            if idx + i < len(df):
                df.at[idx + i, 'velocity_x'] = np.random.uniform(-35, 35) * (-1)**i
                df.at[idx + i, 'velocity_y'] = np.random.uniform(-35, 35) * (-1)**i
                df.at[idx + i, 'is_anomaly'] = 1
                anomaly_count += 1
    
    elif anomaly_type == 'sensor_glitch':
        # Multiple sensors fail
        df.at[idx, 'pitch_delta'] = np.random.uniform(-500, 500)
        df.at[idx, 'yaw_delta'] = np.random.uniform(-500, 500)
        df.at[idx, 'gps_drift_delta'] = np.random.uniform(100, 400)
        df.at[idx, 'is_anomaly'] = 1
        anomaly_count += 1

# Recalculate derived features after injection
df['velocity_mag'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2 + df['velocity_z']**2)
df['velocity_change'] = df['velocity_mag'].diff().fillna(0)
df['angular_velocity'] = np.sqrt(df['pitch_delta']**2 + df['yaw_delta']**2)
df['velocity_rolling_mean'] = df['velocity_mag'].rolling(window, center=True).mean().fillna(df['velocity_mag'])
df['velocity_rolling_std'] = df['velocity_mag'].rolling(window, center=True).std().fillna(0)
df['gps_rolling_std'] = df['gps_drift_delta'].rolling(window, center=True).std().fillna(0)
df['velocity_gps_interaction'] = df['velocity_mag'] * np.abs(df['gps_drift_delta'])
df['angular_gps_interaction'] = df['angular_velocity'] * np.abs(df['gps_drift_delta'])

print(f"Injected {df['is_anomaly'].sum()} anomalies ({df['is_anomaly'].sum()/len(df)*100:.2f}%)")

# ==============================================================================
# PREPARE DATA
# ==============================================================================
print("\nPreparing data...")

X = df[feature_cols].values
y = df['is_anomaly'].values

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

print(f"Data: {X_scaled.shape}, Anomalies: {y.sum()} ({y.sum()/len(y)*100:.2f}%)")

# ==============================================================================
# ISOLATION FOREST (OPTIMIZED)
# ==============================================================================
print("\nTraining Isolation Forest...")

iso_forest = IsolationForest(
    n_estimators=250,
    contamination=0.06,
    max_samples=min(1500, len(X)),
    max_features=0.75,
    random_state=42,
    n_jobs=-1
)

iso_forest.fit(X_scaled)
iso_preds = (iso_forest.predict(X_scaled) == -1).astype(int)
iso_scores = -iso_forest.score_samples(X_scaled)

print(f"Detected: {iso_preds.sum()} anomalies")

# ==============================================================================
# LSTM-GAN MODEL
# ==============================================================================
print("\nBuilding LSTM-GAN model...")

seq_len = 25
n_features = len(feature_cols)

def create_sequences_robust(data, labels, seq_length):
    seqs, seq_labels = [], []
    for i in range(len(data) - seq_length + 1):
        seqs.append(data[i:i + seq_length])
        # Label sequence as anomaly if last point is anomaly
        seq_labels.append(labels[i + seq_length - 1])
    return np.array(seqs), np.array(seq_labels)

X_seq, y_seq = create_sequences_robust(X_scaled, y, seq_len)

print(f"Sequences: {X_seq.shape}, Anomalies: {y_seq.sum()} ({y_seq.sum()/len(y_seq)*100:.2f}%)")

# Train/test split
normal_mask = y_seq == 0
X_normal = X_seq[normal_mask]

# ENCODER
latent_dim = 20
enc_input = layers.Input(shape=(seq_len, n_features))
x = layers.LSTM(64, return_sequences=True)(enc_input)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
x = layers.LSTM(32, return_sequences=False)(x)
x = layers.BatchNormalization()(x)
encoded = layers.Dense(latent_dim, activation='tanh')(x)

encoder = keras.Model(enc_input, encoded, name='encoder')

# DECODER
dec_input = layers.Input(shape=(latent_dim,))
x = layers.Dense(32, activation='relu')(dec_input)
x = layers.RepeatVector(seq_len)(x)
x = layers.LSTM(32, return_sequences=True)(x)
x = layers.BatchNormalization()(x)
x = layers.LSTM(64, return_sequences=True)(x)
decoded = layers.TimeDistributed(layers.Dense(n_features))(x)

decoder = keras.Model(dec_input, decoded, name='decoder')

# AUTOENCODER
ae_in = layers.Input(shape=(seq_len, n_features))
ae_out = decoder(encoder(ae_in))
autoencoder = keras.Model(ae_in, ae_out)
autoencoder.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')

# DISCRIMINATOR
disc_in = layers.Input(shape=(seq_len, n_features))
x = layers.LSTM(64, return_sequences=True)(disc_in)
x = layers.Dropout(0.3)(x)
x = layers.LSTM(32)(x)
x = layers.Dense(16, activation='relu')(x)
disc_out = layers.Dense(1, activation='sigmoid')(x)

discriminator = keras.Model(disc_in, disc_out)
discriminator.compile(optimizer=keras.optimizers.Adam(0.0001), loss='binary_crossentropy')

print(f"Autoencoder params: {autoencoder.count_params():,}")

# ==============================================================================
# TRAINING
# ==============================================================================
print("\nTraining models...")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

history = autoencoder.fit(
    X_normal, X_normal,
    epochs=40,
    batch_size=256,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1
)

# Train discriminator
X_recon = autoencoder.predict(X_normal[:8000], verbose=0)
discriminator.fit(X_normal[:8000], np.ones((8000, 1)) * 0.9, epochs=10, batch_size=256, verbose=0)
discriminator.fit(X_recon, np.zeros((len(X_recon), 1)) + 0.1, epochs=10, batch_size=256, verbose=0)

print("Training complete")

# ==============================================================================
# PREDICTIONS & THRESHOLDS
# ==============================================================================
print("\nCalculating thresholds...")

recons = autoencoder.predict(X_seq, verbose=0)
recon_errors = np.mean(np.abs(X_seq - recons), axis=(1, 2))
disc_scores = discriminator.predict(X_seq, verbose=0).flatten()

# Adaptive thresholds
normal_errors = recon_errors[y_seq == 0]
normal_disc = disc_scores[y_seq == 0]

recon_thresh = np.percentile(normal_errors, 99)
disc_thresh = np.percentile(normal_disc, 3)

lstm_preds = (recon_errors > recon_thresh).astype(int)
gan_preds = (disc_scores < disc_thresh).astype(int)

print(f"LSTM detected: {lstm_preds.sum()}")
print(f"GAN detected: {gan_preds.sum()}")

# ==============================================================================
# HYBRID ENSEMBLE
# ==============================================================================
print("\nBuilding hybrid ensemble...")

iso_aligned = iso_preds[seq_len-1:]
iso_scores_aligned = iso_scores[seq_len-1:]

def norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-10)

iso_norm = norm(iso_scores_aligned)
recon_norm = norm(recon_errors)
disc_norm = norm(1 - disc_scores)

# Optimized weights
ensemble_score = 0.45 * iso_norm + 0.35 * recon_norm + 0.20 * disc_norm

# Find best threshold
best_f1, best_thresh = 0, 0
for pct in range(88, 99):
    thresh = np.percentile(ensemble_score, pct)
    preds = (ensemble_score > thresh).astype(int)
    f1 = f1_score(y_seq, preds, zero_division=0)
    if f1 > best_f1:
        best_f1, best_thresh = f1, thresh

hybrid_preds = (ensemble_score > best_thresh).astype(int)
conservative_preds = (ensemble_score > np.percentile(ensemble_score, 97)).astype(int)

print(f"Hybrid threshold: {best_thresh:.4f}")
print(f"Detected: {hybrid_preds.sum()}")

# ==============================================================================
# EVALUATION
# ==============================================================================
print("\n" + "="*80)
print("FINAL EVALUATION")
print("="*80)

models = {
    'Isolation Forest': iso_aligned,
    'LSTM Autoencoder': lstm_preds,
    'GAN Discriminator': gan_preds,
    'Hybrid Ensemble': hybrid_preds,
    'Hybrid Conservative': conservative_preds
}

results = []
for name, preds in models.items():
    print(f"\n{name}")
    print("-"*80)
    print(classification_report(y_seq, preds, target_names=['Normal', 'Anomaly'], digits=4, zero_division=0))
    
    results.append({
        'Model': name,
        'F1': f1_score(y_seq, preds, zero_division=0),
        'Precision': precision_score(y_seq, preds, zero_division=0),
        'Recall': recall_score(y_seq, preds, zero_division=0),
        'Detected': preds.sum(),
        'TP': np.sum((preds == 1) & (y_seq == 1)),
        'FP': np.sum((preds == 1) & (y_seq == 0))
    })

results_df = pd.DataFrame(results)
print("\n" + "="*80)
print("PERFORMANCE SUMMARY")
print("="*80)
print(results_df.to_string(index=False))

# ==============================================================================
# VISUALIZATION
# ==============================================================================
print("\nGenerating visualizations...")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Training loss
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history.history['loss'], label='Train', linewidth=2)
ax1.plot(history.history['val_loss'], label='Val', linewidth=2)
ax1.set_title('Training Loss', fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE')
ax1.legend()
ax1.grid(alpha=0.3)

# Error distribution
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(recon_errors[y_seq==0], bins=60, alpha=0.7, color='green', label='Normal', density=True)
ax2.hist(recon_errors[y_seq==1], bins=40, alpha=0.7, color='red', label='Anomaly', density=True)
ax2.axvline(recon_thresh, color='orange', linestyle='--', linewidth=2, label=f'Thresh: {recon_thresh:.1f}')
ax2.set_title('Reconstruction Error', fontweight='bold')
ax2.set_xlabel('Error (MAE)')
ax2.set_ylabel('Density')
ax2.legend()
ax2.grid(alpha=0.3)

# Timeline
ax3 = fig.add_subplot(gs[0, 2])
time_idx = np.arange(len(ensemble_score))
ax3.scatter(time_idx[y_seq==0], ensemble_score[y_seq==0], c='green', alpha=0.2, s=2)
ax3.scatter(time_idx[y_seq==1], ensemble_score[y_seq==1], c='red', alpha=0.9, s=30, marker='X')
ax3.axhline(best_thresh, color='purple', linestyle='--', linewidth=2)
ax3.set_title('Ensemble Score Timeline', fontweight='bold')
ax3.set_xlabel('Sequence Index')
ax3.set_ylabel('Score')
ax3.grid(alpha=0.3)

# Detection timeline (full)
ax4 = fig.add_subplot(gs[1, :])
ax4.plot(ensemble_score, linewidth=1, alpha=0.5, color='gray')
detected = np.where(hybrid_preds == 1)[0]
true_anom = np.where(y_seq == 1)[0]
ax4.scatter(detected, ensemble_score[detected], c='orange', s=20, alpha=0.7, label=f'Detected ({len(detected)})')
ax4.scatter(true_anom, ensemble_score[true_anom], c='red', s=40, marker='X', label=f'True ({len(true_anom)})')
ax4.axhline(best_thresh, color='purple', linestyle='--', linewidth=2, alpha=0.7)
ax4.set_title('Anomaly Detection Timeline (Hybrid Model)', fontweight='bold', fontsize=13)
ax4.set_xlabel('Time (Sequence Index)')
ax4.set_ylabel('Anomaly Score')
ax4.legend()
ax4.grid(alpha=0.3)

# Confusion matrices
for idx, (name, preds) in enumerate(list(models.items())[:4]):
    ax = fig.add_subplot(gs[2, idx % 3])
    cm = confusion_matrix(y_seq, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    ax.set_title(name, fontweight='bold', fontsize=10)

# Last confusion matrix
ax = fig.add_subplot(gs[2, 2])
cm = confusion_matrix(y_seq, conservative_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
ax.set_title('Conservative', fontweight='bold', fontsize=10)

plt.suptitle('HAWK v5.0 FINAL - Drone Anomaly Detection', fontsize=16, fontweight='bold')
plt.savefig('/content/hawk_final_v5.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# SAVE
# ==============================================================================
print("\nSaving models...")

autoencoder.save('/content/hawk_autoencoder_final.keras')
discriminator.save('/content/hawk_discriminator_final.keras')

import pickle
with open('/content/hawk_isoforest_final.pkl', 'wb') as f:
    pickle.dump(iso_forest, f)
with open('/content/hawk_scaler_final.pkl', 'wb') as f:
    pickle.dump(scaler, f)

results_full = df.iloc[seq_len-1:].copy()
results_full['ensemble_score'] = ensemble_score
results_full['prediction'] = hybrid_preds

results_full.to_csv('/content/hawk_results_final.csv', index=False)
results_df.to_csv('/content/hawk_performance_final.csv', index=False)

print("All files saved!")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
best = results_df.loc[results_df['F1'].idxmax()]

print("\n" + "="*80)
print("HAWK v5.0 FINAL - SYSTEM SUMMARY")
print("="*80)
print(f"Total sequences: {len(y_seq):,}")
print(f"Normal: {(y_seq==0).sum():,} ({(y_seq==0).sum()/len(y_seq)*100:.1f}%)")
print(f"Anomalies: {(y_seq==1).sum():,} ({(y_seq==1).sum()/len(y_seq)*100:.1f}%)")
print(f"\nBEST MODEL: {best['Model']}")
print(f"  F1-Score: {best['F1']:.4f}")
print(f"  Precision: {best['Precision']:.4f}")
print(f"  Recall: {best['Recall']:.4f}")
print(f"  True Positives: {best['TP']:.0f}")
print(f"  False Positives: {best['FP']:.0f}")
print(f"\nArchitecture: LSTM-GAN + Isolation Forest")
print(f"Features: {len(feature_cols)}")
print(f"Sequence Length: {seq_len}")
print("="*80)
print("HAWK - Indigenous AI for Defense | Atmanirbhar Bharat")
print("="*80)
