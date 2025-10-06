# ============================================================================
# HAWK ANOMALY DETECTION - COMPLETE EVALUATION WITH YOUR DATASET
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_score, recall_score, 
                            f1_score)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import time
warnings.filterwarnings('ignore')

print(f"TensorFlow: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")


# ============================================================================
# LOAD YOUR HAWK DATASET
# ============================================================================
print("\n" + "="*80)
print("LOADING HAWK DATASET")
print("="*80)

# Load your CSV file
data_path = '/content/hawk_data.csv'  # Change path if needed
full_data = pd.read_csv(data_path)

print(f"Dataset loaded: {full_data.shape}")
print(f"Columns: {full_data.columns.tolist()}")
print("\nFirst 5 rows:")
print(full_data.head())

# Check for missing values
print(f"\nMissing values:\n{full_data.isnull().sum()}")

# Handle any NaN/Inf values
full_data = full_data.replace([np.inf, -np.inf], np.nan)
full_data = full_data.fillna(method='ffill').fillna(method='bfill').fillna(0)

# Add timestamp
full_data['timestamp'] = pd.date_range(start='2024-01-01', periods=len(full_data), freq='S')

print(f"\nCleaned data shape: {full_data.shape}")


# ============================================================================
# CREATE SYNTHETIC LABELS FOR EVALUATION
# ============================================================================
print("\n" + "="*80)
print("GENERATING SYNTHETIC ANOMALY LABELS")
print("="*80)

def add_anomaly_labels(data, contamination=0.08):
    """
    Add synthetic anomaly labels based on statistical outliers
    Uses multiple detection methods to create ground truth
    """
    data_copy = data.copy()
    features = ['altitude', 'pitch', 'yaw', 'battery', 
                'velocity_x', 'velocity_y', 'velocity_z', 'gps_drift']
    
    # Initialize labels as normal
    data_copy['label'] = 0
    
    # Method 1: Z-score based outliers
    for col in features:
        z_scores = np.abs((data_copy[col] - data_copy[col].mean()) / data_copy[col].std())
        data_copy.loc[z_scores > 3.5, 'label'] = 1
    
    # Method 2: IQR based outliers
    for col in features:
        Q1 = data_copy[col].quantile(0.25)
        Q3 = data_copy[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2.5 * IQR
        upper_bound = Q3 + 2.5 * IQR
        data_copy.loc[(data_copy[col] < lower_bound) | (data_copy[col] > upper_bound), 'label'] = 1
    
    # Method 3: Rapid changes (derivatives)
    for col in features:
        diff = np.abs(data_copy[col].diff())
        threshold = diff.mean() + 3 * diff.std()
        data_copy.loc[diff > threshold, 'label'] = 1
    
    anomaly_count = data_copy['label'].sum()
    anomaly_rate = (anomaly_count / len(data_copy)) * 100
    
    print(f"Generated {anomaly_count} anomaly labels ({anomaly_rate:.2f}%)")
    print(f"Normal samples: {len(data_copy) - anomaly_count}")
    
    return data_copy

# Add labels to your data
full_data = add_anomaly_labels(full_data)


# ============================================================================
# SPLIT DATA
# ============================================================================
print("\n" + "="*80)
print("SPLITTING DATA")
print("="*80)

# 80-20 train-test split
split_idx = int(len(full_data) * 0.8)
train_data = full_data.iloc[:split_idx].copy()
test_data = full_data.iloc[split_idx:].copy()

print(f"Training set: {len(train_data)} samples")
print(f"Test set: {len(test_data)} samples")
print(f"Test anomalies: {test_data['label'].sum()} ({test_data['label'].mean()*100:.2f}%)")


# ============================================================================
# ENHANCED DETECTOR CLASS
# ============================================================================
class HAWKDetector:
    def __init__(self, sequence_length=20):
        self.features = ['altitude', 'pitch', 'yaw', 'battery',
                        'velocity_x', 'velocity_y', 'velocity_z', 'gps_drift']
        self.sequence_length = sequence_length
        self.n_features = len(self.features)
        self.scaler = MinMaxScaler()
        self.lstm_model = None
        self.isolation_forest = None
        self.lstm_threshold = None
        self.metrics = {}
        self.history = None
        
    def preprocess(self, data, fit=False):
        feature_data = data[self.features].values
        if fit:
            return self.scaler.fit_transform(feature_data)
        return self.scaler.transform(feature_data)
    
    def create_sequences(self, data):
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:i + self.sequence_length])
        return np.array(sequences, dtype=np.float32)
    
    def build_lstm(self):
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(self.sequence_length, self.n_features),
                 return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu', return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            RepeatVector(self.sequence_length),
            LSTM(32, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(64, activation='relu', return_sequences=True),
            Dropout(0.2),
            Dense(self.n_features)
        ])
        model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
        return model
    
    def train(self, train_data, epochs=30, batch_size=128):
        print("\n" + "="*80)
        print("TRAINING HAWK DETECTION SYSTEM")
        print("="*80)
        
        scaled = self.preprocess(train_data, fit=True)
        sequences = self.create_sequences(scaled)
        
        print(f"\nCreated {len(sequences)} sequences of shape {sequences.shape}")
        
        # Train LSTM
        print("\n[1/2] Training LSTM Autoencoder...")
        self.lstm_model = self.build_lstm()
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
        ]
        
        self.history = self.lstm_model.fit(
            sequences, sequences,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Set threshold
        train_pred = self.lstm_model.predict(sequences, verbose=0)
        train_mae = np.mean(np.abs(sequences - train_pred), axis=(1, 2))
        self.lstm_threshold = np.percentile(train_mae, 95)
        print(f"\nLSTM Threshold: {self.lstm_threshold:.6f}")
        
        # Train Isolation Forest
        print("\n[2/2] Training Isolation Forest...")
        self.isolation_forest = IsolationForest(
            contamination=0.08,
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1
        )
        self.isolation_forest.fit(scaled)
        print("Isolation Forest trained")
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
    
    def detect(self, test_data):
        """Detect anomalies with all methods"""
        scaled = self.preprocess(test_data, fit=False)
        sequences = self.create_sequences(scaled)
        
        # LSTM predictions
        predictions = self.lstm_model.predict(sequences, verbose=0)
        reconstruction_errors = np.mean(np.abs(sequences - predictions), axis=(1, 2))
        
        # Align LSTM predictions
        lstm_anomalies = np.zeros(len(test_data), dtype=bool)
        lstm_scores = np.zeros(len(test_data))
        for i, error in enumerate(reconstruction_errors):
            idx = i + self.sequence_length - 1
            if idx < len(test_data):
                lstm_anomalies[idx] = error > self.lstm_threshold
                lstm_scores[idx] = error
        
        # Isolation Forest
        iso_predictions = self.isolation_forest.predict(scaled)
        iso_anomalies = iso_predictions == -1
        iso_scores = self.isolation_forest.score_samples(scaled)
        
        # Combined methods
        combined_or = lstm_anomalies | iso_anomalies
        combined_and = lstm_anomalies & iso_anomalies
        
        # Voting: Use weighted approach
        vote_count = lstm_anomalies.astype(int) + iso_anomalies.astype(int)
        combined_voting = vote_count >= 1  # At least 1 vote
        
        return {
            'lstm_anomalies': lstm_anomalies,
            'lstm_scores': lstm_scores,
            'iso_anomalies': iso_anomalies,
            'iso_scores': iso_scores,
            'combined_or': combined_or,
            'combined_and': combined_and,
            'combined_voting': combined_voting,
            'vote_count': vote_count
        }
    
    def evaluate_all(self, test_data, results):
        """Calculate all metrics"""
        y_true = test_data['label'].values
        
        self.metrics = {}
        
        # LSTM
        self.metrics['LSTM'] = self._calc_metrics(y_true, results['lstm_anomalies'])
        
        # Isolation Forest
        self.metrics['Isolation_Forest'] = self._calc_metrics(y_true, results['iso_anomalies'])
        
        # Combined OR
        self.metrics['Combined_OR'] = self._calc_metrics(y_true, results['combined_or'])
        
        # Combined AND
        self.metrics['Combined_AND'] = self._calc_metrics(y_true, results['combined_and'])
        
        # Voting
        self.metrics['Voting_System'] = self._calc_metrics(y_true, results['combined_voting'])
        
        return self.metrics
    
    def _calc_metrics(self, y_true, y_pred):
        """Calculate metrics"""
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': cm,
            'TN': cm[0, 0] if cm.shape == (2, 2) else 0,
            'FP': cm[0, 1] if cm.shape == (2, 2) else 0,
            'FN': cm[1, 0] if cm.shape == (2, 2) else 0,
            'TP': cm[1, 1] if cm.shape == (2, 2) else 0
        }
    
    def print_metrics(self):
        """Print metrics table"""
        print("\n" + "="*80)
        print("PERFORMANCE METRICS - ALL MODELS")
        print("="*80)
        
        print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 80)
        
        for model, metrics in self.metrics.items():
            print(f"{model:<20} "
                  f"{metrics['accuracy']:<12.4f} "
                  f"{metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f} "
                  f"{metrics['f1_score']:<12.4f}")
        
        print("\n" + "="*80)
        print("CONFUSION MATRIX DETAILS")
        print("="*80)
        
        for model, metrics in self.metrics.items():
            print(f"\n{model}:")
            print(f"  True Negatives:  {metrics['TN']}")
            print(f"  False Positives: {metrics['FP']}")
            print(f"  False Negatives: {metrics['FN']}")
            print(f"  True Positives:  {metrics['TP']}")
    
    def plot_confusion_matrices(self):
        """Plot all confusion matrices"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold')
        
        models = list(self.metrics.keys())
        
        for idx, model in enumerate(models):
            row, col = idx // 3, idx % 3
            cm = self.metrics[model]['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       ax=axes[row, col], cbar=True,
                       xticklabels=['Normal', 'Anomaly'],
                       yticklabels=['Normal', 'Anomaly'])
            axes[row, col].set_title(f'{model}\nF1: {self.metrics[model]["f1_score"]:.4f}', 
                                    fontsize=12, fontweight='bold')
            axes[row, col].set_xlabel('Predicted')
            axes[row, col].set_ylabel('Actual')
        
        # Remove empty subplot
        fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Confusion matrices saved as 'confusion_matrices.png'")
    
    def plot_metrics_comparison(self):
        """Compare all metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance Metrics Comparison', fontsize=16, fontweight='bold')
        
        models = list(self.metrics.keys())
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
        titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        for idx, (metric, title) in enumerate(zip(metrics_names, titles)):
            row, col = idx // 2, idx % 2
            values = [self.metrics[m][metric] for m in models]
            
            bars = axes[row, col].bar(range(len(models)), values, color=colors, 
                                     alpha=0.8, edgecolor='black', linewidth=2)
            axes[row, col].set_title(title, fontsize=14, fontweight='bold')
            axes[row, col].set_ylabel('Score', fontsize=12)
            axes[row, col].set_ylim([0, 1.1])
            axes[row, col].set_xticks(range(len(models)))
            axes[row, col].set_xticklabels(models, rotation=45, ha='right')
            axes[row, col].grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                axes[row, col].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                   f'{val:.3f}', ha='center', va='bottom', 
                                   fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('metrics_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Metrics comparison saved as 'metrics_comparison.png'")
    
    def plot_before_after_voting(self, test_data, results):
        """Compare before and after voting"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Before vs After Voting System Comparison', 
                    fontsize=18, fontweight='bold')
        
        # Sample 2000 points for visualization
        sample_size = min(2000, len(test_data))
        sample_data = test_data.iloc[:sample_size].copy()
        
        # Plot 1: LSTM Only
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(sample_data.index, sample_data['altitude'], 'b-', alpha=0.4, linewidth=1)
        lstm_idx = np.where(results['lstm_anomalies'][:sample_size])[0]
        if len(lstm_idx) > 0:
            ax1.scatter(lstm_idx, sample_data['altitude'].iloc[lstm_idx],
                       c='red', s=20, label=f'LSTM ({len(lstm_idx)})', zorder=5)
        ax1.set_title('LSTM Autoencoder Only', fontweight='bold')
        ax1.set_ylabel('Altitude')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Isolation Forest Only
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(sample_data.index, sample_data['altitude'], 'b-', alpha=0.4, linewidth=1)
        iso_idx = np.where(results['iso_anomalies'][:sample_size])[0]
        if len(iso_idx) > 0:
            ax2.scatter(iso_idx, sample_data['altitude'].iloc[iso_idx],
                       c='orange', s=20, label=f'ISO ({len(iso_idx)})', zorder=5)
        ax2.set_title('Isolation Forest Only', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Plot 3: Voting System
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(sample_data.index, sample_data['altitude'], 'b-', alpha=0.4, linewidth=1)
        vote_idx = np.where(results['combined_voting'][:sample_size])[0]
        if len(vote_idx) > 0:
            ax3.scatter(vote_idx, sample_data['altitude'].iloc[vote_idx],
                       c='purple', s=20, label=f'Voting ({len(vote_idx)})', zorder=5)
        ax3.set_title('Voting System (Combined)', fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # Plot 4-6: Different features
        features_to_plot = ['pitch', 'battery', 'gps_drift']
        for idx, feature in enumerate(features_to_plot):
            ax = fig.add_subplot(gs[1, idx])
            ax.plot(sample_data.index, sample_data[feature], 'g-', alpha=0.4, linewidth=1)
            if len(vote_idx) > 0:
                ax.scatter(vote_idx, sample_data[feature].iloc[vote_idx],
                          c='purple', s=20, zorder=5)
            ax.set_title(f'{feature.capitalize()} with Anomalies', fontweight='bold')
            ax.set_ylabel(feature.capitalize())
            ax.grid(alpha=0.3)
        
        # Plot 7: F1-Score Comparison
        ax7 = fig.add_subplot(gs[2, :])
        models = ['LSTM', 'Isolation_Forest', 'Voting_System']
        f1_scores = [self.metrics[m]['f1_score'] for m in models]
        bars = ax7.bar(models, f1_scores, color=['red', 'orange', 'purple'], 
                      alpha=0.8, edgecolor='black', linewidth=2)
        ax7.set_title('F1-Score Comparison: Individual vs Voting', 
                     fontsize=14, fontweight='bold')
        ax7.set_ylabel('F1-Score', fontsize=12)
        ax7.set_ylim([0, 1])
        ax7.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, val in zip(bars, f1_scores):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.4f}', ha='center', va='bottom', 
                    fontsize=14, fontweight='bold')
        
        plt.savefig('before_after_voting.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Before/After voting comparison saved as 'before_after_voting.png'")
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('LSTM Autoencoder Training History', fontsize=14, fontweight='bold')
        
        # Loss
        axes[0].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_title('Loss (MSE)')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # MAE
        axes[1].plot(self.history.history['mae'], label='Train MAE', linewidth=2)
        axes[1].plot(self.history.history['val_mae'], label='Val MAE', linewidth=2)
        axes[1].set_title('Mean Absolute Error')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Training history saved as 'training_history.png'")


# ============================================================================
# REAL-TIME SIMULATION
# ============================================================================
def simulate_realtime(detector, test_data, n_samples=500):
    """Real-time simulation"""
    print("\n" + "="*80)
    print("REAL-TIME DETECTION SIMULATION")
    print("="*80)
    
    fig, axes = plt.subplots(4, 1, figsize=(18, 14))
    fig.suptitle('Real-Time HAWK Anomaly Detection', fontsize=16, fontweight='bold')
    
    results = detector.detect(test_data.iloc[:n_samples])
    
    time_idx = range(n_samples)
    anomalies = results['combined_voting'][:n_samples]
    anomaly_idx = np.where(anomalies)[0]
    
    # Altitude
    axes[0].plot(time_idx, test_data['altitude'].iloc[:n_samples], 'b-', linewidth=1.5)
    if len(anomaly_idx) > 0:
        axes[0].scatter(anomaly_idx, test_data['altitude'].iloc[anomaly_idx],
                       c='red', s=50, marker='X', label='Anomaly', zorder=5)
    axes[0].set_title('Altitude Monitoring', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Altitude (m)')
    axes[0].legend(loc='upper right')
    axes[0].grid(alpha=0.3)
    
    # Battery
    axes[1].plot(time_idx, test_data['battery'].iloc[:n_samples], 'g-', linewidth=1.5)
    if len(anomaly_idx) > 0:
        axes[1].scatter(anomaly_idx, test_data['battery'].iloc[anomaly_idx],
                       c='red', s=50, marker='X', zorder=5)
    axes[1].set_title('Battery Level', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Battery (%)')
    axes[1].grid(alpha=0.3)
    
    # Velocities
    axes[2].plot(time_idx, test_data['velocity_x'].iloc[:n_samples], 
                label='Vx', linewidth=1.5, alpha=0.7)
    axes[2].plot(time_idx, test_data['velocity_y'].iloc[:n_samples], 
                label='Vy', linewidth=1.5, alpha=0.7)
    axes[2].plot(time_idx, test_data['velocity_z'].iloc[:n_samples], 
                label='Vz', linewidth=1.5, alpha=0.7)
    axes[2].set_title('Velocity Components', fontweight='bold', fontsize=12)
    axes[2].set_ylabel('Velocity (m/s)')
    axes[2].legend(loc='upper right')
    axes[2].grid(alpha=0.3)
    
    # GPS Drift
    axes[3].plot(time_idx, test_data['gps_drift'].iloc[:n_samples], 'm-', linewidth=1.5)
    if len(anomaly_idx) > 0:
        axes[3].scatter(anomaly_idx, test_data['gps_drift'].iloc[anomaly_idx],
                       c='red', s=50, marker='X', zorder=5)
    axes[3].set_title('GPS Drift', fontweight='bold', fontsize=12)
    axes[3].set_xlabel('Time (samples)')
    axes[3].set_ylabel('Drift (m)')
    axes[3].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('realtime_simulation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nProcessed {n_samples} samples")
    print(f"Detected {len(anomaly_idx)} anomalies")
    print("Real-time simulation saved as 'realtime_simulation.png'")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    
    # Initialize detector
    detector = HAWKDetector(sequence_length=20)
    
    # Train
    detector.train(train_data, epochs=30, batch_size=128)
    
    # Plot training history
    detector.plot_training_history()
    
    # Detect on test data
    print("\n" + "="*80)
    print("DETECTING ANOMALIES ON TEST DATA")
    print("="*80)
    results = detector.detect(test_data)
    
    # Evaluate
    metrics = detector.evaluate_all(test_data, results)
    
    # Print metrics
    detector.print_metrics()
    
    # Visualizations
    detector.plot_confusion_matrices()
    detector.plot_metrics_comparison()
    detector.plot_before_after_voting(test_data, results)
    
    # Real-time simulation
    simulate_realtime(detector, test_data, n_samples=1000)
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    test_results = test_data.copy()
    test_results['predicted_anomaly'] = results['combined_voting']
    test_results['lstm_pred'] = results['lstm_anomalies']
    test_results['iso_pred'] = results['iso_anomalies']
    
    test_results.to_csv('hawk_results.csv', index=False)
    print("Results saved to 'hawk_results.csv'")
    
    print("\n" + "="*80)
    print("ALL PROCESSING COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("1. hawk_results.csv")
    print("2. confusion_matrices.png")
    print("3. metrics_comparison.png")
    print("4. before_after_voting.png")
    print("5. training_history.png")
    print("6. realtime_simulation.png")
