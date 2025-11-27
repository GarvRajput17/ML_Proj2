"""
Multi-Layer Perceptron (MLP) Neural Network for COPD Risk Prediction
Implemented from scratch with forward and backward propagation
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


def _f1_score_auto(y_true, y_pred):
    """Compute F1-score with automatic averaging (binary vs macro)."""
    unique_classes = np.unique(y_true)
    average = "binary" if len(unique_classes) <= 2 else "macro"
    return f1_score(y_true, y_pred, average=average)


class MLPNeuralNetwork:
    """
    Multi-Layer Perceptron Neural Network with forward and backward propagation
    """
    
    def __init__(self, layers_sizes, learning_rate=0.001, activation='relu', dropout_rate=0.3):
        """
        Initialize the neural network
        
        Parameters:
        -----------
        layers_sizes : list
            List of layer sizes, e.g., [input_size, hidden1, hidden2, output_size]
        learning_rate : float
            Learning rate for gradient descent
        activation : str
            Activation function: 'relu' or 'sigmoid'
        dropout_rate : float
            Dropout rate for regularization (0.0 to 1.0)
        """
        self.layers_sizes = layers_sizes
        self.learning_rate = learning_rate
        self.activation = activation
        self.dropout_rate = dropout_rate
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Initialize weights using He initialization for ReLU, Xavier for sigmoid
        for i in range(len(layers_sizes) - 1):
            if activation == 'relu':
                # He initialization for ReLU
                w = np.random.randn(layers_sizes[i], layers_sizes[i+1]) * np.sqrt(2.0 / layers_sizes[i])
            else:
                # Xavier initialization for sigmoid
                w = np.random.randn(layers_sizes[i], layers_sizes[i+1]) * np.sqrt(1.0 / layers_sizes[i])
            
            b = np.zeros((1, layers_sizes[i+1]))
            
            self.weights.append(w)
            self.biases.append(b)
        
        # Store activations and z values for backpropagation
        self.activations = []
        self.z_values = []
        self.dropout_masks = []  # Store dropout masks for backprop
    
    def _sigmoid(self, x):
        """Sigmoid activation function"""
        # Clip to avoid overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def _relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def _sigmoid_derivative(self, x):
        """Derivative of sigmoid"""
        s = self._sigmoid(x)
        return s * (1 - s)
    
    def _relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    def _activate(self, x):
        """Apply activation function"""
        if self.activation == 'relu':
            return self._relu(x)
        else:
            return self._sigmoid(x)
    
    def _activate_derivative(self, x):
        """Apply derivative of activation function"""
        if self.activation == 'relu':
            return self._relu_derivative(x)
        else:
            return self._sigmoid_derivative(x)
    
    def forward_propagation(self, X, training=True):
        """
        Forward propagation through the network
        
        Parameters:
        -----------
        X : numpy array
            Input data (n_samples, n_features)
        training : bool
            Whether in training mode (for dropout)
        
        Returns:
        --------
        numpy array: Output predictions
        """
        # Store activations and z values for backpropagation
        self.activations = [X]
        self.z_values = []
        self.dropout_masks = []
        
        # Forward pass through each layer
        current_input = X
        for i in range(len(self.weights)):
            # Linear transformation: z = X * W + b
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            # Apply activation (except for output layer which uses sigmoid for binary classification)
            if i == len(self.weights) - 1:
                # Output layer: use sigmoid for binary classification
                a = self._sigmoid(z)
            else:
                # Hidden layers: use specified activation
                a = self._activate(z)
                
                # Apply dropout during training (not on output layer)
                if training and self.dropout_rate > 0:
                    # Create dropout mask
                    dropout_mask = (np.random.rand(*a.shape) > self.dropout_rate).astype(float)
                    dropout_mask /= (1 - self.dropout_rate)  # Scale to maintain expected value
                    self.dropout_masks.append(dropout_mask)
                    a = a * dropout_mask
            
            self.activations.append(a)
            current_input = a
        
        return current_input
    
    def backward_propagation(self, X, y, output, class_weights=None):
        """
        Backward propagation to compute gradients
        
        Parameters:
        -----------
        X : numpy array
            Input data
        y : numpy array
            True labels
        output : numpy array
            Predicted output from forward propagation
        class_weights : dict or None
            Class weights for handling imbalance
        
        Returns:
        --------
        tuple: (weight_gradients, bias_gradients)
        """
        m = X.shape[0]  # Number of samples
        
        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error (binary cross-entropy loss derivative)
        # For sigmoid output: dL/dz = output - y
        dz = output - y.reshape(-1, 1)
        
        # Apply class weights if provided
        if class_weights is not None:
            weight_0 = class_weights.get(0, 1.0)
            weight_1 = class_weights.get(1, 1.0)
            weights = np.where(y.reshape(-1, 1) == 1, weight_1, weight_0)
            dz = dz * weights
        
        # Backpropagate through each layer
        for i in range(len(self.weights) - 1, -1, -1):
            # Compute gradients for weights and biases
            dW[i] = (1 / m) * np.dot(self.activations[i].T, dz)
            db[i] = (1 / m) * np.sum(dz, axis=0, keepdims=True)
            
            # Compute error for previous layer (if not input layer)
            if i > 0:
                # Error propagation: dz_prev = dz * W^T * activation_derivative
                dz = np.dot(dz, self.weights[i].T)
                dz = dz * self._activate_derivative(self.z_values[i-1])
                
                # Apply dropout mask if dropout was used
                if self.dropout_rate > 0 and len(self.dropout_masks) > 0:
                    mask_idx = i - 1
                    if mask_idx < len(self.dropout_masks):
                        dz = dz * self.dropout_masks[mask_idx]
        
        return dW, db
    
    def update_parameters(self, dW, db):
        """
        Update weights and biases using gradient descent
        
        Parameters:
        -----------
        dW : list
            Weight gradients
        db : list
            Bias gradients
        """
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def compute_loss(self, y_true, y_pred, class_weights=None):
        """
        Compute weighted binary cross-entropy loss
        
        Parameters:
        -----------
        y_true : numpy array
            True labels
        y_pred : numpy array
            Predicted probabilities
        class_weights : dict or None
            Class weights for handling imbalance {0: weight0, 1: weight1}
        
        Returns:
        --------
        float: Loss value
        """
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        y_true = y_true.reshape(-1, 1)
        
        # Binary cross-entropy loss
        if class_weights is not None:
            # Apply class weights
            weight_0 = class_weights.get(0, 1.0)
            weight_1 = class_weights.get(1, 1.0)
            weights = np.where(y_true == 1, weight_1, weight_0)
            loss = -np.mean(weights * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
        else:
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def fit(self, X, y, epochs=100, batch_size=32, validation_data=None, verbose=True, 
            class_weights=None, learning_rate_decay=0.95, patience=10):
        """
        Train the neural network
        
        Parameters:
        -----------
        X : numpy array
            Training features
        y : numpy array
            Training labels
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for mini-batch gradient descent
        validation_data : tuple
            (X_val, y_val) for validation
        verbose : bool
            Whether to print training progress
        class_weights : dict
            Class weights for handling imbalance {0: weight0, 1: weight1}
        learning_rate_decay : float
            Learning rate decay factor per epoch
        patience : int
            Early stopping patience (epochs without improvement)
        """
        X = np.array(X)
        y = np.array(y)
        
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        train_losses = []
        val_losses = []
        val_accuracies = []
        val_f1_scores = []
        
        best_val_f1 = 0
        best_weights = None
        best_biases = None
        patience_counter = 0
        initial_lr = self.learning_rate
        
        for epoch in range(epochs):
            # Learning rate decay
            if epoch > 0 and learning_rate_decay < 1.0:
                self.learning_rate = initial_lr * (learning_rate_decay ** epoch)
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            # Mini-batch gradient descent
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward propagation (training mode with dropout)
                output = self.forward_propagation(X_batch, training=True)
                
                # Compute loss
                batch_loss = self.compute_loss(y_batch, output, class_weights)
                epoch_loss += batch_loss
                
                # Backward propagation
                dW, db = self.backward_propagation(X_batch, y_batch, output, class_weights)
                
                # Update parameters
                self.update_parameters(dW, db)
            
            avg_loss = epoch_loss / n_batches
            train_losses.append(avg_loss)
            
            # Validation
            if validation_data is not None:
                X_val, y_val = validation_data
                val_output = self.forward_propagation(X_val, training=False)  # No dropout in validation
                val_loss = self.compute_loss(y_val, val_output, class_weights)
                val_losses.append(val_loss)
                
                val_pred = (val_output > 0.5).astype(int).flatten()
                val_acc = np.mean(val_pred == y_val)
                val_accuracies.append(val_acc)
                
                # Calculate F1 score for early stopping
                val_f1 = _f1_score_auto(y_val, val_pred)
                val_f1_scores.append(val_f1)
                
                # Early stopping based on F1 score
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if verbose and (epoch + 1) % 5 == 0:  # Print more frequently
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - Val F1: {val_f1:.4f} - LR: {self.learning_rate:.6f}")
                
                # Early stopping
                if patience_counter >= patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                    # Restore best weights
                    if best_weights is not None:
                        self.weights = best_weights
                        self.biases = best_biases
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - LR: {self.learning_rate:.6f}")
        
        return {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'val_accuracy': val_accuracies,
            'val_f1': val_f1_scores
        }
    
    def predict(self, X, threshold=0.5):
        """
        Make predictions
        
        Parameters:
        -----------
        X : numpy array
            Input features
        threshold : float
            Classification threshold (default 0.5)
        
        Returns:
        --------
        numpy array: Binary predictions (0 or 1)
        """
        X = np.array(X)
        output = self.forward_propagation(X, training=False)  # No dropout during inference
        predictions = (output > threshold).astype(int).flatten()
        return predictions
    
    def predict_proba(self, X):
        """
        Predict probabilities
        
        Parameters:
        -----------
        X : numpy array
            Input features
        
        Returns:
        --------
        numpy array: Predicted probabilities
        """
        X = np.array(X)
        output = self.forward_propagation(X, training=False)  # No dropout during inference
        return output.flatten()


def load_data(data_dir: str = None):
    """Load preprocessed feature matrices and targets."""
    print("\n" + "=" * 80)
    print("LOADING DATA FOR MLP NEURAL NETWORK")
    print("=" * 80)

    # Get the script's directory and resolve paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    if data_dir is None:
        data_dir = os.path.join(project_root, "chronic-obstructive")
    
    X_train = pd.read_csv(os.path.join(data_dir, "processed_train_features.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "train_target.csv"))["has_copd_risk"]
    X_test = pd.read_csv(os.path.join(data_dir, "processed_test_features.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    test_patient_ids = test_df["patient_id"]

    # Basic cleanup so the model does not crash.
    if X_train.isnull().values.any():
        print("Detected NaNs in training features -> filling with 0.")
        X_train = X_train.fillna(0)
    if X_test.isnull().values.any():
        print("Detected NaNs in test features -> filling with 0.")
        X_test = X_test.fillna(0)

    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)

    print(f"Training samples: {X_train.shape[0]} | Features: {X_train.shape[1]}")
    print(f"Test samples: {X_test.shape[0]}")
    return X_train, y_train, X_test, test_patient_ids


def train_mlp_ensemble(X_train: pd.DataFrame, y_train: pd.Series):
    """Train multiple MLP models with different architectures and use weighted ensemble."""
    print("\n" + "=" * 80)
    print("TRAINING MLP ENSEMBLE WITH WEIGHTED VOTING")
    print("=" * 80)

    X_np = X_train.to_numpy(dtype=np.float32)
    y_np = y_train.to_numpy(dtype=np.int64)

    # Calculate class weights for imbalanced data (63:37 split)
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_np)
    class_weights_vals = compute_class_weight('balanced', classes=classes, y=y_np)
    class_weights = {int(c): float(w) for c, w in zip(classes, class_weights_vals)}
    print(f"\nClass distribution - Class 0: {np.sum(y_np == 0)}, Class 1: {np.sum(y_np == 1)}")
    print(f"Class weights: {class_weights}")

    # Split data for validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_np,
        y_np,
        test_size=0.2,
        stratify=y_np,
        random_state=42,
    )

    print(f"\nTraining set: {X_tr.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Input features: {X_tr.shape[1]} (239 engineered features)")

    input_size = X_tr.shape[1]
    
    # Create multiple MLP models with different architectures for diversity
    model_configs = [
        {"name": "mlp_small", "layers": [input_size, 128, 64, 1], "lr": 0.002, "dropout": 0.2},
        {"name": "mlp_medium", "layers": [input_size, 256, 128, 1], "lr": 0.0015, "dropout": 0.25},
        {"name": "mlp_deep", "layers": [input_size, 128, 64, 32, 1], "lr": 0.001, "dropout": 0.3},
    ]
    
    models = []
    model_scores = []
    
    print("\n" + "=" * 80)
    print("TRAINING INDIVIDUAL MLP MODELS")
    print("=" * 80)
    
    for config in model_configs:
        print(f"\n--- Training {config['name']} ---")
        print(f"Architecture: {config['layers']}")
        print(f"Learning rate: {config['lr']}, Dropout: {config['dropout']}")
        
        model = MLPNeuralNetwork(
            layers_sizes=config['layers'],
            learning_rate=config['lr'],
            activation='relu',
            dropout_rate=config['dropout']
        )
        
        model.fit(
            X_tr, 
            y_tr,
            epochs=50,
            batch_size=512,
            validation_data=(X_val, y_val),
            verbose=False,  # Less verbose for ensemble
            class_weights=class_weights,
            learning_rate_decay=0.95,
            patience=8
        )
        
        # Evaluate on validation
        val_pred = model.predict(X_val)
        val_f1 = _f1_score_auto(y_val, val_pred)
        val_acc = accuracy_score(y_val, val_pred)
        
        models.append((config['name'], model))
        model_scores.append(val_f1)
        
        print(f"Validation F1: {val_f1:.4f}, Accuracy: {val_acc:.4f}")
    
    # Calculate weights based on F1 scores
    print("\n" + "=" * 80)
    print("CALCULATING WEIGHTED ENSEMBLE")
    print("=" * 80)
    
    total_score = sum(model_scores)
    if total_score > 0:
        # Normalize and square to emphasize better models
        weights = [score / total_score for score in model_scores]
        weights = [w**2 for w in weights]
        weights = [w / sum(weights) for w in weights]
    else:
        weights = [1.0 / len(model_scores)] * len(model_scores)
    
    print(f"\nModel weights (based on F1-scores):")
    for (name, _), score, weight in zip(models, model_scores, weights):
        print(f"  {name:15s} - F1: {score:.4f}, Weight: {weight:.3f}")
    
    # Test different ensemble methods
    print("\n" + "=" * 80)
    print("TESTING ENSEMBLE METHODS")
    print("=" * 80)
    
    # 1. Equal weights
    print("\n1. Equal weights ensemble...")
    equal_proba = np.mean([model.predict_proba(X_val) for _, model in models], axis=0)
    equal_pred = (equal_proba > 0.5).astype(int)
    equal_f1 = _f1_score_auto(y_val, equal_pred)
    print(f"   F1-score: {equal_f1:.4f}")
    
    # 2. Weighted ensemble
    print("\n2. Weighted ensemble...")
    weighted_proba = np.sum([w * model.predict_proba(X_val) for (_, model), w in zip(models, weights)], axis=0)
    weighted_pred = (weighted_proba > 0.5).astype(int)
    weighted_f1 = _f1_score_auto(y_val, weighted_pred)
    print(f"   F1-score: {weighted_f1:.4f}")
    
    # 3. Optimized blending (grid search)
    print("\n3. Optimized probability blending...")
    probas = [model.predict_proba(X_val) for _, model in models]
    best_blend_f1 = 0
    best_blend_weights = None
    
    # Grid search for best blend weights
    for w1 in np.arange(0.2, 0.9, 0.1):
        for w2 in np.arange(0.1, 0.8, 0.1):
            w3 = 1.0 - w1 - w2
            if w3 > 0:
                blend_proba = w1 * probas[0] + w2 * probas[1] + w3 * probas[2]
                blend_pred = (blend_proba > 0.5).astype(int)
                blend_f1 = _f1_score_auto(y_val, blend_pred)
                if blend_f1 > best_blend_f1:
                    best_blend_f1 = blend_f1
                    best_blend_weights = (w1, w2, w3)
    
    print(f"   Best blend F1: {best_blend_f1:.4f}")
    print(f"   Best weights: {[f'{w:.2f}' for w in best_blend_weights]}")
    
    # Select best method
    ensemble_results = [
        ("Equal weights", None, equal_f1),
        ("Weighted (F1-based)", weights, weighted_f1),
        ("Optimized blending", best_blend_weights, best_blend_f1),
    ]
    
    best_name, best_weights, best_f1 = max(ensemble_results, key=lambda x: x[2])
    
    print("\n" + "=" * 80)
    print(f"BEST ENSEMBLE METHOD: {best_name} (F1: {best_f1:.4f})")
    print("=" * 80)
    
    # Create ensemble class
    class MLPEnsemble:
        def __init__(self, models, weights):
            self.models = models
            self.weights = weights
        
        def fit(self, X, y):
            # Models already trained
            return self
        
        def predict(self, X):
            if self.weights is None:
                # Equal weights
                proba = np.mean([model.predict_proba(X) for _, model in self.models], axis=0)
            else:
                # Weighted
                proba = np.sum([w * model.predict_proba(X) for (_, model), w in zip(self.models, self.weights)], axis=0)
            return (proba > 0.5).astype(int)
    
    # Retrain all models on full dataset
    print("\nRetraining all models on full dataset...")
    final_models = []
    for config in model_configs:
        model = MLPNeuralNetwork(
            layers_sizes=config['layers'],
            learning_rate=config['lr'],
            activation='relu',
            dropout_rate=config['dropout']
        )
        model.fit(
            X_np, 
            y_np,
            epochs=50,
            batch_size=512,
            validation_data=None,
            verbose=False,
            class_weights=class_weights,
            learning_rate_decay=0.95,
            patience=8
        )
        final_models.append((config['name'], model))
    
    # Create final ensemble with best weights
    if best_weights is None:
        final_weights = None  # Equal weights
    else:
        final_weights = best_weights
    
    final_ensemble = MLPEnsemble(final_models, final_weights)
    return final_ensemble


def train_mlp(X_train: pd.DataFrame, y_train: pd.Series) -> MLPNeuralNetwork:
    """Train an MLP Neural Network with validation, optimized for 239 features and class imbalance."""
    # Use ensemble by default
    return train_mlp_ensemble(X_train, y_train)


def make_predictions(model, X_test: pd.DataFrame):
    """Return class predictions and probabilities for the test set."""
    X_test_np = X_test.to_numpy(dtype=np.float32)
    preds = model.predict(X_test_np)
    return preds, None


def create_submission(
    patient_ids: pd.Series,
    predictions: np.ndarray,
    output_path: str = None,
):
    """Create submission matching sample_submission format."""
    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_path = os.path.join(project_root, "chronic-obstructive", "submission_mlp.csv")
    
    submission = pd.DataFrame(
        {
            "patient_id": patient_ids.values,
            "has_copd_risk": predictions.astype(int),
        }
    )
    submission.to_csv(output_path, index=False)
    print(f"\n✓ Submission saved to {output_path}")
    print(submission.head())
    return submission


def main():
    X_train, y_train, X_test, patient_ids = load_data()
    model = train_mlp(X_train, y_train)
    predictions, _ = make_predictions(model, X_test)
    create_submission(patient_ids, predictions)


if __name__ == "__main__":
    main()

