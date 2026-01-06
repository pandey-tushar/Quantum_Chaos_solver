"""
Training module for the Lorenz quantum circuit solver.

Implements training loop with Adam optimizer, monitoring, and checkpointing.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
import json
from datetime import datetime


class AdamOptimizer:
    """
    Improved Adam optimizer with adaptive momentum and LR decay.
    
    When loss > threshold and gradient is small, applies extra momentum
    to escape local minima. Also includes LR decay to prevent oscillations.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        adaptive_momentum: bool = True,
        loss_threshold: float = 1.0,
        grad_threshold: float = 0.01,
        lr_decay: float = 0.995,  # Decay factor per step
        grad_clip_percentile: float = 99.0,  # Adaptive clipping at 99th percentile
        grad_clip_value: float = None  # Fixed clip value (if None, uses adaptive)
    ):
        """
        Initialize Adam optimizer with adaptive momentum and LR decay.
        
        Args:
            learning_rate: Initial step size
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            epsilon: Small constant for numerical stability
            adaptive_momentum: Enable adaptive momentum boost
            loss_threshold: Loss threshold for momentum boost
            grad_threshold: Gradient norm threshold (relative)
            lr_decay: Learning rate decay factor (applied each step)
            grad_clip_percentile: Percentile for adaptive gradient clipping (default: 99.0)
            grad_clip_value: Fixed gradient clip value (if None, uses adaptive clipping)
        """
        self.initial_lr = learning_rate
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.adaptive_momentum = adaptive_momentum
        self.loss_threshold = loss_threshold
        self.grad_threshold = grad_threshold
        self.lr_decay = lr_decay
        self.grad_clip_percentile = grad_clip_percentile
        self.grad_clip_value = grad_clip_value
        
        # Initialize moment estimates
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time step
        
        # Track loss and gradients for adaptive momentum
        self.prev_loss = None
        self.prev_grad_norm = None
        
        # Track gradient norms for adaptive clipping
        self.grad_norm_history = []
    
    def step(self, theta: np.ndarray, grad: np.ndarray, loss: float = None) -> np.ndarray:
        """
        Perform one optimization step with adaptive momentum and gradient clipping.
        
        Args:
            theta: Current parameters
            grad: Gradient
            loss: Current loss (for adaptive momentum)
        
        Returns:
            Updated parameters
        """
        # Initialize moments on first step
        if self.m is None:
            self.m = np.zeros_like(theta)
            self.v = np.zeros_like(theta)
        
        self.t += 1
        
        # Compute gradient norm
        grad_norm = np.linalg.norm(grad)
        
        # Adaptive gradient clipping
        if self.grad_clip_value is not None:
            # Use fixed clipping value
            max_grad_norm = self.grad_clip_value
        else:
            # Adaptive clipping based on gradient history
            self.grad_norm_history.append(grad_norm)
            
            # Use percentile-based clipping after warmup
            if len(self.grad_norm_history) > 10:
                max_grad_norm = np.percentile(self.grad_norm_history, self.grad_clip_percentile)
                # But don't clip too aggressively - set minimum threshold
                max_grad_norm = max(max_grad_norm, np.sqrt(len(theta)) * 2)  # At least 2×sqrt(n_params)
            else:
                # During warmup, use parameter-count-based clipping
                max_grad_norm = np.sqrt(len(theta)) * 5  # 5×sqrt(n_params)
            
            # Keep history reasonable size
            if len(self.grad_norm_history) > 100:
                self.grad_norm_history = self.grad_norm_history[-100:]
        
        # Apply clipping if needed
        if grad_norm > max_grad_norm:
            grad = grad * (max_grad_norm / grad_norm)
            grad_norm = max_grad_norm
        
        # Check if we need momentum boost
        boost_momentum = False
        if self.adaptive_momentum and loss is not None:
            # Apply momentum boost if loss is high and gradient is small
            avg_grad = grad_norm / (np.sqrt(len(theta)) + 1e-8)
            if loss > self.loss_threshold and avg_grad < self.grad_threshold:
                boost_momentum = True
        
        # Adaptive beta1 for momentum boost
        beta1 = self.beta1
        if boost_momentum:
            beta1 = min(0.95, self.beta1 + 0.05)  # Increase momentum
        
        # Update biased first moment estimate
        self.m = beta1 * self.m + (1 - beta1) * grad
        
        # Update biased second moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - beta1 ** self.t)
        
        # Compute bias-corrected second moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Adaptive learning rate with decay
        lr = self.learning_rate
        if boost_momentum:
            lr *= 1.5  # Boost during momentum
        
        # Apply LR decay
        self.learning_rate = self.learning_rate * self.lr_decay
        
        # Update parameters
        theta_new = theta - lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return theta_new


class TrainingMonitor:
    """Monitor and log training progress."""
    
    def __init__(self):
        """Initialize training monitor."""
        self.history = {
            'iteration': [],
            'loss_total': [],
            'loss_diff': [],
            'loss_boundary': [],
            'grad_norm': [],
            'time': []
        }
        self.start_time = None
    
    def start(self):
        """Start timing."""
        self.start_time = datetime.now()
    
    def log(
        self,
        iteration: int,
        loss_total: float,
        loss_diff: float,
        loss_boundary: float,
        grad_norm: float
    ):
        """
        Log training metrics.
        
        Args:
            iteration: Current iteration
            loss_total: Total loss
            loss_diff: Differential loss
            loss_boundary: Boundary loss
            grad_norm: Gradient norm
        """
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        self.history['iteration'].append(iteration)
        self.history['loss_total'].append(float(loss_total))
        self.history['loss_diff'].append(float(loss_diff))
        self.history['loss_boundary'].append(float(loss_boundary))
        self.history['grad_norm'].append(float(grad_norm))
        self.history['time'].append(elapsed)
    
    def get_history(self) -> Dict:
        """Get training history."""
        return self.history
    
    def print_progress(
        self,
        iteration: int,
        loss_total: float,
        loss_diff: float,
        loss_boundary: float,
        grad_norm: float
    ):
        """Print training progress."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"Iter {iteration:4d} | "
              f"Loss: {loss_total:10.4f} | "
              f"L_diff: {loss_diff:8.4f} | "
              f"L_bnd: {loss_boundary:8.4f} | "
              f"||∇||: {grad_norm:8.4f} | "
              f"Time: {elapsed:6.1f}s")


def train_lorenz_circuit(
    circuit_obj,
    loss_fn,
    n_iterations: int = 100,
    learning_rate: float = 0.01,
    print_every: int = 10,
    save_every: int = 50,
    results_dir: Optional[Path] = None,
    random_seed: int = 42
) -> Tuple[np.ndarray, Dict]:
    """
    Train the Lorenz quantum circuit.
    
    Args:
        circuit_obj: LorenzQuantumCircuit instance
        loss_fn: LorenzPhysicsLoss instance
        n_iterations: Number of training iterations
        learning_rate: Learning rate for Adam optimizer
        print_every: Print progress every N iterations
        save_every: Save checkpoint every N iterations
        results_dir: Directory to save results
        random_seed: Random seed for parameter initialization
    
    Returns:
        Tuple of (best_parameters, training_history)
    """
    # Initialize parameters
    theta = circuit_obj.initialize_parameters(seed=random_seed)
    print(f"Initialized {len(theta)} parameters")
    
    # Create optimizer
    optimizer = AdamOptimizer(learning_rate=learning_rate)
    
    # Create monitor
    monitor = TrainingMonitor()
    monitor.start()
    
    # Setup results directory
    if results_dir is None:
        results_dir = Path("results")
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Track best parameters
    best_loss = float('inf')
    best_theta = theta.copy()
    
    print()
    print("=" * 80)
    print("TRAINING LORENZ QUANTUM CIRCUIT")
    print("=" * 80)
    print(f"Parameters: {len(theta)}")
    print(f"Time points: {len(loss_fn.t_points)}")
    print(f"Learning rate: {learning_rate}")
    print(f"Iterations: {n_iterations}")
    print("=" * 80)
    print()
    
    # Training loop
    for iteration in range(n_iterations):
        # Compute loss and components
        loss_total, loss_diff, loss_boundary = loss_fn.compute_total_loss(
            theta, return_components=True
        )
        
        # Compute gradient
        grad = loss_fn.compute_gradient(theta)
        grad_norm = np.linalg.norm(grad)
        
        # Log metrics
        monitor.log(iteration, loss_total, loss_diff, loss_boundary, grad_norm)
        
        # Print progress
        if iteration % print_every == 0:
            monitor.print_progress(iteration, loss_total, loss_diff, loss_boundary, grad_norm)
        
        # Update best parameters
        if loss_total < best_loss:
            best_loss = loss_total
            best_theta = theta.copy()
        
        # Save checkpoint
        if (iteration + 1) % save_every == 0:
            checkpoint_file = results_dir / f"checkpoint_iter_{iteration+1}.npz"
            np.savez(
                checkpoint_file,
                theta=theta,
                iteration=iteration,
                loss_total=loss_total,
                loss_diff=loss_diff,
                loss_boundary=loss_boundary
            )
        
        # Update parameters
        theta = optimizer.step(theta, grad)
    
    # Final evaluation
    loss_total, loss_diff, loss_boundary = loss_fn.compute_total_loss(
        theta, return_components=True
    )
    grad = loss_fn.compute_gradient(theta)
    grad_norm = np.linalg.norm(grad)
    
    monitor.log(n_iterations, loss_total, loss_diff, loss_boundary, grad_norm)
    
    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Final loss: {loss_total:.6f}")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Improvement: {(monitor.history['loss_total'][0] - best_loss):.6f}")
    print("=" * 80)
    
    # Save final results
    final_file = results_dir / "final_parameters.npz"
    np.savez(
        final_file,
        theta=best_theta,
        final_theta=theta,
        best_loss=best_loss,
        final_loss=loss_total
    )
    
    # Save training history
    history_file = results_dir / "training_history.json"
    with open(history_file, 'w') as f:
        json.dump(monitor.get_history(), f, indent=2)
    
    print(f"\nResults saved to: {results_dir}")
    print(f"  - {final_file.name}")
    print(f"  - {history_file.name}")
    
    return best_theta, monitor.get_history()


def load_checkpoint(checkpoint_file: Path) -> Dict:
    """
    Load training checkpoint.
    
    Args:
        checkpoint_file: Path to checkpoint file
    
    Returns:
        Dictionary with checkpoint data
    """
    data = np.load(checkpoint_file)
    return {
        'theta': data['theta'],
        'iteration': int(data['iteration']),
        'loss_total': float(data['loss_total']),
        'loss_diff': float(data['loss_diff']),
        'loss_boundary': float(data['loss_boundary'])
    }


def load_training_results(results_dir: Path) -> Tuple[np.ndarray, Dict]:
    """
    Load final training results.
    
    Args:
        results_dir: Directory containing results
    
    Returns:
        Tuple of (best_parameters, training_history)
    """
    results_dir = Path(results_dir)
    
    # Load parameters
    params_file = results_dir / "final_parameters.npz"
    data = np.load(params_file)
    best_theta = data['theta']
    
    # Load history
    history_file = results_dir / "training_history.json"
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    return best_theta, history

