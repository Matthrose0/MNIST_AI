"""
nur noch das aktuell auskommentierte interessant der rest ist schon in main.py
"""
import numpy as np
from typing import List, Dict, Any
from numpy.typing import NDArray
from network.core import NetworkCore


def run_single_experiment(
    config: Dict[str, Any], 
    data: tuple[NDArray[np.float64], NDArray[np.uint8], 
                NDArray[np.float64], NDArray[np.uint8]], 
    runs: int = 3
) -> Dict[str, Any]:
    """Execute single experiment configuration multiple times"""
    X_train, y_train, X_test, y_test = data
    results: Dict[str, Any] = {'config': config, 'runs': [], 'loss_landscape': None}
    
    print(f"\n{config['name']}:")
    print(f"  Architecture from hidden layers: {config['hidden_architecture']}")
    print(f"  Activation: {config['activation']}")
    
    # Execute multiple runs
    for run in range(runs):
        np.random.seed(42 + run)
        results['runs'].append(
            execute_training_run(config, X_train, y_train, X_test, y_test, run)
        )
    
    # Calculate statistics
    results['summary'] = calculate_statistics(results['runs'])
    
    # # Generate loss landscape for best run (optional visualization)
    # best_run_idx = np.argmax([r['accuracy'] for r in results['runs']])
    # if best_run_idx == 0:  # Only for first run to save time
    #     results['loss_landscape'] = generate_loss_visualization(
    #         config, X_test[:100], y_test[:100]
    #     )
    
    return results


def execute_training_run(
    config: Dict[str, Any], 
    X_train: NDArray[np.float64], 
    y_train: NDArray[np.uint8],
    X_test: NDArray[np.float64], 
    y_test: NDArray[np.uint8],
    run_num: int) -> Dict[str, Any]:
    """Execute single training run"""
    print(f"    Run {run_num + 1}...")
    
    # Create network
    network = NetworkCore.create_network(
        input_size=784,
        hidden_sizes=config['hidden_architecture'],
        output_size=10,
        hidden_activation_types=config['activation']
    )
    
    # Train
    train_stats = NetworkCore.train_network(
        network=network,
        X_train=X_train,
        y_train=y_train,
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        optimizer_type=config.get('optimizer', 'adam'),
        debuging=True
    )
    
    # Evaluate
    eval_stats = NetworkCore.evaluate_network(network, X_test, y_test)
    
    print(f"    Run {run_num + 1} Acc: {eval_stats['accuracy']:.2%}, Time: {train_stats['training_time']:.1f}s")
    
    return {
        'run_number': run_num + 1,
        'accuracy': eval_stats['accuracy'],
        'test_loss': eval_stats['test_loss'],
        'confusion_matrix': eval_stats['confusion_matrix'],
        'training_time': train_stats['training_time'],
        'network': network  # Store for potential visualization
    }


def calculate_statistics(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate mean and std for experiment runs"""
    accuracies = [r['accuracy'] for r in runs]
    losses = [r['test_loss'] for r in runs]
    times = [r['training_time'] for r in runs]
    
    return {
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'mean_loss': np.mean(losses),
        'std_loss': np.std(losses),
        'mean_time': np.mean(times)
    }


# def generate_loss_visualization(config: Dict, X: np.ndarray, 
#                                y: np.ndarray) -> np.ndarray:
#     """Generate loss landscape for visualization"""
#     network = NetworkCore.create_network(
#         input_size=784,
#         hidden_sizes=config['architecture'],
#         output_size=10,
#         activation_type=config['activation']
#     )
    
#     # Quick training for visualization
#     NetworkCore.train_network(
#         network, X, y, epochs=50, 
#         learning_rate=config['learning_rate'],
#         verbose=False
#     )
    
#     return NetworkCore.collect_loss_landscape(network, X, y, resolution=15)

# def create_comparison_heatmap(results: List[Dict]):
#     """Create heatmap data for activation function comparison"""
#     print("\n" + "=" * 70)
#     print(" ACTIVATION FUNCTION COMPARISON MATRIX")
#     print("=" * 70)
    
#     # Extract unique architectures and activations
#     architectures = list(set(str(r['config']['architecture']) for r in results))
#     activations = list(set(r['config']['activation'] for r in results))
    
#     # Create accuracy matrix
#     matrix = np.zeros((len(architectures), len(activations)))
    
#     for i, arch in enumerate(architectures):
#         for j, act in enumerate(activations):
#             # Find matching result
#             for r in results:
#                 if (str(r['config']['architecture']) == arch and 
#                     r['config']['activation'] == act):
#                     matrix[i, j] = r['summary']['mean_accuracy']
#                     break
    
#     # Print matrix
#     print(f"\n{'Architecture':<20}", end="")
#     for act in activations:
#         print(f"{act.upper():<15}", end="")
#     print()
#     print("-" * 65)
    
#     for i, arch in enumerate(architectures):
#         print(f"{arch:<20}", end="")
#         for j in range(len(activations)):
#             print(f"{matrix[i, j]:<15.2%}", end="")
#         print()
    
#     # Identify patterns
#     print("\nKey Insights:")
#     best_per_arch = matrix.max(axis=1)
#     best_per_act = matrix.max(axis=0)
    
#     for i, arch in enumerate(architectures):
#         best_act_idx = matrix[i].argmax()
#         print(f"  {arch}: Best with {activations[best_act_idx].upper()}")
    
#     overall_best_act = activations[best_per_act.argmax()]
#     print(f"\nOverall best activation: {overall_best_act.upper()}")