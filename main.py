# Dataformat in MNIST: 0 which number it is, then 1..785 pixeldata (28*28=784)
# working in dataformat float32bit for accurracy
# link to MNIST: later
import os
# use 12 threads for numpy calculation efficiency
# best setup, tested for personal computer
os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"

import numpy as np
import time
import sys
from typing import List, Dict, Any
from numpy.typing import NDArray
from network.core import NetworkCore
from network.datatransfer import DataTransfer

# helper class for logging in file because of long runtime ========================================
class Tee:
    """Write to both console and file"""
    def __init__(self, filename: str):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()
tee = Tee('logs/experiment_log.txt')
sys.stdout = tee


# Execute single experiment =======================================================================
def run_single_experiment(
    config: Dict[str, Any], 
    data: tuple[NDArray[np.float32], NDArray[np.uint8], 
                NDArray[np.float32], NDArray[np.uint8]], 
    runs: int = 5
) -> Dict[str, Any]:
    """Execute single experiment configuration multiple times"""
    X_train, y_train, X_test, y_test = data
    results: Dict[str, Any] = {'config': config, 'runs': [], 'loss_landscape': None}
    
    print(f"\n{config['name']}:")
    print(f"  Architecture from hidden layers: {config['hidden_architecture']}")
    print(f"  Activation: {config['activation']}")
    
    # Execute multiple runs
    for run in range(runs):
        rng = np.random.default_rng(42 + run)
        results['runs'].append(
            execute_training_run(config, X_train, y_train, X_test, y_test, run, rng)
        )
    
    # Calculate statistics and aggregate confusion matrices
    results['summary'] = calculate_statistics(results['runs'])
    results['aggregated_confusion_matrix'] = _aggregate_confusion_matrices(
        [r['confusion_matrix'] for r in results['runs']]
    )
    
    return results

def _aggregate_confusion_matrices(cms: List[NDArray[np.uint64]]) -> NDArray[np.uint64]:
    """Aggregate multiple confusion matrices by summing them"""
    if not cms:
        return np.array([])
    
    aggregated = np.zeros_like(cms[0])
    for cm in cms:
        aggregated += cm
    
    return aggregated

def execute_training_run(
    config: Dict[str, Any], 
    X_train: NDArray[np.float32], 
    y_train: NDArray[np.uint8],
    X_test: NDArray[np.float32], 
    y_test: NDArray[np.uint8],
    run_num: int,
    rng: np.random.Generator | None = None) -> Dict[str, Any]:
    """Execute single training run"""
    print(f"    Run {run_num + 1}...")
    
    # Create network
    network = NetworkCore.create_network(
        input_size=784,
        hidden_sizes=config['hidden_architecture'],
        output_size=10,
        hidden_activation_types=config['activation'],
        rng=rng
    )
    
    # Train
    train_stats = NetworkCore.train_network(
        network=network,
        X_train=X_train,
        y_train=y_train,
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        optimizer_type=config.get('optimizer', 'adam'),
        debugging=True,
        rng=rng
    )
    
    # Evaluate
    eval_stats = NetworkCore.evaluate_network(network, X_test, y_test)
    
    print(f"    Run {run_num + 1} Acc: {eval_stats['accuracy']:.2%}, "
          f"Time: {train_stats['training_time']:.1f}s")
    
    return {
        'run_number': run_num + 1,
        'accuracy': eval_stats['accuracy'],
        'train_accuracy': train_stats['train_accuracy'],
        'test_loss': eval_stats['test_loss'],
        'confusion_matrix': eval_stats['confusion_matrix'],
        'training_time': train_stats['training_time']
    }

def calculate_statistics(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate mean and std for experiment runs"""
    accuracies: List[float] = [r['accuracy'] for r in runs]
    train_accuracies: List[float] = [r['train_accuracy'] for r in runs]
    losses: List[np.float32] = [r['test_loss'] for r in runs]
    times: List[float] = [r['training_time'] for r in runs]
    
    return {
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'mean_train_accuracy': np.mean(train_accuracies),
        'std_train_accuracy': np.std(train_accuracies),
        'mean_loss': np.mean(losses),
        'std_loss': np.std(losses),
        'mean_time': np.mean(times)
    }

# Summary / Analysis ==============================================================================
def print_summary(results: List[Dict[str, Any]]):
    """
    Print formatted summary of all experiments
    
    Args:
        results: List of experiment results from run_single_experiment
    """
    # Preprocess all data once - avoid redundant loops
    stats = _compute_statistics(results)
    
    # Part 1: Individual experiment results
    print("\n" + "=" * 100)
    print("EXPERIMENT RESULTS")
    print("=" * 100)
    print_individual_experiments(results)
    print_experiment_means(stats['by_activation'])
    
    # Part 2: Best configuration
    print("\n" + "=" * 100)
    print("BEST CONFIGURATION")
    print("=" * 100)
    best = max(results, key=lambda x: x['summary']['mean_accuracy'])
    print(f"\n{best['config']['name']}")
    print(f"{best['summary']['mean_accuracy']:.2%} "
          f"(±{best['summary']['std_accuracy']:.2%})")
    
    # Part 3: Analysis (reuse precomputed stats)
    print("\n" + "=" * 100)
    print("ANALYSIS")
    print("=" * 100)
    print_overfitting_and_speed_analysis(stats['by_activation'])
    print_confusion_matrices(results, stats['best_by_activation'])

def _compute_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute all statistics once to avoid redundant loops
    
    Returns dict with:
        - by_activation: stats per activation function
        - best_by_activation: best result per activation
    """
    by_activation: Dict[str, Dict[str, List[float]]] = {}
    best_by_activation = {}
    
    for r in results:
        act = r['config']['activation']
        
        # Initialize storage for this activation
        if act not in by_activation:
            by_activation[act] = {
                'acc': [], 'train_acc': [], 'loss': [], 
                'time': [], 'overfit': []
            }
        
        # Collect all run data
        for run in r['runs']:
            by_activation[act]['acc'].append(run['accuracy'])
            by_activation[act]['train_acc'].append(run['train_accuracy'])
            by_activation[act]['loss'].append(run['test_loss'])
            by_activation[act]['time'].append(run['training_time'])
            by_activation[act]['overfit'].append(
                run['train_accuracy'] - run['accuracy']
            )
        
        # Track best per activation
        if act not in best_by_activation or \
           r['summary']['mean_accuracy'] > \
           best_by_activation[act]['summary']['mean_accuracy']:
            best_by_activation[act] = r
    
    return {
        'by_activation': by_activation,
        'best_by_activation': best_by_activation
    }

def print_individual_experiments(results: List[Dict[str, Any]]):
    """
    Print results grouped by experiment
    
    Output format example:
        Activation      Architecture    Accuracy (±std)         Loss            Time
        LeakyReLU       [64, 32]        75.23% (±1.2%)          0.7234          45.2s
                        [128, 64]       76.45% (±0.9%)          0.6891          52.1s
    """
    # Group results by experiment and activation function
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for r in results:
        group = r['config']['experiment_group']
        act = r['config']['activation']
        
        if group not in grouped:
            grouped[group] = {}
        if act not in grouped[group]:
            grouped[group][act] = []
        
        grouped[group][act].append(r)


    # print
    for group_name, activations in grouped.items():
        print(f"\n{group_name}:")
        print("-" * 100)
        print(f"{'Activation':<16}{'Architecture':<27}{'Accuracy (±std)':<22}"
              f"{'Train Accuracy (±std)':<22}{'Loss':<10}{'Time':<10}")
        
        for act, res_list in activations.items():
            for idx, r in enumerate(res_list):
                s = r['summary']
                arch = str(r['config']['hidden_architecture'])
                acc_str = f"{s['mean_accuracy']:.2%} (±{s['std_accuracy']:.2%})"
                train_acc_str = f"{s['mean_train_accuracy']:.2%} (±{s['std_train_accuracy']:.2%})"
                
                print(f"{act if idx == 0 else '':<16}"
                      f"{arch:<27}"
                      f"{acc_str:<22}"
                      f"{train_acc_str:<22}"
                      f"{s['mean_loss']:<10.4f}"
                      f"{s['mean_time']:<9.1f}s")
            print()

def print_experiment_means(stats: Dict[str, Dict[str, List[float]]]):
    """Print mean values per activation (uses precomputed stats)"""
    print("\n" + "=" * 100)
    print("MEAN PERFORMANCE ACROSS ALL ARCHITECTURES")
    print("=" * 100)
    print(f"{'Activation':<16}{'Test Acc (±std)':<22}{'Train Acc (±std)':<22}"
          f"{'Loss':<12}{'Time [s]':<10}")
    
    for act in sorted(stats.keys()):
        s = stats[act]
        print(f"{act:<16}"
              f"{np.mean(s['acc']):>6.2%} (±{np.std(s['acc']):<6.2%})  "
              f"{np.mean(s['train_acc']):>6.2%} (±{np.std(s['train_acc']):<6.2%})  "
              f"{np.mean(s['loss']):<12.4f}"
              f"{np.mean(s['time']):<9.1f}s")

def print_overfitting_and_speed_analysis(stats: Dict[str, Dict[str, List[float]]]):
    """Analyze overfitting and training speed (uses precomputed stats)"""
    all_activations = sorted(stats.keys())
    
    # Print overfitting analysis
    print("\n" + "=" * 100)
    print("OVERFITTING ANALYSIS")
    print("=" * 100)
    print(f"{'Activation':<16}{'Avg Overfit (±std)':<30}")
    print("-" * 100)
    for act in all_activations:
        avg = np.mean(stats[act]['overfit'])
        std = np.std(stats[act]['overfit'])
        print(f"{act:<16}{avg:>9.2%} (±{std:<6.2%})")
    
    # Print training speed analysis
    print("\n" + "=" * 100)
    print("TRAINING SPEED COMPARISON")
    print("=" * 100)
    print(f"{'Activation':<16}{'Avg Training Time (±std)':<30}")
    print("-" * 100)
    for act in all_activations:
        avg = np.mean(stats[act]['time'])
        std = np.std(stats[act]['time'])
        print(f"{act:<16}{avg:>12.1f}s (±{std:<6.1f}s)")

def print_confusion_matrices(results: List[Dict[str, Any]], 
                            best_by_act: Dict[str, Dict[str, Any]]):
    """Print confusion matrices for top 3 and best per activation"""
    print("\n" + "=" * 100)
    print("CONFUSION MATRICES")
    print("=" * 100)
    
    # Top 3 overall configurations
    print("\nTop 3 Overall Configurations:")
    print("-" * 100)
    top3 = sorted(results, key=lambda x: x['summary']['mean_accuracy'], 
                  reverse=True)[:3]
    
    for i, r in enumerate(top3, 1):
        print(f"\n#{i}: {r['config']['name']} "
              f"(Accuracy: {r['summary']['mean_accuracy']:.4f})")
        print_confusion_matrix(r['aggregated_confusion_matrix'])
    
    # Best configuration per activation (reuse precomputed best)
    print("\n" + "=" * 100)
    print("Best Configuration per Activation Function:")
    print("=" * 100)
    
    for act in sorted(best_by_act.keys()):
        r = best_by_act[act]
        print(f"\n{act}: {r['config']['name']} "
              f"(Accuracy: {r['summary']['mean_accuracy']:.4f})")
        print_confusion_matrix(r['aggregated_confusion_matrix'])

def print_confusion_matrix(cm: NDArray[np.uint64]):
    """Print confusion matrix in readable format with axis labels"""
    if cm.size == 0:
        print("  [Empty confusion matrix]")
        return
    
    # Print title
    print("\n  Confusion Matrix (rows=True labels, cols=Predictions):")
    
    # Print column header
    print("       ", end="")
    for i in range(cm.shape[1]):
        print(f"{i:>5}", end="")
    print()
    
    # Print separator
    print("     " + "-" * (cm.shape[1] * 5 + 1))
    
    # Print matrix rows with row labels
    for i in range(cm.shape[0]):
        print(f"  {i} | ", end="")
        for j in range(cm.shape[1]):
            print(f"{cm[i, j]:>5}", end="")
        print()
    
    # Print per-class accuracy
    print("\n  Per-class accuracy:")
    for i in range(cm.shape[0]):
        if cm[i].sum() > 0:
            acc = cm[i, i] / cm[i].sum()
            print(f"    Class {i}: {acc:.3f}")

# Experiments =====================================================================================
def create_configs() -> List[Dict[str, Any]]:
    """Create experiment configurations"""
    
    # Define base configurations
    architectures = {
        'shallow': [128, 64],
        'deep': [256, 128, 128, 128, 64],
        'wide': [1024]
    }
    
    activations = ['ReLU', 'LeakyReLU', 'ELU', 'Tanh']
    
    # Default parameters for all experiments
    defaults: Dict[str, Any] = {
        'archs': 'shallow',
        'epochs': 100,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'use_reduced_data': False
    }
    
    # Define experiment scenarios (only specify what changes)
    experiments: List[Dict[str, Any]] = [
        {
            'name': 'Baseline',
            'description': 'Standard hyperparameters'
        },
        {
            'name': 'High_LR',
            'description': 'Fast learning, learning rate sensitivity',
            'learning_rate': 0.01
        },
        {
            'name': 'Fast_Convergence',
            'description': 'Few epochs - convergence speed',
            'epochs': 30
        },
        {
            'name': 'Long_Training',
            'description': 'Many epochs - overfitting',
            'epochs': 400
        },
        {
            'name': 'Small_Dataset',
            'description': 'Reduced training data - overfitting test',
            'use_reduced_data': True
        },
        {
            'name': 'Deep_Network',
            'description': 'Deep architecture - gradient flow',
            'archs': 'deep'
        },
        {
            'name': 'Wide_Network',
            'description': 'Wide architecture',
            'archs': 'wide'
        },
        
    ]
    
    # Generate all configurations
    configs: List[Dict[str, Any]] = []
    configs.append({
        'name': 'linear_baseline_shallow',
        'experiment_group': 'no_activation',
        'hidden_architecture': architectures['shallow'],
        'activation': 'Linear',
        'epochs': 100,
        'learning_rate': 0.001,
        'optimizer': 'adam'
    })
    print(f"\nno_activation:")
    print(f"  Baseline without activation functions")
    print(f"  Architecture: {architectures['shallow']}")
    print(f"  LR: 0.001,  Epochs: 100")

    for exp in experiments:
        # Merge defaults with experiment-specific overrides
        exp_params = {**defaults, **exp}
        arch: list[int] = architectures[exp_params.pop('archs')]
        name: str = exp_params.pop('name')
        description: str = exp_params.pop('description')

        print(f"[{exp['name']}] {exp['description']}")
        print(f"\n{name}:")
        print(f"  {description}")
        print(f"  Architecture: {arch}")
        print(f"  LR: {exp_params['learning_rate']}, "
              f"  Epochs: {exp_params['epochs']}")
        
        for activation in activations:
            config: Dict[str, Any] = {
                'name': f"{name.lower()}_{activation}",
                'experiment_group': name,
                'hidden_architecture': arch,
                'activation': activation,
                **exp_params
            }
            configs.append(config)
    
    print(f"\nTotal configurations: {len(configs)}")
    print(f"Experiment groups: {len(experiments)+1}")
    
    return configs

def run_experiments():
    """
    Main execution to run multiple experiments, which are defined in create_configs
    Seed for randomizer is set at experiment creation -> every experiment has the same startseed
    """
    print("=" * 70)
    print(" NEURAL NETWORK - ACTIVATION FUNCTION ANALYSIS")
    print("=" * 70)

    # Load and prepare data
    print("\nLoading MNIST data...")
    full_train_x, full_train_y = DataTransfer.readDataFromCSV('MNIST/mnist_train.csv', 0, 30000) # 3000
    test_x, test_y = DataTransfer.readDataFromCSV('MNIST/mnist_test.csv', 0, 10000) #500
    
    full_train_x: NDArray[np.float32] = full_train_x / 255.0
    test_x: NDArray[np.float32] = test_x / 255.0
    
    reduced_train_x, reduced_train_y = full_train_x[:5000], full_train_y[:5000]

    print(f"  Full training set: {full_train_x.shape}")
    print(f"  Reduced training set: {reduced_train_x.shape}")
    print(f"  Test set: {test_x.shape}")

    # Create configurations
    configs = create_configs()
    total_runs = len(configs) * 5
    t = total_runs * 500
    print(f"\nTotal experiments: {len(configs)} configs × 5 runs = {total_runs}")
    print(f"This will take some time...\nEstimated Time (1 Run=500s): "
          f"{int(t//3600):02d}:{int((t%3600)//60):02d}:{int(t%60):02d} \n")

    # Run experiments
    results: List[Dict[str, Any]] = []
    start = time.time()

    for i, config in enumerate(configs):
        print(f"\n{'=' * 70}")
        print(f"Experiment {i+1}/{len(configs)}: {config['experiment_group']}")
        print(f"  Config: {config['name']}")
        
        # Select dataset based on experiment
        use_reduced: bool = config.get('use_reduced_data', False)
        data = (
            (reduced_train_x, reduced_train_y, test_x, test_y) 
            if use_reduced 
            else (full_train_x, full_train_y, test_x, test_y)
        )
        
        if use_reduced:
            print(f"  Using reduced dataset ({reduced_train_x.shape[0]} samples)")
        
        results.append(run_single_experiment(config, data))


        # Progress update
        if (i + 1) % 4 == 0:
            elapsed = time.time() - start
            remaining = (elapsed / (i + 1)) * (len(configs) - i - 1)
            print(f"\n  Progress: {i+1}/{len(configs)} ({100*(i+1)/len(configs):.1f}%)")
            print(f"  Estimated time remaining: {remaining/60:.1f} minutes")
    
    # Overall summary
    print_summary(results)

    # Final statistics
    total_time = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENTS COMPLETED")
    print(f"  Total time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    print(f"  Average per run: {total_time/len(configs):.1f}s")
    print(f"  Total networks trained: {len(configs) * 5}")
    print("=" * 70)

def train_web_model():
    # not connected to webapp
    
    np.random.default_rng(42)
    # Load data
    print("\nLoading MNIST data...")
    data = (
        *DataTransfer.readDataFromCSV('MNIST/mnist_train.csv', 0, 50000),
        *DataTransfer.readDataFromCSV('MNIST/mnist_test.csv', 0, 10000)
    )
    print(f"  Train: {data[0].shape}, Test: {data[2].shape}")

    # Run experiment
    start = time.time()

    # Create network
    network = NetworkCore.create_network(
        input_size=784,
        hidden_sizes=[128, 128, 64],
        output_size=10,
        hidden_activation_types='ReLU'
    )
    
    # Train
    train_stats = NetworkCore.train_network(
        network=network,
        X_train=data[0]/255.0,
        y_train=data[1],
        epochs=80,
        learning_rate=0.001,
        optimizer_type='adam',
        debugging=True
    )
    
    # Evaluate
    eval_stats = NetworkCore.evaluate_network(network, data[2]/255.0, data[3])
    
    # Summary
    print(f"Acc: {eval_stats['accuracy']:.2%}")
    print(f"Train Acc: {train_stats['train_accuracy']:.2%}")
    print(f"Time: {train_stats['training_time']:.1f}s")
    print(f"Loss: {eval_stats['test_loss']:.2f}")
    print(f"Confusion Matrix:\n{eval_stats['confusion_matrix']}")

    # export
    DataTransfer.save_network("MNIST/network0.csv",network.get_layers(), network.get_activations_as_string())
    
    print(f"Total experiment time: {time.time() - start:.1f}s")
    print("=" * 70)

if __name__ == "__main__":
    # run_experiments()
    train_web_model()
    


sys.stdout = sys.__stdout__
tee.close()