import time
import numpy as np
from typing import List, Dict, Type, Optional, Any
from dataclasses import dataclass
from .components import *
from .datatransfer import DataTransfer

@dataclass
class NetworkLayer:
    layer: Layer
    activation: Activations
@dataclass
class Network:
    """
    Represents a neural network with its architecture and layers.
    
    Attributes:
        architecture (List[int]): 
            List of integers representing the architecture of the network.
            Each integer represents the number of neurons in a layer.
        layers (List[NetworkLayer]): List of NetworkLayer objects representing the layers of the network.
    """
    architecture: List[int]
    layers: List[NetworkLayer]

    def get_layers(self) -> List[Layer]:
        # Extract only the Layer objects from NetworkLayer
        return [nl.layer for nl in self.layers]
    def get_activations_as_string(self) -> List[str]:
        return [NetworkCore.get_activation_name(nl.activation) for nl in self.layers]

class NetworkCore:
    activation_map: Dict[str, Type[Activations]] = {
        'Linear': ActivationLinear,
        'ReLU': ActivationReLU,
        'LeakyReLU': ActivationLeakyReLU,
        'ELU': ActivationELU,
        'Tanh': ActivationTanh,
        'Softmax': ActivationSoftMax
    }
    @staticmethod
    def get_activation(activation_type: str) -> Activations:
        """Returns an instance of the specified activation function."""
        return NetworkCore.activation_map[activation_type]()  # creates ne instance with ()
    @staticmethod
    def get_activation_name(activation: Activations) -> str:
        """Returns string representation for a given activation function."""
        for name, cls in NetworkCore.activation_map.items():
            if isinstance(activation, cls):
                return name
        raise ValueError(f"Unknown activation type: {type(activation)}")
    
    @staticmethod
    def create_network(
        input_size: int, 
        hidden_sizes: List[int],
        output_size: int,
        hidden_activation_types: List[str] | str = 'ReLU',
        output_activation_type: str = 'Softmax',
        rng: np.random.Generator | None = None
    ) -> Network:
        """
        Creates a neural network from given parameters.

        Args:
            input_size (int): Number of input neurons
            hidden_sizes (List[int]): List of number of neurons in each hidden layer
            hidden_activation_types (List[str] | str): List of activation types for each hidden 
                layer or a single activation type for all hidden layers
            output_size (int): Number of output neurons
            output_activation_type (str, optional): Activation type for output layer. Defaults to 'Softmax'.
        Returns:
            Dict[str, Any]: Dictionary containing the network layers and architecture.
        """
        layers: List[Layer] = []
        activations: List[Activations] = []
        
        prev_size = input_size

        if isinstance(hidden_activation_types, str):
            hidden_activation_types = [hidden_activation_types] * len(hidden_sizes)
        
        if len(hidden_activation_types) != len(hidden_sizes):
            raise ValueError(
                f"Expected {len(hidden_sizes)} activation types, "
                f"got {len(hidden_activation_types)}"
            )
        
        for hidden_size, activation_type in zip(hidden_sizes, hidden_activation_types):
            layers.append(Layer(prev_size, hidden_size, rng))
            activations.append(NetworkCore.get_activation(activation_type))
            prev_size = hidden_size
        
        layers.append(Layer(prev_size, output_size, rng))
        activations.append(NetworkCore.get_activation(output_activation_type))
        
        return Network(
            [input_size] + hidden_sizes + [output_size],
            [NetworkLayer(layer, activation) 
             for layer, activation in zip(layers, activations)]
        )
        
    @staticmethod
    def create_network_from_file(input_size: int, filename: str) -> Network:
        """
        Creates a neural network from a saved file.

        Reads the network structure from the file and creates
        all layers with their saved weights, biases, and activation functions.

        Args:
            input_size (int): Expected input size for the network
            filename (str): Path to the CSV file containing the network data

        Returns:
            Network: A fully configured network loaded from the file

        Raises:
            ValueError: If the file format is invalid or the input size doesn't match
            FileNotFoundError: If the specified file doesn't exist
        """
        network_structure = DataTransfer.load_network_structure(filename)

        if not network_structure:
            raise ValueError(f"No layers found in file: {filename}")

        if network_structure[0][0] != input_size:
            raise ValueError(
                f"Input size mismatch: file expects {network_structure[0][0]} "
                f"but {input_size} was specified"
            )

        layers = [Layer(n_inputs, n_neurons) for n_inputs, n_neurons, _ in network_structure]
        
        # Activation functions now come directly from the file headers
        activation_names = DataTransfer.load_layers_data(filename, layers)

        return Network(
            [input_size] + [n_neurons for _, n_neurons, _ in network_structure],
            [NetworkLayer(layer, NetworkCore.get_activation(activation))
            for layer, activation in zip(layers, activation_names)]
        )

    @staticmethod
    def forward_pass(network: Network, X: NDArray[np.float32]) -> NDArray[np.float32]:    
        """
        Forward pass through the network.

        Args:
            network (Network): Network instance
            X (NDArray[np.float32]): Input array

        Returns:
            NDArray[np.float32]: Output array after forward pass
        """
        current_input = X
        for network_layer in network.layers:
            network_layer.layer.forward(current_input)
            network_layer.activation.forward(network_layer.layer.output)
            current_input = network_layer.activation.output
        
        return current_input
    
    @staticmethod
    def backward_pass(network: Network, loss_function: Loss, y_true: NDArray[np.uint8]):
        loss_function.calculate(network.layers[-1].activation.output, y_true)
        gradients = loss_function.backward(network.layers[-1].activation.output, y_true)
        
        for i in range(len(network.layers) - 1, -1, -1):
            network.layers[i].activation.backward(gradients)
            network.layers[i].layer.backward(network.layers[i].activation.dInputs)
            gradients = network.layers[i].layer.dInputs
    
    @staticmethod
    def train_network(
        network: Network, 
        X_train: NDArray[np.float32], 
        y_train: NDArray[np.uint8],
        epochs: int = 100, 
        learning_rate: float = 0.001, 
        batch_size: int = 64,
        optimizer_type: str = 'adam', 
        debugging: bool = True,
        rng: np.random.Generator | None = None
    ) -> Dict[str, Any]:
        """
        Train the network.

        Args:
            network (Network): Network instance
            X_train (NDArray[np.float32]): Training data
            y_train (NDArray[np.uint8]): Training labels
            epochs (int, optional): Number of training epochs.
            learning_rate (float, optional): Learning rate for the optimizer.
            batch_size (int, optional): Batch size for training.
            optimizer_type (str, optional): Type of optimizer to use.
            debugging (bool, optional): Enable debug mode.

        Returns:
            Dict[str, Any]: Dictionary containing training statistics:
              (training_time, final_loss, test_accuracy, loss_history, epochs)
        """
        # Setup
        if rng is None:
            rng = np.random.default_rng()
        
        n_samples: int = X_train.shape[0]

        loss_function = Loss_CategoricalCrossentropy()
        
        if optimizer_type.lower() == 'adam':
            optimizer = AdamOptimizer(learningRate=learning_rate)
        else:
            optimizer = simpleOptimizer(learningRate=learning_rate)
        
        # Tracking
        start_time = time.time()
        loss_history: List[np.float32] = []
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss: np.float32 = np.float32(0.0)
            
            # Create shuffled indices for this epoch
            # Ensures random mini-batches, to avoid order-dependent gradient bias
            indices: NDArray[np.int32] = np.arange(n_samples, dtype=np.int32)
            rng.shuffle(indices)
            
            # Mini-batch training
            for start_idx in range(0, n_samples, batch_size):
                end = min(start_idx + batch_size, n_samples)
                batch_idx: NDArray[np.int32] = indices[start_idx:end]

                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                # Forward pass
                output = NetworkCore.forward_pass(network, X_batch)

                # Compute loss for this batch
                loss = loss_function.calculate(output, y_batch)
                epoch_loss += loss * X_batch.shape[0] # Scale loss by batch size (last batch may be smaller)

                # Backward pass
                NetworkCore.backward_pass(network, loss_function, y_batch)

                # Update parameters
                for network_layer in network.layers:
                    optimizer.update(network_layer.layer)

            # Average loss across all samples
            avg_loss = epoch_loss / n_samples
            loss_history.append(avg_loss)

            # Debug output
            if debugging and epoch % (epochs // 10) == 0:
                elapsed = time.time() - start_time
                print(f"      Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} - Time: {elapsed:.2f}s")
        
        # calculate final stats
        training_time = time.time() - start_time

        # final training accuracy
        outputs: NDArray[np.float32] = NetworkCore.forward_pass(network, X_train)
        predictions: NDArray[np.uint8] = np.argmax(outputs, axis=1)
        train_accuracy: np.float32 = np.mean(predictions == y_train,dtype=np.float32)
        
        return {
            'training_time': training_time,
            'final_loss': loss_history[-1],
            'train_accuracy': train_accuracy,
            'loss_history': loss_history,
            'epochs': epochs
        }
    
    @staticmethod
    def evaluate_network(
        network: Network, 
        X_test: NDArray[np.float32], 
        y_test: NDArray[np.uint8],
        examplecount: int = 0
    ) -> Dict[str, Any]:
        """
        Evaluate the network on the given test data.

        Args:
            network (Network): Network instance
            X_test (NDArray[np.float32]): Input array for test data
            y_test (NDArray[np.uint8]): True labels for test data
            examplecount (int, optional): Number of examples to extract from the predictions.

        Returns:
            Dict[str, Any]: Dictionary containing the evaluation metrics (test_loss, accuracy, evaluation_time, confusion_matrix, examples)
        """

        start_time = time.time()
        
        outputs = NetworkCore.forward_pass(network, X_test)
        
        loss_function = Loss_CategoricalCrossentropy()
        test_loss = loss_function.calculate(outputs, y_test)
        
        # claculate accuracy
        predictions: NDArray[np.uint8] = np.argmax(outputs, axis=1)
        true_labels: NDArray[np.uint8] = y_test if len(y_test.shape) == 1 else np.argmax(y_test, axis=1)
        accuracy = np.mean(predictions == true_labels,dtype=np.float32)

        # extract examples
        examples: Optional[NDArray[np.uint8]] = None
        if examplecount > 0:
            examples = predictions[:examplecount]
            true_labels = true_labels[:examplecount]

        # calculate confusion matrix
        num_classes: int = outputs.shape[1]
        confusion_matrix: NDArray[np.uint64] = np.zeros(shape=(num_classes, num_classes), dtype=np.uint64)
        for true, pred in zip(true_labels, predictions):
            confusion_matrix[true, pred] += 1
        
        evaluation_time = time.time() - start_time
        
        return {
            'test_loss': test_loss,
            'accuracy': accuracy,
            'evaluation_time': evaluation_time,
            'confusion_matrix': confusion_matrix,
            **({'y_pred': examples, 'y_true': true_labels} if examplecount > 0 else {})
        }
    

    @staticmethod
    def collect_loss_landscape(
        network: Network, 
        X: NDArray[np.float32], 
        y: NDArray[np.uint8], 
        layer_idx: int = 0,
        resolution: int = 20
    ) -> NDArray[np.float32]:
        """
        Creates a 2D loss landscape visualization by perturbing weights 
        PS: NOT USED IN PROJECT
        
        Returns:
            2D array of loss values
        """
        loss_function = Loss_CategoricalCrossentropy()
        original_weights: NDArray[np.float32] = network.layers[layer_idx].layer.weights.copy()
        
        # Create perturbation grid
        alpha_range = np.linspace(-0.5, 0.5, resolution)
        beta_range = np.linspace(-0.5, 0.5, resolution)
        loss_landscape = np.zeros((resolution, resolution), dtype=np.float32)
        
        # Random direction vectors
        dir1 = np.random.randn(*original_weights.shape)
        dir2 = np.random.randn(*original_weights.shape)
        dir1 /= np.linalg.norm(dir1)
        dir2 /= np.linalg.norm(dir2)
        
        # Calculate loss for each point
        for i, alpha in enumerate(alpha_range):
            for j, beta in enumerate(beta_range):
                # Perturb weights
                perturbed: NDArray[np.float32] = original_weights + alpha * dir1 + beta * dir2
                network.layers[layer_idx].layer.weights = perturbed
                
                # Forward pass and calculate loss
                output = NetworkCore.forward_pass(network, X)
                loss = loss_function.calculate(output, y)
                loss_landscape[i, j] = loss
        
        # Restore original weights
        network.layers[layer_idx].layer.weights = original_weights
        return loss_landscape