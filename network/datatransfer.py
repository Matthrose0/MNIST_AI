import numpy as np
from numpy.typing import NDArray
import csv
from typing import TextIO, List, Tuple, Optional
from network.components import Layer
np.random.seed(0)

# Dataformat in MNIST: 0 which number it is 1..785 pixeldata (28*28=784)
# working in dataformat float32bit for accurracy

class DataTransfer:
    """Utility class for reading/writing neural network data from/to CSV files"""
    @staticmethod
    def readDataFromCSV(
        csvFilePath: str, 
        startIndex: int = 0, 
        endIndex: int = 10
    ) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
        """
        Reads data from CSV file and returns tuple of images and labels.

        Args:
            csvFilePath (str): Path to CSV file
            startIndex (int, optional): Start index for reading data. Defaults to 0.
            endIndex (int, optional): End index for reading data. Defaults to 10.

        Returns:
            tuple[NDArray[np.float32], NDArray[np.uint8]]: Tuple containing images and labels
        """
        labels: list[int] = []
        images: list[list[int]] = []

        csvfile = open(csvFilePath, mode='r', newline='')
        csvReader = csv.reader(csvfile)
        for i, row in enumerate(csvReader):
            if i < startIndex:
                continue
            if i >= endIndex:
                break
            
            # Take only the first value (which number the immage shows)
            labels.append(int(row[0])) 
            # get pixeldata for one immage(28*28 with shades of gray -> values 0-255)
            pixels = [int(value) for value in row[1:785]]
            images.append(pixels)
        
        csvfile.close()
        return (np.array(images, dtype=np.float32), np.array(labels, dtype=np.uint8))


    @staticmethod
    def save_network(
        filename: str, 
        layers: List[Layer], 
        activation_functions: List[str]
    ) -> None:
        """
        Saves the given layers to a CSV file.

        File format:
        - For each layer:
            - LAYER: inputs, neurons, activation_function
            - Bias values (comma-separated)
            - Weight matrix (one row per line)
        
        Args:
            filename (str): Path to CSV file
            layers (List[Layer]): Layers to save
        """
        with open(filename, "w") as f:
            for activation_function, layer in zip(activation_functions, layers):
                DataTransfer._save_layer_data(layer, activation_function, f)

    @staticmethod
    def _save_layer_data(layer: Layer, activation_function: str, f: TextIO):
        """Saves a single layer to a CSV file"""
        # Write layer header: inputs,neurons,activation_function
        n_inputs, n_neurons = layer.weights.shape
        f.write(f"LAYER:{n_inputs},{n_neurons},{activation_function}\n")

        # Save biases (flatten in case it's 2D array)
        biases_flat = layer.biases.flatten()
        bias_line = ','.join(str(bias) for bias in biases_flat)
        f.write(f"{bias_line}\n")
        
        # safe weights (one row per line)
        weight_row: NDArray[np.float32]
        for weight_row in layer.weights:
            weight_line = ','.join(str(weight) for weight in weight_row)
            f.write(f"{weight_line}\n")
    
    @staticmethod
    def load_network_structure(filename: str) -> List[Tuple[int, int, str]]:
        """
        Loads the network structure from a CSV file without creating layers.
        
        Returns a list of tuples containing:
        - (n_inputs, n_neurons, activation_function_name) for each layer
        
        Args:
            filename (str): Path to CSV file
            
        Returns:
            List (Tuple[int, int, str]): Network structure information
        """
        network_structure: List[Tuple[int, int, str]] = []

        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                header = DataTransfer._read_layer_header(line)
                if not header: continue

                network_structure.append(header)

        if not network_structure:
            raise ValueError("Empty network file or invalid format")
        
        return network_structure
    
    @staticmethod
    def _read_layer_header(line: str) -> Optional[Tuple[int, int, str]]:
        if not line.startswith("LAYER:"):
            return None
        n_inputs, n_neurons, activation_function = line[6:].split(',')

        if not n_inputs.isdigit() or not n_neurons.isdigit():
            raise ValueError(f"Invalid layer header: {line}")
        
        return int(n_inputs), int(n_neurons), activation_function

    @staticmethod
    def load_layers_data(filename: str, layers: List[Layer]) -> List[str]:
        """
        Loads layers from CSV file, returns list of activation function names per layer.

        File format:
        - For each layer:
            - LAYER: inputs, neurons, activation_function
            - Bias values (comma-separated)
            - Weight matrix (one row per line)
    
        Args:
            filename (str): Path to CSV file
            layers (List[Layer]): Layers to load
        """
        """"""
        with open(filename, "r") as f:
            lines = [line.strip() for line in f.readlines()]

        activation_functions: List[str] = []
        line_index = 0
        for layer in layers:
            line_index, activation = DataTransfer._load_layer_data(lines, line_index, layer)
            activation_functions.append(activation)
        return activation_functions
    
    @staticmethod
    def _load_layer_data(lines: list[str], start_index: int, layer: Layer) -> Tuple[int, str]:
        """Loads a single layer, returns (next_line_index, activation_function_name)."""
        # Check if we have enough lines left (min 1 neuron / 1 input -> 1 bias / 1 weight)
        if start_index + 2 >= len(lines):
            raise ValueError(f"Unexpected end of file at line {start_index}")

        # Read layer header to get dimensions
        header_line = lines[start_index]
        if not header_line.startswith("LAYER:"):
            raise ValueError(f"Expected layer header, got: {header_line}")

        try:
            parts = header_line[6:].split(',')  # Skip "LAYER:" prefix
            n_inputs, n_neurons, activation_function = int(parts[0]), int(parts[1]), parts[2]
        except (ValueError, IndexError):
            raise ValueError(
                f"Invalid layer header at line {start_index}: "
                f"'{header_line}', expected 'LAYER:inputs,neurons,activation_function"
            )

        if n_inputs <= 0 or n_neurons <= 0:
            raise ValueError(
                f"Invalid dimensions at line {start_index}: inputs={n_inputs}, neurons={n_neurons}"
            )

        # Check if we have enough lines for biases + weights
        current_index = start_index + 1 # line after header
        required_lines = 1 + n_inputs  # 1 bias line + n_inputs weight rows
        if current_index + required_lines > len(lines):
            raise ValueError(
                f"Not enough lines for layer: need {required_lines} "
                f"lines but only {len(lines) - current_index} remaining"
            )
        
        # Load biases
        layer.biases = DataTransfer._parse_line_to_array(lines[current_index])
        if len(layer.biases) != n_neurons:
            raise ValueError(
                f"Bias count mismatch at line {current_index}: "
                f"got {len(layer.biases)} biases, expected {n_neurons}"
            )
        current_index += 1

        # Load weights from multiple lines
        weight_rows = [
            DataTransfer._validate_weight_row(
                lines[current_index + i], i, current_index + i, n_neurons
            )
            for i in range(n_inputs)
        ]
        layer.weights = np.array(weight_rows)

        return current_index + n_inputs, activation_function
    
    @staticmethod
    def _validate_weight_row(
        line: str, 
        row_index: int,
        line_number: int,
        expected_count: int
    ) -> NDArray[np.float32]:
        """Parse and validate a single weight row"""
        weight_row = DataTransfer._parse_line_to_array(line)
        if len(weight_row) != expected_count:
            raise ValueError(
                f"Weight row {row_index} at line {line_number}: "
                f"got {len(weight_row)} weights, expected {expected_count}"
            )
        return weight_row
    
    @staticmethod
    def _parse_line_to_array(line: str) -> NDArray[np.float32]:
        """Convert CSV line to numpy array, filtering empty strings"""
        return np.array([float(x) for x in line.split(',') if x], dtype=np.float32)






import matplotlib.pyplot as plt

# X: shape (n_samples, 784)
def show_images(X: NDArray[np.float32], n: int=5):
    for i in range(n):
        plt.imshow(X[i].reshape(28, 28), cmap="gray") # type: ignore
        plt.show() # type: ignore