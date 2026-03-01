import numpy as np
from numpy.typing import NDArray
from typing import Dict
from abc import ABC,abstractmethod

class Layer:
    def __init__(self, n_inputs: int, n_neurons: int, rng: np.random.Generator | None = None):
        if rng is None: rng = np.random.default_rng()
        self.weights: NDArray[np.float32] = 0.10 * rng.standard_normal((n_inputs, n_neurons), dtype=np.float32)
        self.biases: NDArray[np.float32] = np.zeros((1, n_neurons), dtype=np.float32)
    
    def forward(self, inputs: NDArray[np.float32]):
        self.inputs = inputs.copy()
        self.output: NDArray[np.float32] = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dValues: NDArray[np.float32]):
        # dValues = gradients of loss from layer after
        self.dWeights: NDArray[np.float32] = np.dot(self.inputs.T, dValues)
        self.dBiases: NDArray[np.float32] = np.sum(dValues, axis=0, keepdims=True)
        self.dInputs: NDArray[np.float32] = np.dot(dValues, self.weights.T)


class Activations(ABC):
    @abstractmethod
    def forward(self, inputs: NDArray[np.float32]):
        self.inputs = inputs
        self.output: NDArray[np.float32]
    @abstractmethod
    def backward(self, dValues: NDArray[np.float32]):
        self.dInputs: NDArray[np.float32] = dValues.copy()

class ActivationLinear(Activations):
    def forward(self, inputs: NDArray[np.float32]):
        self.inputs = inputs
        self.output = inputs
    def backward(self, dValues: NDArray[np.float32]):
        self.dInputs = dValues.copy()

class ActivationReLU(Activations):
    def forward(self, inputs: NDArray[np.float32]):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dValues: NDArray[np.float32]):
        self.dInputs = dValues.copy()
        self.dInputs[self.inputs <= 0] = 0

class ActivationLeakyReLU(Activations):
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
    def forward(self, inputs: NDArray[np.float32]):
        self.inputs = inputs
        self.output = np.where(inputs > 0, inputs, self.alpha * inputs)
    def backward(self, dValues: NDArray[np.float32]):
        self.dInputs = dValues.copy()
        self.dInputs[self.inputs <= 0] *= self.alpha

class ActivationELU(Activations):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def forward(self, inputs: NDArray[np.float32]):
        self.inputs = inputs
        inputs_clipped = np.clip(inputs, -88, 88) # to avoid overflow
        self.output = np.where(inputs_clipped > 0, inputs_clipped, self.alpha * (np.exp(inputs_clipped) - 1))

    def backward(self, dValues: NDArray[np.float32]):
        self.dInputs = dValues.copy()
        self.dInputs[self.inputs <= 0] *= self.alpha * np.exp(self.inputs[self.inputs <= 0])
        
class ActivationSoftMax(Activations):
    def forward(self, inputs: NDArray[np.float32]):
        # transforming to probabilities
        self.inputs = inputs
        expValues: NDArray[np.float32] = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        self.output = expValues / np.sum(expValues, axis=1, keepdims = True)
    
    def backward(self, dValues: NDArray[np.float32]):
        self.dInputs = np.empty_like(dValues)
        for index, (single_output, single_dValues) in enumerate(zip(self.output, dValues)):
            single_output: NDArray[np.float32] = single_output.reshape(-1, 1)
            jacobian_matrix: NDArray[np.float32] = (
                np.diagflat(single_output) 
                - np.dot(single_output, single_output.T)
            )
            self.dInputs[index] = np.dot(jacobian_matrix, single_dValues)

class ActivationTanh(Activations):
    def forward(self, inputs: NDArray[np.float32]):
        self.inputs = inputs
        inputs_clipped = np.clip(inputs, -88, 88) # to avoid overflow
        ePow: NDArray[np.float32] = np.exp(inputs_clipped)
        eNeg: NDArray[np.float32] = np.exp(-inputs_clipped)
        self.output = (ePow - eNeg) / (ePow + eNeg)
    
    def backward(self, dValues: NDArray[np.float32]):
        self.dInputs = (1 - self.output**2) * dValues


class Loss(ABC):
    def calculate(self, output: NDArray[np.float32], y: NDArray[np.uint8]) -> np.float32:
        """
        Calculates the loss of the model on the given data.

        Args:
            output (NDArray[np.float32]): The output of the model.
            y (NDArray[np.uint8]): The true labels.

        Returns:
            float: The mean loss of the model on the given data.
        """
        sampleLosses: NDArray[np.float32] = self.forward(output, y)
        self.dataLoss: np.float32 = np.mean(sampleLosses, dtype=np.float32)
        return self.dataLoss
    
    @abstractmethod
    def forward(
        self, yPred: NDArray[np.float32], yTrue: NDArray[np.uint8]
    ) -> NDArray[np.float32]:
        pass
    @abstractmethod
    def backward(
        self, dValues: NDArray[np.float32], yTrue: NDArray[np.uint8]
    ) -> NDArray[np.float32]:
        pass

class Loss_CategoricalCrossentropy(Loss):
    def forward(
        self, 
        yPred: NDArray[np.float32], 
        yTrue: NDArray[np.uint8]
    ) -> NDArray[np.float32]:
        """
        Forward pass for Categorical Crossentropy Loss.

        Args:
            yPred (NDArray[np.float32]): Predicted probabilities of each class
            yTrue (NDArray[np.uint8]): True labels, either in class indices or one-hot encoded
        
        Returns:
            NDArray[np.float32]: Negative log likelihoods of the loss
        """
        self.yPred = yPred
        self.yTrue = yTrue

        samples = len(yPred)
        yPredClipped = np.clip(yPred, 1e-10, 1 - 1e-10)

        if len(yTrue.shape) == 1:
            # yTrue are classindices [0,1,2,...]
            correctConfidences = yPredClipped[range(samples), yTrue]
        elif len(yTrue.shape) == 2:
            # yTrue are one-hot encoded labels
            correctConfidences = np.sum(yPred * yTrue, axis = 1)
        else:
            raise ValueError(f"Unexpected yTrue Shape: {yTrue.shape}")
        
        negativeLogLikelihoods = -np.log(correctConfidences)
        return negativeLogLikelihoods
    
    def backward(self, dValues: NDArray[np.float32], yTrue: NDArray[np.uint8]):
        """
        Backward pass for Categorical Crossentropy Loss.

        Args:
            dValues (NDArray[np.float32]): Error gradients of the loss with respect to the output
            yTrue (NDArray[np.uint8]): True labels, either in class indices or one-hot encoded

        Returns:
            self.dInputs (NDArray[np.float32]): Error gradients of the loss with respect to the input
        """
        samples = len(dValues)
        labels = len(dValues[0])

        # convert yTrue to One-Hot encoding if necessary
        yTrue_onehot: NDArray[np.float32]
        if len(yTrue.shape) == 1:
            yTrue_onehot = np.eye(labels)[yTrue].astype(np.float32)
        else:
            yTrue_onehot = yTrue.astype(np.float32)

        # clipping values to avoid dividing by 0
        dValuesClipped: NDArray[np.float32] = np.clip(dValues, 2e-25, 1 - 1e-10)
        # dürfte theoretisch nicht funktionieren mit 1e-25 hat aber 
        # trotzdem was geändert, wenn es kleiner war

        self.dInputs = -yTrue_onehot / dValuesClipped
        self.dInputs = self.dInputs / samples
        return self.dInputs

class simpleOptimizer:
    def __init__(self, learningRate: float=0.01):
        self.learningRate = learningRate
    
    def update(self, layer: Layer):
        layer.weights -= self.learningRate * layer.dWeights
        layer.biases -= self.learningRate * layer.dBiases

class AdamOptimizer:
    def __init__(
        self, 
        learningRate: float=0.001, 
        beta1: float=0.9, 
        beta2: float=0.999, 
        epsilon: float=1e-8
    ):
        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.t = 0 # timeStep initialize
        self.m_w: Dict[Layer, NDArray[np.float32]] = {} # m = 1. moment/-um (average gradient)
        self.m_b: Dict[Layer, NDArray[np.float32]] = {}
        self.v_w: Dict[Layer, NDArray[np.float32]] = {} # v = 2. moment/-um (average square of gradients)
        self.v_b: Dict[Layer, NDArray[np.float32]] = {}
    
    def _initialize_moments(self, layer: Layer) -> None:
        """Initialize momentum terms for a new layer"""
        if layer not in self.m_w:
            self.m_w[layer] = np.zeros_like(layer.weights)
            self.v_w[layer] = np.zeros_like(layer.weights)
            self.m_b[layer] = np.zeros_like(layer.biases)
            self.v_b[layer] = np.zeros_like(layer.biases)
    
    def _compute_adam_update(
            self, 
            m: NDArray[np.float32], 
            v: NDArray[np.float32], 
            dParam: NDArray[np.float32]
        ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """
        Compute Adam update for given momentum terms and gradient (= dParam)
        
        Returns:
            Tuple of (updated_m, updated_v, parameter_update)
        """
        # Update biased first and second moment estimates
        m = self.beta1 * m + (1 - self.beta1) * dParam
        v = self.beta2 * v + (1 - self.beta2) * (dParam ** 2)

        # Compute bias-corrected moment estimates
        m_hat: NDArray[np.float32] = m / (1 - self.beta1 ** self.t) # "hat" are the changed versions of the moments
        v_hat: NDArray[np.float32] = v / (1 - self.beta2 ** self.t)

        # Compute parameter update
        param_update: NDArray[np.float32] = (
            self.learningRate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        )
        return m, v, param_update
    
    def update(self, layer: Layer):
        """
        Update layer parameters using Adam optimization algorithm
        
        Args:
            layer: The layer to update
        """
        self.t += 1

        self._initialize_moments(layer)

        # Update weights
        self.m_w[layer], self.v_w[layer], m_w_hat = self._compute_adam_update(
            self.m_w[layer], self.v_w[layer], layer.dWeights
        )
        layer.weights -= m_w_hat

        # Update biases
        self.m_b[layer], self.v_b[layer], m_b_hat = self._compute_adam_update(
            self.m_b[layer], self.v_b[layer], layer.dBiases
        )
        layer.biases -= m_b_hat