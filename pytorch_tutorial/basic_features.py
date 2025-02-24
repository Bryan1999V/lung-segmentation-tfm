"""Implement the Pytorch basic functionalities for neural networks."""

import torch
from torch import nn
import logging

log = logging.getLogger(__name__)


class SimpleClassifier(nn.Module):
    """Model with an input layer, a hidden layer with tanh as activation function and an output layer."""

    def __init__(self, n_neurons_input_layer: int, n_neurons_hidden_layer: int, n_neurons_output_layer: int) -> None:
        """
        Initialize the modules with the given parameter values.

        :param n_neurons_input_layer: nodes number of the input layer.
        :param n_neurons_hidden_layer: nodes number of the hidden layer.
        :param n_neurons_output_layer: nodes number of the output layer.
        """
        super().__init__()

        self._linear1 = nn.Linear(in_features=n_neurons_input_layer, out_features=n_neurons_hidden_layer)
        self._act_func1 = nn.Tanh()
        self._linear2 = nn.Linear(in_features=n_neurons_hidden_layer, out_features=n_neurons_output_layer)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Perform the main calculation of the model to return the prediction value.

        :param input_data: values of the input vector with a shape that matches with the number of the input neurons.
        :return: value of the output vector with a shape that matches with the number of the output neurons.
        """
        x = self._linear1(input_data)
        x = self._act_func1(x)
        return self._linear2(x)


def basic_tensor_operations() -> None:
    """Debug the basic and main operations with tensor object."""
    log.debug("Tensor object with 3 matrixes and each matrix has 4 rows and 2 columns (3, 4, 2):")
    log.debug(f"\n{torch.Tensor(3, 4, 2)}")

    log.debug("Tensor object from nested list: [[1, 2], [3, 4]] - (4, 2) shape:")
    log.debug(f"\n{torch.Tensor([[1, 2], [3, 4]])}")

    log.debug("Tensor object with zeros and (4, 3, 4) dimensions:")
    log.debug(f"\n{torch.zeros(size=(4, 3, 4), dtype=torch.int8)}")

    log.debug("Tensor object with ones and (6, 5, 3) dimensions:")
    log.debug(f"\n{torch.ones(size=(6, 5, 3), dtype=torch.float16)}")

    log.debug("Tensor object with random values between 0-1 and (2, 6, 5) dimensions:")
    log.debug(f"\n{torch.rand(size=(2, 6, 5), dtype=torch.float)}")

    log.debug(
        "Tensor object with random values sampled from a normal distribution with mean 0, variance 1 and (4, 3, 4) "
        "dimensions:"
    )
    log.debug(f"\n{torch.randn(size=(4, 3, 4))}")

    log.debug("Tensor object with the numbers between 0-20:")
    log.debug(f"\n{torch.arange(start=0, end=21, step=1)}")

    log.debug("Convert the last Tensor object to numpy array:")
    log.debug(f"\n{torch.arange(start=0, end=21, step=1).numpy()}")

    x1 = torch.rand(size=(3, 3))
    x2 = torch.rand(size=(3, 3))
    log.debug(f"First tensor object:\n{x1}")
    log.debug(f"Second tensor object:\n{x2}")

    log.debug(f"Sum of 2 tensor objects:\n{x1 + x2}")
    x2.add_(x1)
    log.debug(f"Sum of 2 tensor objects and save in second object:\n{x2}")

    log.debug(f"Change the before shape (3, 3) to (9, 1):\n{x1.view(size=(9, 1))}")
    log.debug(f"Permute the new shape\n{x1.view(size=(9, 1)).permute(0, 1)}")


def gradient_of_input() -> None:
    """Debug a gradient operation value depending on the input value with tensor object."""
    log.debug("Function to optimize: y = (1 / len(x)) * sum((x + 2)^2 + 3)")
    x = torch.arange(end=3, dtype=torch.float32, requires_grad=True)
    log.debug(f"Input data:\n{x}")

    y = (((x + 2) ** 2) + 3).mean()
    log.debug(f"Output value: {y}")

    y.backward()
    log.debug(f"Gradient value:\n{x.grad}")


def show_simple_classifier_module_info() -> None:
    """Show the information related to the SimpleClassifier module."""
    log.debug("SimpleClassifier model with 2 input neurons, 4 hidden neurons and 1 output neurons.")

    model = SimpleClassifier(2, 4, 1)
    log.debug(f"\n{model}")

    log.debug("Display every parameter and its shape value:")
    for name, param in model.named_parameters():
        log.debug(f"Parameter <{name}> - Shape <{param.shape}>")


def basic_features() -> None:
    """Show the general information about Pytorch."""
    log.debug(f"Pytorch version: {torch.__version__}")

    show_simple_classifier_module_info()
