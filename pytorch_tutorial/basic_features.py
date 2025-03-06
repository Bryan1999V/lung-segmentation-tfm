"""Implement the Pytorch basic functionalities for neural networks."""

import logging

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils import data

log = logging.getLogger(__name__)


class SimpleClassifier(nn.Module):
    """Model with an input layer, a hidden layer with tanh as activation function and an output layer."""

    def __init__(
        self,
        n_neurons_input_layer: int,
        n_neurons_hidden_layer: int,
        n_neurons_output_layer: int,
    ) -> None:
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


class XORDataset(data.Dataset):
    """
    XOR dataset generation with data and ground-truth values.

    A gaussian noise is applied to the data vector points.
    """

    _MIN_VALUE = 0
    _MAX_VALUE_EXCLUDED = 2
    _DATA_NUMBER_COLUMNS = 2

    def __init__(self, len_data: int, standard_deviation: float) -> None:
        """
        Initialize the dataset with given parameters.

        :param len_data: number of data points to generate.
        :param standard_deviation: noise value to apply to dataset.
        """
        super().__init__()
        self._data, self._gt = self._generate_continuous_xor_data(len_data, standard_deviation)

    def __len__(self) -> int:
        """
        Return the number of the data points.

        :return: length of the generated data.
        """
        return self._data.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, float]:
        """
        Get a specific data point and ground-truth values by the given index number.

        :param index: reference number of the data to return from the generated arrays.
        :return: a tuple with the data point and ground-truth values.
        """
        return self._data[index], self._gt[index]

    def _generate_continuous_xor_data(
        self,
        len_data: int,
        standard_deviation: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a continuous XOR dataset with data points and ground-truth labels.

        The data points have a bit of gaussian noise.

        :param len_data: number of data points to generate.
        :param standard_deviation: noise value to apply to dataset.
        :return: data points and ground-truth values, respectively, as a Tensor object.
        """
        original_data = torch.randint(
            low=self._MIN_VALUE,
            high=self._MAX_VALUE_EXCLUDED,
            size=(len_data, self._DATA_NUMBER_COLUMNS),
            dtype=torch.float32,
        )
        gt = (original_data.sum(dim=1) == 1).to(torch.long)
        data_with_noise = original_data + (standard_deviation * torch.randn(original_data.shape))

        return data_with_noise, gt

    def plot_graph(self) -> None:
        """Plot a graph with every sample of the dataset."""
        data = self._data.cpu().numpy()
        label = self._gt.cpu().numpy()
        data_0 = data[label == 0]
        data_1 = data[label == 1]

        plt.figure(figsize=(4, 4))
        plt.scatter(data_0[:, 0], data_0[:, 1], edgecolors="#333", label="Class 0")
        plt.scatter(data_1[:, 0], data_1[:, 1], edgecolors="#333", label="Class 1")
        plt.title("XOR Dataset samples")
        plt.ylabel(r"$x_2$")
        plt.ylabel(r"$x_1$")
        plt.legend()
        plt.show()


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


def show_dataset_info() -> None:
    """Show the relevant Pytorch information of the XOR dataset class."""
    log.debug("Generate XOR dataset with 500 samples.")
    xor_dataset = XORDataset(len_data=500, standard_deviation=0.1)

    log.debug(f"Length of the XOR dataset: {len(xor_dataset)}")

    min_random_index = 0
    max_random_index = len(xor_dataset)
    n_samples = 10
    random_index = torch.randint(low=min_random_index, high=max_random_index, size=(n_samples,))
    log.debug(random_index)
    for i in random_index:
        x, y = xor_dataset[i]
        log.debug(f"Getting the <{i}> index data from dataset:\n{x}\n{y}\n")

    xor_dataset.plot_graph()


def create_dataloader() -> None:
    """Create a DataLoader object from the XOR dataset."""
    batch_size = int(input("Set the batch size: "))

    xor_dataset = XORDataset(500, 0.1)
    data_loader = data.DataLoader(dataset=xor_dataset, batch_size=batch_size, shuffle=True)

    inputs, labels = next(iter(data_loader))
    log.debug(f"Data input for first batch ({inputs.size()}):\n{inputs}")
    log.debug(f"Data labels for first batch ({labels.size()}):\n{labels}")


def train_model(model: SimpleClassifier, n_epochs: int) -> None:
    """
    Train the SimpleClassifier model with the XOR dataset.

    For this training session, the optimizer is a Stochastic Gradient Descent with a learning rate of 0.1 and the loss
    function is Binary Cross Entropy with Logits. The workflow is:

    1. Create a DataLoader object with a batch size of 125.
    3. Define the optimizer with the model parameters and the learning rate.
    4. Iterate over the number of epochs and the DataLoader object.
    5. Calculate the predictions with the model and the input data.
    6. Calculate the loss value with the predictions and the ground-truth values.
    7. Reset the optimizer gradients to zero.
    8. Calculate the gradients with the loss value.
    9. Update the model parameters with the optimizer.

    :param model: SimpleClassifier model to train.
    :param n_epochs: number of epochs to train the model.
    """
    log.debug("Train the SimpleClassifier model with the XOR dataset.")
    train_dataset = XORDataset(len_data=2500, standard_deviation=0.1)
    train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=125, shuffle=True)

    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_module = nn.BCEWithLogitsLoss()

    for epoch in torch.arange(n_epochs):
        log.debug(f"Epoch {epoch + 1}/{n_epochs}")

        for inputs, gt in train_dataloader:
            preds = model(inputs)
            preds = preds.squeeze(dim=1)

            loss = loss_module(preds, gt.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        log.debug(f"\tLoss value: {loss.item()}")

    log.debug(f"Model info:\n{model.state_dict()}")


def eval_model(model: SimpleClassifier) -> None:
    """
    Evaluate the SimpleClassifier model with the XOR dataset.

    For this evaluation session, a Sigmoid function is applied to the predictions given by the model using a new
    dataset. Then, the true predictions and the total number of predictions are calculated to estimate the accuracy of
    the model.

    :param model: SimpleClassifier model to evaluate.
    """
    sigmoid_threshold = 0.5
    true_preds = 0
    num_preds = 0
    eval_dataset = XORDataset(len_data=500, standard_deviation=0.1)
    eval_dataloader = data.DataLoader(dataset=eval_dataset, batch_size=125, shuffle=False, drop_last=False)

    model.eval()

    with torch.no_grad():
        for inputs, labels in eval_dataloader:
            preds = model(inputs)
            preds = preds.squeeze(dim=1)
            preds = torch.sigmoid(preds)
            pred_labels = (preds >= sigmoid_threshold).long()

            true_preds += (pred_labels == labels).sum()
            num_preds += labels.shape[0]

        accuracy = true_preds / num_preds
        log.debug(f"Accuracy of the model: {100.0 * accuracy:4.2f}%")


def basic_features() -> None:
    """Show the general information about Pytorch."""
    log.debug(f"Pytorch version: {torch.__version__}")
    log.debug(f"Cuda avaliable: {torch.cuda.is_available()}")

    model = SimpleClassifier(2, 4, 1)
    train_model(model=model, n_epochs=100)
    eval_model(model=model)
