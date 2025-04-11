import torch
import torch.nn as nn

class MultinomialLogisticRegression(nn.Module):
    def __init__(self, input_size=784, k=10):
        """
        Initialize a MultinomialLogisticRegression object
        
        Args:
            input_size: size of the input vector. For MNIST it's 784
            k: number of classes. For MNIST it's 10
        """
        super().__init__()
        self.input_size = input_size
        self.fully_connected = nn.Linear(input_size, k, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward propagation of input data x
        
        Args:
            x: input values
            
        Returns:
            torch.Tensor: predictions for input x
        """
        return self.fully_connected(x)