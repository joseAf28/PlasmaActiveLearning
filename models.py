import torch
import torch.nn as nn
import torch.optim as optim


#### Simple Feedforward Neural Network
class SimpleNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                index: int = 0, dropout_rate: float = 0.1, do_dropout: bool = False):
        
        super(SimpleNet, self).__init__()
        self.index = index
        self.do_dropout = do_dropout
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate) if do_dropout else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x