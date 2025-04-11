import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
from model import MultinomialLogisticRegression
from config import *

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using PyTorch version:', torch.__version__, ' device:', device)

class MNISTDataset(Dataset):
    def __init__(self, path_csv: str):
        """
        Initialize the dataset using a path to a CSV file
        
        Args:
            path_csv: path to a CSV file containing train or test data
        """
        super().__init__()
        assert os.path.exists(path_csv) and os.path.isfile(path_csv), f'Path {path_csv} does not point to a file'
        self.path_csv = path_csv
        
        _all_digits = pd.read_csv(path_csv, header=None).values
        _y = _all_digits[:, 0]
        _x = _all_digits[:, 1:]
        _x = _x / 255  # Normalize pixel values to [0,1]
        
        self.x = torch.Tensor(_x)
        self.y = torch.Tensor(_y)
        
    def __len__(self):
        """
        Return the length of the current dataset
        
        Returns:
            Length of the dataset
        """
        return len(self.y)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the pair (input, associated_label) from position index
        
        Args:
            index: position from which to return the vector and its class
            
        Returns:
            pair: input vector of 784 elements, class label 0-9
        """
        return self.x[index, :], self.y[index]

def train_model():
    model = MultinomialLogisticRegression(input_size=INPUT_SIZE, k=NUM_CLASSES)
    model.to(device)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    train_ds = MNISTDataset(TRAIN_DATA_PATH)
    test_ds = MNISTDataset(TEST_DATA_PATH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).long()
            
            optimizer.zero_grad()
            
            y_hat = model(x)
            
            loss = loss_fn(y_hat, y)
            running_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}')
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device).long()
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    print(f'Accuracy on test set: {100 * correct / total:.2f}%')
    
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f'Model saved to {MODEL_SAVE_PATH}')

if __name__ == "__main__":
    train_model()