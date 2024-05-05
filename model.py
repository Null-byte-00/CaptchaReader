import torch
from torch import nn


class SingleDigitModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(start_dim=0),
            nn.Linear(33856, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Sigmoid(),
        )
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.005)
    
    def forward(self, x):
        return self.stack(x)
    
    def train_datapoint(self, x, y, verbose=False):
        #self.optimizer.zero_grad()
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)
        loss.backward()
        #self.optimizer.step()
        if verbose:
            print(f"Loss: {loss.item()}")
        return loss.item()
    
    def end_batch(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def save(self, file_name='models/model.pth'):
        torch.save(self.state_dict(), file_name)
    
    def load(self, file_name='models/model.pth'):
        self.load_state_dict(torch.load(file_name))