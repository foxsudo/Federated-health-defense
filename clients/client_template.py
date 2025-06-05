import flwr as fl
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from utils.model import HealthNet
import pandas as pd
import numpy as np

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, x_train, y_train):
        self.cid = cid
        self.model = HealthNet()
        self.x_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.dataset = TensorDataset(self.x_train, self.y_train)
        self.loader = DataLoader(self.dataset, batch_size=32, shuffle=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.model.train()
        for epoch in range(1):
            for x_batch, y_batch in self.loader:
                optimizer.zero_grad()
                output = self.model(x_batch)
                loss = F.cross_entropy(output, y_batch)
                loss.backward()
                optimizer.step()
        return self.get_parameters(config), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        output = self.model(self.x_train)
        loss = F.cross_entropy(output, self.y_train)
        acc = (output.argmax(1) == self.y_train).float().mean().item()
        return float(loss), len(self.x_train), {"accuracy": acc}

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
