from clients.client_template import FlowerClient
from clients import load_and_split_data
import flwr as fl
import numpy as np

if __name__ == "__main__":
    datasets = load_and_split_data()
    x_train, y_train = datasets[3]

    # Inverser 30% des labels pour simuler une attaque
    poison_idx = np.random.choice(len(y_train), size=int(0.3 * len(y_train)), replace=False)
    y_train[poison_idx] = 1 - y_train[poison_idx]

    client = FlowerClient(cid="3", x_train=x_train, y_train=y_train)
    fl.client.start_numpy_client("localhost:8080", client=client)
