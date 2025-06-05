import multiprocessing
from clients.client_template import FlowerClient
from clients import load_and_split_data
import flwr as fl
import numpy as np

datasets = load_and_split_data()  # Charger les données une fois ici (global)

def run_client(cid: int):
    x_train, y_train = datasets[cid]

    # Simuler une attaque sur client 3 (exemple)
    if cid == 3:
        poison_idx = np.random.choice(len(y_train), size=int(0.3 * len(y_train)), replace=False)
        y_train[poison_idx] = 1 - y_train[poison_idx]

    client = FlowerClient(cid=str(cid), x_train=x_train, y_train=y_train)
    fl.client.start_client(
        server_address="localhost:8080",
        client=client.to_client()
    )

if __name__ == "__main__":
    processes = []
    for cid in range(5):  # 5 clients
        p = multiprocessing.Process(target=run_client, args=(cid,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
