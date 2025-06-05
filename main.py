from clients.client_template import FlowerClient
from clients import load_and_split_data
import flwr as fl

if __name__ == "__main__":
    datasets = load_and_split_data()

    # Démarrer un client (change l’indice pour en lancer plusieurs)
    def client_fn(cid):
        x_train, y_train = datasets[int(cid)]
        return FlowerClient(cid, x_train, y_train)

    fl.client.start_client(server_address="localhost:8080", client=client_fn("0").to_client())


