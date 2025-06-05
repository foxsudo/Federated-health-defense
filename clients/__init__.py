import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_split_data():
    df = pd.read_csv("data/synthetic_health.csv")
    features = df.drop("diagnostic", axis=1).values
    labels = df["diagnostic"].values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Découper en 5 clients
    split_data = np.array_split(features, 5)
    split_labels = np.array_split(labels, 5)
    return [(split_data[i], split_labels[i]) for i in range(5)]
