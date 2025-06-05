import pandas as pd
import numpy as np
import os

def generate_data(path="data/synthetic_health.csv", n_samples=1000):
    np.random.seed(42)
    df = pd.DataFrame({
        "age": np.random.randint(20, 80, n_samples),
        "tension": np.random.normal(120, 15, n_samples),
        "battement_cardiaque": np.random.normal(75, 10, n_samples),
        "glycemie": np.random.normal(90, 15, n_samples),
        "diagnostic": np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    })
    os.makedirs("data", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Données sauvegardées dans : {path}")

if __name__ == "__main__":
    generate_data()
