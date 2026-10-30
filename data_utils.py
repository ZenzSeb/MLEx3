"""Data loading and splitting for Exercise Sheet 03. You do not need to edit this file.

ames_housing.csv contains 2413 house sales in Ames, Iowa (USA), 2006--2010 (De Cock, 2011).
Areas are in m^2, the price in thousands of US dollars (k$).
"""
import os

import pandas as pd

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ames_housing.csv")


def load_housing():
    """Returns the features as a pandas DataFrame of shape (N, D) (one column per feature) and the prices y as a
    np.ndarray of shape (N,)."""
    data = pd.read_csv(DATA_FILE)

    return data.drop(columns="price"), data["price"].to_numpy()


def train_test_split(x, y, n_train, rng):
    """Randomly splits (x, y) into n_train training examples and the remaining test examples."""
    permutation = rng.permutation(len(y))
    train, test = permutation[:n_train], permutation[n_train:]

    return x[train], x[test], y[train], y[test]
