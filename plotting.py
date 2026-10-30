"""Plotting functions for Exercise Sheet 03. All functions are provided, you do not need to edit this file.

All plots are saved to the folder plots/ next to this file.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#8a5cc2"]


def _save(fig, filename):
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, filename)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {path}")


def plot_explained_variance(ratio, errors, filename="1_explained_variance.pdf"):
    """Left: cumulative explained variance ratio. Right: mean squared reconstruction error. Both over K (Task 1.2)."""
    Ks = np.arange(1, len(ratio) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    ax1.plot(Ks, np.cumsum(ratio), "o-", color=COLORS[0], linewidth=2)
    ax1.axhline(0.9, color="gray", linestyle="--", linewidth=1, label="90 %")
    ax1.set_ylabel("cumulative explained variance ratio")
    ax1.set_ylim(0, 1.02)
    ax1.legend()

    ax2.plot(Ks, errors, "o-", color=COLORS[1], linewidth=2)
    ax2.set_ylabel("mean squared reconstruction error")
    ax2.set_ylim(bottom=0)

    for ax in (ax1, ax2):
        ax.set_xlabel("number of principal directions $K$")
        ax.set_xticks(Ks)
        ax.grid(True, alpha=0.3)

    _save(fig, filename)


def plot_pc_scatter(Z, y, ratio, filename="1_pc_scatter.pdf"):
    """Houses in the plane of the first two principal component scores, colored by the price (Task 1.2).

    Z: scores of shape (N, 2), y: prices of shape (N,), ratio: explained variance ratio of shape (D,)
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))

    points = ax.scatter(Z[:, 0], Z[:, 1], c=y, s=6, cmap="viridis", alpha=0.7)
    fig.colorbar(points, ax=ax, label="price (k\\$)")

    ax.set_xlabel(f"score $z_1$ ({100 * ratio[0]:.0f} % of the variance)")
    ax.set_ylabel(f"score $z_2$ ({100 * ratio[1]:.0f} % of the variance)")
    ax.grid(True, alpha=0.3)

    _save(fig, filename)


def plot_polynomial_fits(z_train, y_train, z_test, y_test, grid, curves, filename="2_polynomial_fits.pdf"):
    """Training and test houses and the fitted polynomials over the score z_1 (Task 2.2).

    curves: dict {degree: predictions of shape (len(grid),) on the grid}
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.scatter(z_test, y_test, s=4, color="lightgray", label="test houses")
    ax.scatter(z_train, y_train, s=25, color="black", zorder=3, label="training houses")
    for color, (degree, curve) in zip(COLORS, curves.items()):
        ax.plot(grid, curve, color=color, linewidth=2, label=f"degree $d = {degree}$")

    ax.set_ylim(0, 1.1 * y_test.max())
    ax.set_xlabel("score $z_1$ on the first principal direction")
    ax.set_ylabel("price (k\\$)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small")

    _save(fig, filename)


def plot_errors(degrees, train_mse, test_mse, val_mse, filename="2_errors.pdf"):
    """Training, validation (holdout) and test MSE over the polynomial degree, logarithmic y-axis (Task 2.2)."""
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(degrees, train_mse, "o-", color=COLORS[0], linewidth=2, label="training MSE")
    ax.plot(degrees, val_mse, "^-", color=COLORS[2], linewidth=2, label="validation MSE (holdout)")
    ax.plot(degrees, test_mse, "s-", color=COLORS[1], linewidth=2, label="test MSE")

    ax.set_yscale("log")
    ax.set_xlabel("polynomial degree $d$")
    ax.set_ylabel(r"MSE (k\$$^2$)")
    ax.set_xticks(degrees)
    ax.grid(True, alpha=0.3)
    ax.legend()

    _save(fig, filename)


def plot_cv(degrees, cv_mse, test_mse, d_cv, filename="3_cross_validation.pdf"):
    """Cross-validation MSE and test MSE over the polynomial degree and the selected degree (Task 3.2)."""
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(degrees, cv_mse, "o-", color=COLORS[2], linewidth=2, label="cross-validation MSE")
    ax.plot(degrees, test_mse, "s-", color=COLORS[1], linewidth=2, label="test MSE")
    ax.axvline(d_cv, color="gray", linestyle="--", linewidth=1.5, label=f"selected degree $d = {d_cv}$")

    ax.set_yscale("log")
    ax.set_xlabel("polynomial degree $d$")
    ax.set_ylabel(r"MSE (k\$$^2$)")
    ax.set_xticks(degrees)
    ax.grid(True, alpha=0.3)
    ax.legend()

    _save(fig, filename)
