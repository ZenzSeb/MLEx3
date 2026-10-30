"""Exercise Sheet 03 -- PCA and Model Selection: Predicting House Prices.

Run all tasks:            python main.py
Run only selected tasks:  python main.py 1 3
Plots are saved to the folder plots/.

Fill in the TODOs in task1.py, task2.py and task3.py. You do not need to edit this file.
"""
import sys

import numpy as np

from task1 import task_1
from task2 import task_2
from task3 import task_3

SEED = 2026  # do not change

TASKS = {"1": task_1, "2": task_2, "3": task_3}


def main():
    selected = sys.argv[1:] or list(TASKS)

    for name in selected:
        if name not in TASKS:
            sys.exit(f"Unknown task '{name}', choose from: {', '.join(TASKS)}")

        print(f"\n========== Task {name} ==========")
        TASKS[name](np.random.default_rng(SEED))  # every task gets the same seed, so tasks can be run separately


if __name__ == "__main__":
    main()
