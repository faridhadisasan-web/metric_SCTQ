from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(
    data: List[float],
    bins: int,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Union[str, Path],
) -> None:
    """Plot a histogram of data and save to file."""
    plt.figure()
    plt.hist(data, bins=bins, color="skyblue", edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(output_path)
    plt.close()
