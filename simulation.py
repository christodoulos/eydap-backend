import sys
import matplotlib.pyplot as plt
import numpy as np


def create_random_chart(x, y, z):
    # Create random data for the charts
    x_values = np.random.rand(10)
    y_values = np.random.rand(10)

    # Create first chart
    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, c="blue", label="Random Data")
    plt.title(f"Chart 1 with input numbers: {x}, {y}, {z}")
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.legend()
    plt.grid(True)
    plt.savefig("chart1.png")
    plt.close()  # Close the figure to free up memory

    # Create second chart
    plt.figure(figsize=(8, 6))
    plt.plot(
        x_values,
        y_values,
        color="green",
        linestyle="--",
        marker="o",
        label="Random Data",
    )
    plt.title(f"Chart 2 with input numbers: {x}, {y}, {z}")
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.legend()
    plt.grid(True)
    plt.savefig("chart2.png")
    plt.close()  # Close the figure to free up memory


if __name__ == "__main__":
    num1, num2, num3 = 1, 2, 3
    create_random_chart(num1, num2, num3)
