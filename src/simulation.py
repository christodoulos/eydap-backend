import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from uuid import uuid1

matplotlib.use("Agg")


def create_random_chart(base_folder, num1, num2):
    # Create random data for the charts
    x_values = np.random.rand(10)
    y_values = np.random.rand(10)
    # Create first chart
    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, c="blue", label="Random Data")
    plt.title(f"Chart 1 with input numbers: {num1}, {num2}")
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{base_folder}/chart1.png")
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
    plt.title(f"Chart 2 with input numbers: {num1}, {num2}")
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{base_folder}/chart2.png")
    plt.close()  # Close the figure to free up memory


def run(user_id, num1, num2):
    new_uuid = uuid1()
    print(f"SAVE IN DB NEW UUID: {new_uuid}")
    base_folder = f"output/{user_id}/{new_uuid}"
    os.makedirs(base_folder, exist_ok=True)
    time.sleep(5)
    create_random_chart(base_folder, num1, num2)


# if __name__ == "__main__":
#     user_id, num1, num2 = sys.argv[1:]
#     run(user_id, num1, num2)
