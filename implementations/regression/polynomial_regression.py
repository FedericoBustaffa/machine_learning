import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys


def predict(coef, x):
    y = 0
    e = len(coef) - 1
    for c in coef:
        y += c * (x ** e)
        e -= 1

    return y


def error(points, coef):
    total_error = 0.0
    n = len(points[0])

    for i in range(n):
        x = points[0][i]
        y = points[1][i]
        total_error += (y - (predict(coef, x))) ** 2

    return total_error / float(n)


def fit(points, coef, L):
    n = len(points[0])

    gradients = [0] * len(coef)
    for i in range(n):
        x = points[0][i]
        y = points[1][i]

        e = len(coef) - 1
        for i in range(len(gradients)):
            gradients[i] += (y - predict(coef, x)) * (x**e)
            e -= 1

    for i in range(len(coef)):
        coef[i] = coef[i] + gradients[i] * L

    return coef


if __name__ == "__main__":

    if len(sys.argv) != 5:
        print(f"USAGE: py {sys.argv[0]} <degree> <L> <epochs> <dataset>")
        quit()

    # configurazione
    degree = int(sys.argv[1])
    L = float(sys.argv[2])
    epochs = int(sys.argv[3])
    dataset = sys.argv[4]

    coef = [0 for _ in range(degree + 1)]

    # dataset
    data = pd.read_csv(f"../datasets/{dataset}")
    x_values = data.x.values
    y_values = data.y.values
    points = np.array([x_values, y_values])

    # fitting
    x_min = points[0].min()
    x_max = points[0].max()
    y_min = points[1].min()
    y_max = points[1].max()

    for i in range(epochs):
        coef = fit(points, coef, L)

        if i % 50 == 0:
            x = np.linspace(x_min, x_max, 100)
            y = predict(coef, x)

            # plot
            plt.clf()

            plt.title("Regressione Polinomiale")
            plt.grid()

            plt.scatter(points[0], points[1], color="black")
            plt.plot(x, y, color="red")

            plt.pause(0.0001)

    # risultato
    print(f"Errore: {error(points, coef)}")

    plt.show()
