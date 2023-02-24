import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys


def predict(coef, x):
    y = np.empty(x.shape)
    powers = np.arange(len(coef))
    for i in range(len(x)):
        y[i] = np.sum(coef * x[i]**powers)

    return y


def error(points, coef):
    total_error = (points[1] - predict(coef, points[0]))**2
    return np.mean(total_error)


def fit(points, coef, L):
    gradients = np.zeros(len(coef))
    x = points[0]
    y = points[1]
    for i in range(len(coef)):
        gradients[i] = - 2 * np.mean((y - predict(coef, x)) * x**i)
    coef = coef - gradients * L

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

    coef = np.zeros(degree + 1)

    # dataset
    data = pd.read_csv(f"../datasets/{dataset}")
    points = data.to_numpy().T
    print(points)

    # samples per il plot
    x_min = points[0].min()
    x_max = points[0].max()

    # effective learning rate (learning rate for each sample)
    L_eff = L / len(points[0])

    x_samples = np.linspace(x_min, x_max, 100)
    y_samples = predict(coef, x_samples)

    # fitting
    for i in range(epochs):
        coef = fit(points, coef, L_eff)

        if i % 50 == 0:
            x_samples = np.linspace(x_min, x_max, 100)
            y_samples = predict(coef, x_samples)

            # plot
            plt.clf()

            plt.title("Regressione Polinomiale")
            plt.grid()

            plt.scatter(points[0], points[1], color="black")
            plt.plot(x_samples, y_samples, color="red")

            plt.pause(0.00001)

    # risultato
    print(f"Errore: {error(points, coef)}")

    plt.show()
