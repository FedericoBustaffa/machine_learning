import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys


def predict(coef, x):
    y = np.empty(x.shape)
    powers = np.arange(len(coef))
    for i in range(len(x)):
        y[i] = np.sum(coef * x[i]**powers)
    #y = 0
    #e = len(coef) - 1
    #for c in coef:
    #    y += c * (x ** e)
    #    e -= 1
    return y


def error(points, coef):
    #total_error = 0.0
    #n = len(points[0])

    #for i in range(n):
    #    x = points[0][i]
    #    y = points[1][i]
    #    total_error += (y - (predict(coef, x))) ** 2

    total_error = (points[1] - predict(coef, points[0]))**2
    return np.mean(total_error)


def fit(points, coef, L):
    #n = len(points[0])

    gradients = np.zeros(len(coef))
    for i in range(len(coef)):
        gradients[i] = - 2 * np.mean( (points[1] - predict(coef, points[0])) * points[0]**i )
    #for i in range(n):
    #    x = points[0][i]
    #    y = points[1][i]

    #    e = len(coef) - 1
    #    for i in range(len(gradients)):
    #        gradients[i] += (y - predict(coef, x)) * (x**e)
    #        e -= 1

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

    #coef = [0 for _ in range(degree + 1)]
    coef = np.zeros(degree + 1)

    # dataset
    data = pd.read_csv(dataset)
    #x_values = data.x.values
    #y_values = data.y.values
    #points = np.array([x_values, y_values])
    points = data.to_numpy().T

    # samples per il plot
    x_min = points[0].min()
    x_max = points[0].max()

    # effective learning rate (learning rate for each sample)
    L_eff = L / len(points[0])

    plt.plot(np.linspace(x_min, x_max, 100), predict(coef,np.linspace(x_min, x_max, 100)))
    plt.grid()

    # fitting
    for i in range(epochs):
        coef = fit(points, coef, L_eff)

        if i % 50 == 0:
            x = np.linspace(x_min, x_max, 100)
            y = predict(coef, x)

            # plot
            plt.clf()

            plt.title("Regressione Polinomiale")
            plt.grid()

            plt.scatter(points[0], points[1], color="black")
            plt.plot(x, y, color="red")

            plt.pause(0.1)

    # risultato
    print(f"Errore: {error(points, coef)}")

    plt.show()
