import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def error(points, a, b, c):
    total_error = 0.0
    n = len(points[0])

    for i in range(n):
        x = points[0][i]
        y = points[1][i]
        total_error += (y - (a * x * x + b * x + c)) ** 2

    return total_error / float(n)


def fit(points, a_now, b_now, c_now, L):
    n = len(points[0])

    a_gradient = 0
    b_gradient = 0
    c_gradient = 0

    for i in range(n):
        x = points[0][i]
        y = points[1][i]

        a_gradient += (y - (a_now * x * x + b_now * x + c)) * x * x
        b_gradient += (y - (a_now * x * x + b_now * x + c)) * x
        c_gradient += (y - (a_now * x * x + b_now * x + c))

    new_a = a_now + a_gradient * L
    new_b = b_now + b_gradient * L
    new_c = c_now + c_gradient * L

    return new_a, new_b, new_c


def predict(a, b, x):
    return a * x * x + b * x + c


if __name__ == "__main__":
    a = 0
    b = 0
    c = 0
    L = 0.05
    epochs = 5000

    # dataset
    data = pd.read_csv("dataset1.csv")
    x_values = data.x.values
    y_values = data.y.values
    points = np.array([x_values, y_values])

    # fitting
    for i in range(epochs):
        a, b, c = fit(points, a, b, c, L)

        if i % 75 == 0:
            x = np.linspace(0, 1.1, 50)
            y = predict(a, b, x)

            # plot
            plt.clf()
            plt.title("Regressione Polinomiale")
            plt.xlim(0, 1.1)
            plt.ylim(0, 1)
            plt.scatter(points[0], points[1], color="black")
            plt.plot(x, y, color="red")
            plt.pause(0.0001)

    # risultato
    print(f"A: {a}")
    print(f"B: {b}")
    print(f"C: {c}")
    print(f"Errore: {error(points, a, b, c)}")

    plt.show()
