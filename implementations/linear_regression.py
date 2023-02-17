import numpy as np
import matplotlib.pyplot as plt
import random


def error(points, a, b):
    total_error = 0.0
    n = len(points[0])

    for i in range(n):
        x = points[0][i]
        y = points[1][i]
        total_error += (y - (a * x + b)) ** 2

    return total_error / float(n) 


def fit(points, a_now, b_now, L):
    n = len(points[0])

    a_gradient = 0.0
    b_gradient = 0.0

    for i in range(n):
        x = points[0][i]
        y = points[1][i]

        a_gradient += (y - (a_now * x + b_now)) * x
        b_gradient += (y - (a_now * x + b_now))

    new_a = a_now + a_gradient * L
    new_b = b_now + b_gradient * L

    return new_a, new_b


if __name__ == "__main__":
    # parametri iniziali
    a = 0
    b = 0
    L = 0.0005
    epochs = 3000

    # dataset
    x_values = np.linspace(1, 10, 10)
    y_values = [5 + np.random.randint(-10, 10) / 10 for i in x_values ]
    points = np.array([ x_values, y_values ])

    # fitting
    for i in range(epochs):
        a, b = fit(points, a, b, L)
        
        if i % 50 == 0:
            x = np.linspace(0, 11, 20)
            y = a * x + b
            
            # plot
            plt.clf()
            plt.title("Regressione Lineare")
            plt.xlim(0, 11)
            plt.ylim(0, 10)
            plt.scatter(points[0], points[1], color="black")
            plt.plot(x, y, color="red")
            plt.pause(0.0001)


    # risultato
    print(f"Coefficiente angolare: {a}")
    print(f"Quota: {b}")
    print(f"Errore: {error(points, a, b)}")
    
    plt.show()

